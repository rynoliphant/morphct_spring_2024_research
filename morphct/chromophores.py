import copy
import itertools
from collections import defaultdict
import os
import sys

from openbabel import openbabel
from openbabel import pybel
import numpy as np
from scipy.spatial import Delaunay

from morphct import execute_qcc as eqcc
from morphct import helper_functions as hf


class Chromophore:
    def __init__(
        self,
        chromo_id,
        snap,
        atomic_ids,
        species,
        conversion_dict,
        reorganization_energy = 0.3064,
        vrh_delocalization = 2e-10
    ):
        self.id = chromo_id
        if species.lower() not in ["donor", "acceptor"]:
            raise TypeError("Species must be either donor or acceptor")
        self.species = species.lower()
        self.reorganization_energy = reorganization_energy
        self.vrh_delocalization = vrh_delocalization
        self.atomic_ids = atomic_ids

        # Sets unwrapped_center, center, and image attributes
        self._set_center(snap, atomic_ids)

        self.qcc_input = eqcc.write_qcc_inp(snap, atomic_ids, conversion_dict)

        # Now to create a load of placeholder parameters to update later when we
        # have the full list/energy levels.
        # The self.neighbors list contains one element for each chromophore
        # within parameterDict['maximum_hop_distance'] of this one (including
        # periodic boundary conditions). Its format is
        # [[neighbor1_ID, relative_image_of_neighbor1],...]
        self.neighbors = []
        self.dissociation_neighbors = []
        # The molecular orbitals of this chromophore have not yet been
        # calculated, but they will simply be floats.
        self.homo = None
        self.homo_1 = None
        self.lumo = None
        self.lumo_1 = None

        # The neighbor_delta_e and neighbor_ti are lists where each element
        # describes the different in important molecular orbital or transfer
        # integral between this chromophore and each neighbor. The list indices
        # here are the same as in self.neighbors for coherence.
        self.neighbors_delta_e = []
        self.neighbors_ti = []

    def __repr__(self):
        return "Chromophore {} ({}): {} atoms at {:.3f} {:.3f} {:.3f}".format(
                self.id,
                self.species,
                len(self.atomic_ids),
                *self.center
                )

    def _set_center(self, snap, atomic_ids):
        box = snap.configuration.box[:3]
        unwrapped_pos = snap.particles.position + snap.particles.image * box
        center = np.mean(unwrapped_pos[atomic_ids], axis=0)
        img = np.zeros(3)
        while (center+img*box < -box/2).any() or (center+img*box > box/2).any():
            img[np.where(center < -box/2)] += 1
            img[np.where(center > box/2)] -= 1
        self.unwrapped_center = center
        self.center = center + img * box
        self.image = img

    def get_MO_energy(self):
        if self.species == "acceptor":
            return self.lumo
        elif self.species == "donor":
            return self.homo


def get_chromo_ids_smiles(snap, smarts_str, conversion_dict):
    mol = openbabel.OBMol()
    for i in range(snap.particles.N):
        a = mol.NewAtom()
        element = conversion_dict[
                snap.particles.types[snap.particles.typeid[i]]
                ]
        a.SetAtomicNum(element.atomic_number)
        a.SetVector(*[float(x) for x in snap.particles.position[i]])

    for i,j in snap.bonds.group:
        # openbabel indexes atoms from 1
        # AddBond(i_index, j_index, bond_order)
        mol.AddBond(int(i+1), int(j+1), 1)

    pybelmol = pybel.Molecule(mol)
    # This will correctly set the bond order
    # (necessary for smarts matching)
    pybelmol.OBMol.PerceiveBondOrders()

    smarts = pybel.Smarts(smarts_str)
    # shift indices by 1
    atomic_ids = [np.array(i)-1 for i in smarts.findall(pybelmol)]
    return atomic_ids


def create_super_cell(chromophore_list, box_size):
    for chromophore in chromophore_list:
        chromophore.super_cell_posns = []
        chromophore.super_cell_images = []
        for x_image in range(-1, 2):
            for y_image in range(-1, 2):
                for z_image in range(-1, 2):
                    chromophore.super_cell_posns.append(
                        np.array(chromophore.posn)
                        + (
                            np.array([x_image, y_image, z_image])
                            * (np.array(box_size))
                        )
                    )
                    chromophore.super_cell_images.append(
                        np.array([x_image, y_image, z_image])
                    )
    return chromophore_list


def get_voronoi_neighbors(tri, chromo_list):
    n_list = defaultdict(set)
    for p in tri.vertices:
        for i, j in itertools.permutations(p, 2):
            n_list[chromo_list[i].periodic_ID].add(chromo_list[j].periodic_ID)
    return n_list


class SupercellChromo:
    def __init__(self):
        self.species = None
        self.original_ID = None
        self.periodic_ID = None
        self.position = None
        self.image = None


def update_chromophore_list_voronoi(
    IDs_to_update, super_cell_chromos, neighbor_IDs, chromophore_list, sim_dims
):
    # IDs to Update is a list of the periodic chromophores with the image
    # [0, 0, 0]
    for periodic_ID in IDs_to_update:
        # Obtain the real chromophore corresponding to this periodic_ID
        chromophore1 = chromophore_list[
            super_cell_chromos[periodic_ID].original_ID
        ]
        assert np.array_equal(super_cell_chromos[periodic_ID].image, [0, 0, 0])
        # Get latest neighbor information
        chromo1neighbor_IDs = [
            neighbor_data[0] for neighbor_data in chromophore1.neighbors
        ]
        chromo1dissociation_neighbor_IDs = [
            neighbor_data[0]
            for neighbor_data in chromophore1.dissociation_neighbors
        ]
        for neighbor_periodic_ID in neighbor_IDs[periodic_ID]:
            neighbor_super_cell_chromo = super_cell_chromos[
                neighbor_periodic_ID
            ]
            chromophore2 = chromophore_list[
                neighbor_super_cell_chromo.original_ID
            ]
            chromo2neighbor_IDs = [
                neighbor_data[0] for neighbor_data in chromophore2.neighbors
            ]
            chromo2dissociation_neighbor_IDs = [
                neighbor_data[0]
                for neighbor_data in chromophore2.dissociation_neighbors
            ]
            relative_image = neighbor_super_cell_chromo.image
            if chromophore1.species == chromophore2.species:
                if chromophore2.ID not in chromo1neighbor_IDs:
                    chromophore1.neighbors.append(
                        [chromophore2.ID, list(np.array(relative_image))]
                    )
                    chromophore1.neighbors_delta_E.append(None)
                    chromophore1.neighbors_TI.append(None)
                    chromo1neighbor_IDs.append(chromophore2.ID)
                if chromophore1.ID not in chromo2neighbor_IDs:
                    chromophore2.neighbors.append(
                        [chromophore1.ID, list(-np.array(relative_image))]
                    )
                    chromophore2.neighbors_delta_E.append(None)
                    chromophore2.neighbors_TI.append(None)
                    chromo2neighbor_IDs.append(chromophore1.ID)
            else:
                if chromophore2.ID not in chromo1dissociation_neighbor_IDs:
                    chromophore1.dissociation_neighbors.append(
                        [chromophore2.ID, list(np.array(relative_image))]
                    )
                    chromo1dissociation_neighbor_IDs.append(chromophore2.ID)
                if chromophore1.ID not in chromo2dissociation_neighbor_IDs:
                    chromophore2.dissociation_neighbors.append(
                        [chromophore1.ID, list(-np.array(relative_image))]
                    )
                    chromo2dissociation_neighbor_IDs.append(chromophore1.ID)
    return chromophore_list


def determine_neighbors_voronoi(chromophore_list, parameter_dict, sim_dims):
    box_size = [axis[1] - axis[0] for axis in sim_dims]
    # First create the supercell
    super_cell = create_super_cell(chromophore_list, box_size)
    donor_chromos = []
    acceptor_chromos = []
    all_chromos = []
    chromo_index = 0
    for chromophore in super_cell:
        for index, position in enumerate(chromophore.super_cell_posns):
            chromo = super_cell_chromo()
            chromo.species = chromophore.species
            chromo.original_ID = chromophore.ID
            chromo.periodic_ID = chromo_index
            chromo.position = position
            chromo.image = chromophore.super_cell_images[index]
            chromo_index += 1
            if chromophore.species.lower() == "donor":
                donor_chromos.append(chromo)
            elif chromophore.species.lower() == "acceptor":
                acceptor_chromos.append(chromo)
            all_chromos.append(chromo)
    # Now obtain the positions and send them to the Delaunay Triangulation
    # Then get the voronoi neighbors
    all_positions = [chromo.position for chromo in all_chromos]
    # Initialise the neighbor dictionaries
    all_neighbors = {}
    # Update the relevant neighbor dictionaries if we have the right
    # chromophore types in the system. Also log the chromophoreIDs from the
    # original simulation volume (non-periodic). Chromophores in the original
    # simulation volume will be every 27th (there are 27 periodic images in the
    # triple range(-1,2)), beginning from #13 ((0, 0, 0) is the thirteenth
    # element of the triple range(-1,2)) up to the length of the list in
    # question.
    original_all_chromo_IDs = []
    try:
        if parameter_dict["permit_hops_through_opposing_chromophores"]:
            # Need to only consider the neighbors of like chromophore species
            donor_positions = [chromo.position for chromo in donor_chromos]
            acceptor_positions = [
                chromo.position for chromo in acceptor_chromos
            ]
            donor_neighbors = {}
            acceptor_neighbors = {}
            original_donor_chromo_IDs = []
            original_acceptor_chromo_IDs = []
            for chromophore in all_chromos:
                if np.array_equal(chromophore.image, [0, 0, 0]):
                    original_all_chromo_IDs.append(chromophore.periodic_ID)
                    if chromophore.species.lower() == "donor":
                        original_donor_chromo_IDs.append(
                            chromophore.periodic_ID
                        )
                    elif chromophore.species.lower() == "acceptor":
                        original_acceptor_chromo_IDs.append(
                            chromophore.periodic_ID
                        )
            if len(donor_positions) > 0:
                print("Calculating Neighbours of donor Moieties")
                donor_neighbors = get_voronoi_neighbors(
                    Delaunay(donor_positions), donor_chromos
                )
                print("Updating the chromophore list for donor chromos")
                chromophore_list = update_chromophore_list_voronoi(
                    original_donor_chromo_IDs,
                    all_chromos,
                    donor_neighbors,
                    chromophore_list,
                    sim_dims,
                )
            if len(acceptor_positions) > 0:
                print("Calculating Neighbours of acceptor Moieties")
                acceptor_neighbors = get_voronoi_neighbors(
                    Delaunay(acceptor_positions), acceptor_chromos
                )
                print("Updating the chromophore list for acceptor chromos")
                chromophore_list = update_chromophore_list_voronoi(
                    original_acceptor_chromo_IDs,
                    all_chromos,
                    acceptor_neighbors,
                    chromophore_list,
                    sim_dims,
                )
        else:
            raise KeyError
    except KeyError:
        # Default behaviour - carriers are blocked by the opposing species
        for chromophore in all_chromos:
            if np.array_equal(chromophore.image, [0, 0, 0]):
                original_all_chromo_IDs.append(chromophore.periodic_ID)
    print("Calculating Neighbours of All Moieties")
    all_neighbors = get_voronoi_neighbors(
        Delaunay(all_positions), all_chromos
    )
    print("Updating the chromophore list for dissociation neighbors")
    chromophore_list = update_chromophore_list_voronoi(
        original_all_chromo_IDs,
        all_chromos,
        all_neighbors,
        chromophore_list,
        sim_dims,
    )
    return chromophore_list


def determine_neighbors_cut_off(chromophore_list, parameter_dict, sim_dims):
    for chromophore1 in chromophore_list:
        print(
            "\rIdentifying neighbors of chromophore {:05d} of {:05d}...".format(
                chromophore1.ID, len(chromophore_list) - 1
            ),
            end=" ",
        )
        if sys.stdout is not None:
            sys.stdout.flush()
        for chromophore2 in chromophore_list:
            # Skip if chromo2 is chromo1
            if chromophore1.ID == chromophore2.ID:
                continue
            delta_posn = chromophore2.posn - chromophore1.posn
            relative_image_of_chromo2 = [0, 0, 0]
            # Consider periodic boundary conditions
            for axis in range(3):
                half_box_length = (sim_dims[axis][1] - sim_dims[axis][0]) / 2.0
                while delta_posn[axis] > half_box_length:
                    delta_posn[axis] -= sim_dims[axis][1] - sim_dims[axis][0]
                    relative_image_of_chromo2[axis] -= 1
                while delta_posn[axis] < -half_box_length:
                    delta_posn[axis] += sim_dims[axis][1] - sim_dims[axis][0]
                    relative_image_of_chromo2[axis] += 1
            separation = np.linalg.norm(delta_posn)
            # If proximity is within tolerance, add these chromophores as
            # neighbors. Base check is against the maximum of the donor and
            # acceptor hop distances. A further separation check is made if the
            # chromophores are the same type to make sure we don't exceed the
            # maximum specified hop distance for the carrier type.
            if separation <= max(
                [
                    parameter_dict["maximum_hole_hop_distance"],
                    parameter_dict["maximum_electron_hop_distance"],
                ]
            ):
                # Only add the neighbors if they haven't already been added so
                # far
                chromo1neighbor_IDs = [
                    neighbor_data[0]
                    for neighbor_data in chromophore1.neighbors
                ]
                chromo2neighbor_IDs = [
                    neighbor_data[0]
                    for neighbor_data in chromophore2.neighbors
                ]
                chromo1dissociation_neighbor_IDs = [
                    neighbor_data[0]
                    for neighbor_data in chromophore1.dissociation_neighbors
                ]
                chromo2dissociation_neighbor_IDs = [
                    neighbor_data[0]
                    for neighbor_data in chromophore2.dissociation_neighbors
                ]
                # Also, make the delta_E and the T_ij lists as long as the
                # neighbor lists for easy access later
                if chromophore1.species == chromophore2.species:
                    if (
                        (chromophore1.species.lower() == "donor")
                        and (
                            separation
                            >= parameter_dict["maximum_hole_hop_distance"]
                        )
                    ) or (
                        (chromophore1.species.lower() == "acceptor")
                        and (
                            separation
                            >= parameter_dict["maximum_electron_hop_distance"]
                        )
                    ):
                        continue
                    if chromophore2.ID not in chromo1neighbor_IDs:
                        chromophore1.neighbors.append(
                            [chromophore2.ID, relative_image_of_chromo2]
                        )
                        chromophore1.neighbors_delta_E.append(None)
                        chromophore1.neighbors_TI.append(None)
                    if chromophore1.ID not in chromo2neighbor_IDs:
                        chromophore2.neighbors.append(
                            [
                                chromophore1.ID,
                                list(-np.array(relative_image_of_chromo2)),
                            ]
                        )
                        chromophore2.neighbors_delta_E.append(None)
                        chromophore2.neighbors_TI.append(None)
                else:
                    # NOTE: Modifying this so that only dissociation neigbours in the
                    # same periodic image are considered.
                    if (
                        chromophore2.ID not in chromo2dissociation_neighbor_IDs
                    ) and (
                        np.all(np.isclose(relative_image_of_chromo2, [0, 0, 0]))
                    ):
                        chromophore1.dissociation_neighbors.append(
                            [chromophore2.ID, [0, 0, 0]]
                        )
                    if (
                        chromophore1.ID not in chromo1dissociation_neighbor_IDs
                    ) and (
                        np.all(np.isclose(relative_image_of_chromo2, [0, 0, 0]))
                    ):
                        chromophore2.dissociation_neighbors.append(
                            [chromophore1.ID, [0, 0, 0]]
                        )
    print("")
    return chromophore_list
