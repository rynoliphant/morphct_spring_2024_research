import copy
import itertools
import os
import sys
import numpy as np
from collections import defaultdict
from scipy.spatial import Delaunay
from morphct import helper_functions as hf


class Chromophore:
    def __init__(
        self,
        chromo_id,
        snap,
        atomic_ids,
        species,
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

        # get the chromo positions
        # qcc_input

        self.unwrapped_center, self.center, self.image = self._get_center(
                snap, atomic_ids
        )
        self.qcc_input = None
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

        # The neighbor_delta_E and neighbor_TI are lists where each element
        # describes the different in important molecular orbital or transfer
        # integral between this chromophore and each neighbor. The list indices
        # here are the same as in self.neighbors for coherence.
        self.neighbors_delta_E = []
        self.neighbors_TI = []

    def __repr__(self):
        return "Chromophore {} ({}): {} atoms at {:.3f} {:.3f} {:.3f}".format(
                self.id,
                self.species,
                len(self.atomic_ids),
                *self.center
                )

    def _get_center(self, snap, atomic_ids):
        box = snap.configuration.box[:3]
        unwrapped_pos = snap.particles.position + snap.particles.image * box
        center = np.mean(unwrapped_pos[atomic_ids], axis=0)
        img = np.zeros(3)
        while (center+img*box < -box/2).any() or (center+img*box > box/2).any():
            img[np.where(center < -box/2)] += 1
            img[np.where(center > box/2)] -= 1
        wrapped_center = center + img * box
        return center, wrapped_center, img

    def get_MO_energy(self):
        if self.species == "acceptor":
            return self.lumo
        elif self.species == "donor":
            return self.homo


def calculate_chromophores(
    CG_morphology_dict,
    AA_morphology_dict,
    CG_to_AAID_master,
    parameter_dict,
    sim_dims,
):
    # We make the assumption that a chromophore consists of one of each of the
    # CG site types described by the same template file. For instance, if we
    # have 3 sites 'A', 'B' and 'C' described in one file and one site 'D'
    # described in another file then there are two chromophores species
    # described by A-B-C and D. This will be treated automatically because the
    # D's shouldn't be bonded to anything in the CGMorphologyDict if they are
    # small molecules.
    # Therefore, we need to assign each CG site in the morphology to a
    # particular chromophore, so first, it's important to generate a
    # `neighbor_list' of all bonded atoms
    print("Determining chromophores in the system...")
    bonded_atoms = hf.obtain_bonded_list(CG_morphology_dict["bond"])
    chromophore_list = [i for i in range(len(CG_morphology_dict["type"]))]
    for CG_site_ID, chromophore_ID in enumerate(chromophore_list):
        CG_site_type = CG_morphology_dict["type"][CG_site_ID]
        types_in_this_chromophore = [CG_site_type]
        chromophore_list, types_in_this_chromophore = update_chromophores(
            CG_site_ID,
            chromophore_list,
            bonded_atoms,
            CG_morphology_dict["type"],
            types_in_this_chromophore,
            parameter_dict,
        )
    chromophore_data = {}
    for atom_ID, chromo_ID in enumerate(chromophore_list):
        if chromo_ID not in list(chromophore_data.keys()):
            chromophore_data[chromo_ID] = [atom_ID]
        else:
            chromophore_data[chromo_ID].append(atom_ID)
    # Now rename the chromophore IDs so that they increment sensibly (they will
    # be used later for the orca files)
    old_keys = sorted(chromophore_data.keys())
    for new_key, old_key in enumerate(old_keys):
        chromophore_data[new_key] = chromophore_data.pop(old_key)
    print(
        "{:d} chromophores successfully identified!".format(
            len(list(chromophore_data.keys()))
        )
    )
    # Now let's create a list of all the chromophore instances which contain all
    # of the information we could ever want about them.
    chromophore_instances = []
    for chromo_ID, chromophore_CG_sites in chromophore_data.items():
        print(
            "\rCalculating properties of chromophore {:05d} of {:05d}...".format(
                chromo_ID, len(list(chromophore_data.keys())) - 1
            ),
            end=" ",
        )
        if sys.stdout is not None:
            sys.stdout.flush()
        chromophore_instances.append(
            chromophore(
                chromo_ID,
                chromophore_CG_sites,
                CG_morphology_dict,
                AA_morphology_dict,
                CG_to_AAID_master,
                parameter_dict,
                sim_dims,
            )
        )
    print("")
    return chromophore_instances


def calculate_chromophores_AA(
    CG_morphology_dict,
    AA_morphology_dict,
    CG_to_AAID_master,
    parameter_dict,
    sim_dims,
    rigid_bodies=None,
):
    # If rigid_bodies == None:
    # This function works in the same way as the coarse-grained version above,
    # except this one iterates through the AA bonds instead. This is FAR SLOWER
    # and so shouldn't be done, except in the case where the coarse-grained
    # morphology does not exist (because we started with an atomistic morphology
    # and are only interested in running KMC on it)
    # If rigid_bodies == AA_morphology_dict['body']:
    # This function uses the rigid bodies specified in
    # parameter_dict['AA_rigid_body_species'], and those which have not been
    # specified by iterating through the AA bond list, to determine the
    # chromophores in the system. This is the slowest way to calculate
    # chromophores, but is useful for systems such as BDT-TPD, where there are
    # multiple chromophores of differing species present in the same molecule.
    # As above, this code will only run if an atomistic morphology has been
    # input to MorphCT. If it is coarse-grained, the CG-based
    # "calculate_chromophore" function will be used, and will also be a lot
    # faster.
    # The parameter_dict['AA_rigid_body_species'] is a dictionary with two keys,
    # 'donor' or 'acceptor'. Each element in the value list corresponds to a new
    # chromophore. These aren't the only atoms that belong to this chromophore,
    # however - there might be a bunch of aliphatic/flexible atoms that are
    # connected, so we need to make sure that we add those too.
    print("Determining chromophores in the system...")
    bonded_atoms = hf.obtain_bonded_list(AA_morphology_dict["bond"])
    chromophore_list = [i for i in range(len(AA_morphology_dict["type"]))]
    for AA_site_ID, chromophore_ID in enumerate(chromophore_list):
        AA_site_type = AA_morphology_dict["type"][AA_site_ID]
        chromophore_list = update_chromophores_AA(
            AA_site_ID,
            chromophore_list,
            bonded_atoms,
            parameter_dict,
            rigid_bodies,
        )
    chromophore_data = {}
    for atom_ID, chromo_ID in enumerate(chromophore_list):
        if chromo_ID not in list(chromophore_data.keys()):
            chromophore_data[chromo_ID] = [atom_ID]
        else:
            chromophore_data[chromo_ID].append(atom_ID)
    # Now rename the chromophore IDs so that they increment sensibly (they will
    # be used later for the orca files)
    old_keys = sorted(chromophore_data.keys())
    for new_key, old_key in enumerate(old_keys):
        chromophore_data[new_key] = chromophore_data.pop(old_key)
    print(
        "{:d} chromophores successfully identified!".format(
            len(list(chromophore_data.keys()))
        )
    )
    # Now let's create a list of all the chromophore instances which contain all
    # of the information we could ever want about them.
    chromophore_instances = []
    for chromo_ID, chromophore_CG_sites in chromophore_data.items():
        print(
            "\rCalculating properties of chromophore {:05d} of {:05d}...".format(
                chromo_ID, len(list(chromophore_data.keys())) - 1
            ),
            end=" ",
        )
        if sys.stdout is not None:
            sys.stdout.flush()
        chromophore_instances.append(
            chromophore(
                chromo_ID,
                chromophore_CG_sites,
                CG_morphology_dict,
                AA_morphology_dict,
                CG_to_AAID_master,
                parameter_dict,
                sim_dims,
            )
        )
    print("")
    return chromophore_instances


def update_chromophores(
    atom_ID,
    chromophore_list,
    bonded_atoms,
    CG_type_list,
    types_in_this_chromophore,
    parameter_dict,
):
    # Recursively add all neighbors of atom number atom_ID to this chromophore,
    # providing the same type does not already exist in it
    try:
        for bonded_atom in bonded_atoms[atom_ID]:
            bonded_type = CG_type_list[bonded_atom]
            # First check that the bonded_atom's type is not already in this
            # chromophore.
            # Also, check that the type to be added is of the same electronic
            # species as the ones added previously, or == 'None'
            if (bonded_type not in types_in_this_chromophore) and (
                (
                    parameter_dict["CG_site_species"][bonded_type].lower()
                    == "none"
                )
                or (
                    parameter_dict["CG_site_species"][bonded_type].lower()
                    == list(
                        set(
                            [
                                parameter_dict["CG_site_species"][x].lower()
                                for x in types_in_this_chromophore
                            ]
                        )
                    )[0]
                )
            ):
                # If the atomID of the bonded atom is larger than that of the
                # current one, update the bonded atom's ID to the current one's
                # to put it in this chromophore, then iterate through all of the
                # bonded atom's neighbors
                if chromophore_list[bonded_atom] > chromophore_list[atom_ID]:
                    chromophore_list[bonded_atom] = chromophore_list[atom_ID]
                    types_in_this_chromophore.append(bonded_type)
                    chromophore_list, types_in_this_chromophore = update_chromophores(
                        bonded_atom,
                        chromophore_list,
                        bonded_atoms,
                        CG_type_list,
                        types_in_this_chromophore,
                        parameter_dict,
                    )
                # If the atomID of the current atom is larger than that of the
                # bonded one, update the current atom's ID to the bonded one's
                # to put it in this chromophore, then iterate through all of the
                # current atom's neighbors
                elif chromophore_list[bonded_atom] < chromophore_list[atom_ID]:
                    chromophore_list[atom_ID] = chromophore_list[bonded_atom]
                    types_in_this_chromophore.append(CG_type_list[atom_ID])
                    chromophore_list, types_in_this_chromophore = update_chromophores(
                        atom_ID,
                        chromophore_list,
                        bonded_atoms,
                        CG_type_list,
                        types_in_this_chromophore,
                        parameter_dict,
                    )
                # Else: both the current and the bonded atom are already known
                # to be in this chromophore, so we don't have to do anything
                # else.
    except KeyError:
        # This means that there are no bonded CG sites (i.e. it's a single
        # chromophore)
        pass
    return chromophore_list, types_in_this_chromophore


def update_chromophores_AA(
    atom_ID, chromophore_list, bonded_atoms, parameter_dict, rigid_bodies=None
):
    # This version of the update chromophores function does not check for CG
    # site types, instead just adding all bonded atoms. Therefore it should only
    # be used in the case of already-atomistic morphologies (no CG morph
    # specified) containing ONLY small molecules
    try:
        for bonded_atom in bonded_atoms[atom_ID]:
            if rigid_bodies is not None:
                # Skip if the bonded atom belongs to a different rigid body
                if (
                    (rigid_bodies[bonded_atom] != -1)
                    and (rigid_bodies[atom_ID] != -1)
                ) and (rigid_bodies[bonded_atom] != rigid_bodies[atom_ID]):
                    continue
            # If the atomID of the bonded atom is larger than that of the
            # current one, update the bonded atom's ID to the current one's to
            # put it in this chromophore, then iterate through all of the bonded
            # atom's neighbors
            if chromophore_list[bonded_atom] > chromophore_list[atom_ID]:
                chromophore_list[bonded_atom] = chromophore_list[atom_ID]
                chromophore_list = update_chromophores_AA(
                    bonded_atom,
                    chromophore_list,
                    bonded_atoms,
                    parameter_dict,
                    rigid_bodies,
                )
            # If the atomID of the current atom is larger than that of the
            # bonded one, update the current atom's ID to the bonded one's to
            # put it in this chromophore, then iterate through all of the
            # current atom's neighbors
            elif chromophore_list[bonded_atom] < chromophore_list[atom_ID]:
                chromophore_list[atom_ID] = chromophore_list[bonded_atom]
                chromophore_list = update_chromophores_AA(
                    atom_ID,
                    chromophore_list,
                    bonded_atoms,
                    parameter_dict,
                    rigid_bodies,
                )
            # Else: both the current and the bonded atom are already known to be
            # in this chromophore, so we don't have to do anything else.
    except KeyError:
        # This means that there are no bonded CG sites (i.e. it's a single
        # chromophore)
        pass
    return chromophore_list


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


class super_cell_chromo:
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


def chromo_sort(chromophore_list):
    for index, chromo in enumerate(chromophore_list):
        if index != chromo.ID:
            print(
                "Inconsistency found in the ordering of the chromophore_list, rewriting"
                " the chromophore_list in the correct order..."
            )
            new_chromophore_list = []
            for chromo in chromophore_list:
                new_chromophore_list.append(0)
            for chromo in chromophore_list:
                new_chromophore_list[chromo.ID] = chromo
            chromophore_list = new_chromophore_list
            return chromophore_list
    return chromophore_list


def main(
    AA_morphology_dict,
    CG_morphology_dict,
    CG_to_AAID_master,
    parameter_dict,
    chromophore_list,
):
    sim_dims = [
        [-AA_morphology_dict["lx"] / 2.0, AA_morphology_dict["lx"] / 2.0],
        [-AA_morphology_dict["ly"] / 2.0, AA_morphology_dict["ly"] / 2.0],
        [-AA_morphology_dict["lz"] / 2.0, AA_morphology_dict["lz"] / 2.0],
    ]
    if len(parameter_dict["CG_to_template_dirs"]) > 0:
        # Normal operation using the coarse-grained morphology
        chromophore_list = calculate_chromophores(
            CG_morphology_dict,
            AA_morphology_dict,
            CG_to_AAID_master,
            parameter_dict,
            sim_dims,
        )
    elif (len(parameter_dict["CG_site_species"]) == 1) and (
        len(parameter_dict["AA_rigid_body_species"]) == 0
    ):
        # Small molecule system with only one electronic species
        chromophore_list = calculate_chromophores_AA(
            CG_morphology_dict,
            AA_morphology_dict,
            CG_to_AAID_master,
            parameter_dict,
            sim_dims,
        )
    else:
        # Other system, with electronically active species specified as rigid
        # bodies using AA_rigid_body_species in parameter file
        chromophore_list = calculate_chromophores_AA(
            CG_morphology_dict,
            AA_morphology_dict,
            CG_to_AAID_master,
            parameter_dict,
            sim_dims,
            rigid_bodies=AA_morphology_dict["body"],
        )
    chromophore_list = chromo_sort(chromophore_list)
    if parameter_dict["use_voronoi_neighbors"] is True:
        chromophore_list = determine_neighbors_voronoi(
            chromophore_list, parameter_dict, sim_dims
        )
    else:
        chromophore_list = determine_neighbors_cut_off(
            chromophore_list, parameter_dict, sim_dims
        )
    # Now we have updated the chromophore_list, rewrite the pickle with this new
    # information.
    pickle_name = os.path.join(
        parameter_dict["output_morphology_directory"],
        "code",
        "".join([os.path.splitext(parameter_dict["morphology"])[0], ".pickle"]),
    )
    hf.write_pickle(
        (
            AA_morphology_dict,
            CG_morphology_dict,
            CG_to_AAID_master,
            parameter_dict,
            chromophore_list,
        ),
        pickle_name,
    )
    return (
        AA_morphology_dict,
        CG_morphology_dict,
        CG_to_AAID_master,
        parameter_dict,
        chromophore_list,
    )


if __name__ == "__main__":
    try:
        pickle_file = sys.argv[1]
    except:
        print(
            "Please specify the pickle file to load to continue the pipeline from this"
            " point."
        )
    pickle_data = hf.load_pickle(pickle_file)
    AA_morphology_dict = pickle_data[0]
    CG_morphology_dict = pickle_data[1]
    CG_to_AAID_master = pickle_data[2]
    parameter_dict = pickle_data[3]
    chromophore_list = pickle_data[4]
    main(
        AA_morphology_dict,
        CG_morphology_dict,
        CG_to_AAID_master,
        parameter_dict,
        chromophore_list,
    )
