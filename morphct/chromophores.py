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


def create_supercell(chromo_list, box):
    for chromo in chromo_list:
        chromo._supercell_centers = []
        chromo._supercell_images = []
        for xyz_image in itertools.product(range(-1,2), repeat=3):
            xyz_image = np.array(xyz_image)
            chromo._supercell_centers.append(chromo.center + xyz_image * box)
            chromo._supercell_images.append(xyz_image)
    return chromo_list


def get_voronoi_neighbors(tri, all_chromos):
    n_list = defaultdict(set)
    for p in tri.vertices:
        for i, j in itertools.permutations(p, 2):
            n_list[all_chromos[i].periodic_id].add(all_chromos[j].periodic_id)
    return n_list


class SupercellChromo:
    def __init__(self):
        self.species = None
        self.original_id = None
        self.periodic_id = None
        self.center = None
        self.image = None


def update_chromo_list_voronoi(
    update_ids, sc_chromos, neighbor_ids, chromo_list
):
    # update_ids is a list of the periodic chromophores with image [0, 0, 0]
    for periodic_id in update_ids:
        # Obtain the real chromophore corresponding to this periodic_id
        chromo1 = chromo_list[sc_chromos[periodic_id].original_id]
        # Get latest neighbor information
        chromo1_neighbor_ids = [i for i,img in chromo1.neighbors]
        chromo1_d_neighbor_ids = [i for i,img in chromo1.dissociation_neighbors]
        for neighbor_p_id in neighbor_ids[periodic_id]:
            neighbor_sc_chromo = sc_chromos[neighbor_p_id]
            chromo2 = chromo_list[neighbor_sc_chromo.original_id]
            chromo2_neighbor_ids = [i for i,img in chromo2.neighbors]
            chromo2_d_neighbor_ids = [
                    i for i,img in chromo2.dissociation_neighbors
            ]
            relative_image = neighbor_sc_chromo.image
            if chromo1.species == chromo2.species:
                if chromo2.id not in chromo1_neighbor_ids:
                    chromo1.neighbors.append([chromo2.id, relative_image])
                    chromo1.neighbors_delta_e.append(None)
                    chromo1.neighbors_ti.append(None)
                    chromo1_neighbor_ids.append(chromo2.id)
                if chromo1.id not in chromo2_neighbor_ids:
                    chromo2.neighbors.append([chromo1.id, -relative_image])
                    chromo2.neighbors_delta_e.append(None)
                    chromo2.neighbors_ti.append(None)
                    chromo2_neighbor_ids.append(chromo1.id)
            else:
                if chromo2.id not in chromo1_d_neighbor_ids:
                    chromo1.dissociation_neighbors.append(
                        [chromo2.id, relative_image]
                    )
                    chromo1_d_neighbor_ids.append(chromo2.id)
                if chromo1.id not in chromo2_d_neighbor_ids:
                    chromo2.dissociation_neighbors.append(
                        [chromo1.id, -relative_image]
                    )
                    chromo2_d_neighbor_ids.append(chromo1.id)
    return chromo_list


def determine_neighbors_voronoi(chromo_list, snap):
    box = snap.configuration.box[:3]

    # First create the supercell
    supercell = create_supercell(chromo_list, box)
    donor_chromos = []
    acceptor_chromos = []
    all_chromos = []
    chromo_index = 0
    for chromo in supercell:
        for index, center in enumerate(chromo._supercell_centers):
            sc_chromo = SupercellChromo()
            sc_chromo.species = chromo.species
            sc_chromo.original_id = chromo.id
            sc_chromo.periodic_id = chromo_index
            sc_chromo.center = center
            sc_chromo.image = chromo._supercell_images[index]
            chromo_index += 1
            all_chromos.append(sc_chromo)

    # Now obtain the positions and send them to the Delaunay Triangulation
    # Then get the voronoi neighbors
    all_centers = [chromo.center for chromo in all_chromos]

    # Update the relevant neighbor dictionaries if we have the right
    # chromophore types in the system. Also log the chromophoreIDs from the
    # original simulation volume (non-periodic). Chromophores in the original
    # simulation volume will be every 27th (there are 27 periodic images in the
    # triple range(-1,2)), beginning from #13 ((0, 0, 0) is the thirteenth
    # element of the triple range(-1,2)) up to the length of the list in
    # question.
    original_all_chromo_ids = []

    # Default behaviour - carriers are blocked by the opposing species
    for chromo in all_chromos:
        if np.array_equal(chromo.image, [0, 0, 0]):
            original_all_chromo_ids.append(chromo.periodic_id)

    print("Calculating neighbours of all moieties")
    all_neighbors = get_voronoi_neighbors(Delaunay(all_centers), all_chromos)
    print("Updating the chromophore list for dissociation neighbors")
    chromo_list = update_chromo_list_voronoi(
        original_all_chromo_ids,
        all_chromos,
        all_neighbors,
        chromo_list
    )
    return chromo_list
