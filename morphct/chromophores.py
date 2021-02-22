import copy
import itertools
from collections import defaultdict
import os
import sys

import freud
from openbabel import openbabel
from openbabel import pybel
import numpy as np
from scipy.spatial import Delaunay

from morphct import execute_qcc as eqcc
from morphct import helper_functions as hf


class Chromophore:
    """
    Attributes
    ----------

    Methods
    -------
    """
    def __init__(
        self,
        chromo_id,
        snap,
        atom_ids,
        species,
        conversion_dict,
        reorganization_energy = 0.3064,
        vrh_delocalization = 2e-10
    ):
        """
        Parameters
        ----------

        """
        self.id = chromo_id
        if species.lower() not in ["donor", "acceptor"]:
            raise TypeError("Species must be either donor or acceptor")
        self.species = species.lower()
        self.reorganization_energy = reorganization_energy
        self.vrh_delocalization = vrh_delocalization
        self.atom_ids = atom_ids
        self.n_atoms = len(atom_ids)

        # Sets unwrapped_center, center, and image attributes
        self._set_center(snap, atom_ids)

        self.qcc_input = eqcc.write_qcc_inp(snap, atom_ids, conversion_dict)

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
                self.n_atoms,
                *self.center
                )

    def _set_center(self, snap, atom_ids):
        box = snap.configuration.box[:3]
        unwrapped_pos = snap.particles.position + snap.particles.image * box
        center = np.mean(unwrapped_pos[atom_ids], axis=0)
        img = np.zeros(3)
        while (center+img*box < -box/2).any() or (center+img*box > box/2).any():
            img[np.where(center < -box/2)] += 1
            img[np.where(center > box/2)] -= 1
        self.unwrapped_center = center
        self.center = center + img * box
        self.image = img

    def get_MO_energy(self):
        """
        Parameters
        ----------

        Returns
        -------

        """
        if self.species == "acceptor":
            return self.lumo
        elif self.species == "donor":
            return self.homo


def get_chromo_ids_smiles(snap, smarts_str, conversion_dict):
    """
    Parameters
    ----------

    Returns
    -------

    """
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
    atom_ids = [np.array(i)-1 for i in smarts.findall(pybelmol)]
    return atom_ids


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


def set_neighbors_voronoi(chromo_list, snap, conversion_dict, d_cut=10):
    """
    Parameters
    ----------

    Returns
    -------

    """
    voronoi = freud.locality.Voronoi()
    freudbox = freud.box.Box(*snap.configuration.box)
    centers = [chromo.center for chromo in chromo_list]
    voronoi.compute((freudbox, centers))

    box = snap.configuration.box[:3]
    qcc_pairs = []
    neighbors = []
    for (i,j) in voronoi.nlist:
        if (i,j) not in neighbors and (j,i) not in neighbors:
            chromo_i = chromo_list[i]
            chromo_j = chromo_list[j]
            centers = []
            distances = []
            images = []
            # calculate which of the periodic image is closest
            # shift chromophore j, hold chromophore i in place
            for xyz_image in itertools.product(range(-1,2), repeat=3):
                xyz_image = np.array(xyz_image)
                sc_center = chromo_j.center + xyz_image * box
                images.append(xyz_image)
                centers.append(sc_center)
                distances.append(np.linalg.norm(sc_center - chromo_i.center))
            imin = distances.index(min(distances))
            if distances[imin] > d_cut:
                continue

            rel_image = images[imin]
            j_shift = centers[imin] - chromo_j.unwrapped_center
            chromo_i.neighbors.append([j, rel_image])
            chromo_i.neighbors_delta_e.append(None)
            chromo_i.neighbors_ti.append(None)
            chromo_j.neighbors.append([i, -rel_image])
            chromo_j.neighbors_delta_e.append(None)
            chromo_j.neighbors_ti.append(None)
            neighbors.append((i,j))
            qcc_input = eqcc.write_qcc_pair_input(
                snap, chromo_i, chromo_j, j_shift, conversion_dict
                )
            qcc_pairs.append(((i,j), qcc_input))
    return qcc_pairs
