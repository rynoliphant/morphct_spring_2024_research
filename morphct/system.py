import os
import time

import gsd, gsd.hoomd
import numpy as np

from morphct.chromophores import Chromophore, set_neighbors_voronoi
from morphct.execute_qcc import (
    singles_homolumo, dimer_homolumo, set_energyvalues
)
from morphct.mobility_kmc import run_kmc
from morphct import kmc_analyze


class System():
    """An object for containing the system.

    Parameters
    ----------
    gsdfile
    outpath
    frame
    scale
    conversion_dict

    Attributes
    ----------
    snap
    conversion_dict
    chromophores

    Methods
    -------
    """

    def __init__(
        self, gsdfile, outpath, frame=-1, scale=1.0, conversion_dict=None,
    ):
        with gsd.hoomd.open(name=gsdfile, mode="rb") as f:
            snap = f[frame]

        # It is expected that the snapshot is in the Angstrom length scale.
        # If not, the scaling factor is used to adjust
        snap.particles.position *= scale
        snap.configuration.box[:3] *= scale

        self.snap = snap
        self.conversion_dict = conversion_dict
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        self.outpath = outpath
        self._chromophores = []
        self._comp = None
        self.qcc_pairs = None
        self._dinds = []
        self._ainds = []

    @property
    def chromophores(self):
        """"""
        return self._chromophores

    def add_chromophores(self, indices, species, chromophore_kwargs={}):
        """

        Parameters
        ----------

        """
        start = len(self.chromophores)
        for i, ind in enumerate(indices):
            self._chromophores.append(
                Chromophore(
                    i+start,
                    self.snap,
                    ind,
                    species,
                    self.conversion_dict,
                    **chromophore_kwargs
                )
            )
        if species == "acceptor":
            self._ainds += indices
        else:
            self._dinds += indices

    def compute_energies(self, dcut=None):
        """

        Parameters
        ----------

        """
        if dcut is None:
            dcut = min(self.snap.configuration.box[:3]/2)
        self.qcc_pairs = set_neighbors_voronoi(
            self.chromophores, self.snap, self.conversion_dict, d_cut=dcut
        )
        print(f"There are {len(self.qcc_pairs)} chromophore pairs")

        s_filename = os.path.join(self.outpath, "singles_energies.txt")
        d_filename = os.path.join(self.outpath, "dimer_energies.txt")

        t0 = time.perf_counter()
        print("Starting singles energy calculation...")
        data = singles_homolumo(self.chromophores, s_filename)
        t1 = time.perf_counter()
        print(f"Finished in {t1-t0:.2f} s. Output written to {s_filename}.")

        print("Starting dimer energy calculation...")
        dimer_data = dimer_homolumo(
            self.qcc_pairs, self.chromophores, d_filename
        )
        t2 = time.perf_counter()
        print(f"Finished in {t2-t1:.2f} s. Output written to {d_filename}.")

    def set_energies(self, s_filename, d_filename, dcut=None):
        """

        Parameters
        ----------

        """
        if self.qcc_pairs is None:
            if dcut is None:
                dcut = min(self.snap.configuration.box[:3]/2)
            self.qcc_pairs = set_neighbors_voronoi(
                self.chromophores, self.snap, self.conversion_dict, d_cut=dcut
            )
        set_energyvalues(self.chromophores, s_filename, d_filename)
        print("Energies set.")

    def run_kmc(
        self,
        lifetimes,
        temp,
        n_holes=0,
        n_elec=0,
        seed=42,
        combine=True,
        carrier_kwargs={},
        verbose=0
    ):
        kmc_dir = os.path.join(self.outpath, "kmc")
        if not os.path.exists(kmc_dir):
            os.makedirs(kmc_dir)
        data = run_kmc(
            lifetimes,
            kmc_dir,
            self.chromophores,
            self.snap,
            temp,
            n_holes=n_holes,
            n_elec=n_elec,
            seed=seed,
            combine=combine,
            carrier_kwargs=carrier_kwargs,
            verbose=verbose
        )

        kmc_analyze.main(data, temp, self.chromophores, self.snap, kmc_dir)

    def visualize_qcc_input(self, i, single=True):
        """Visualize the input for pyscf using mbuild.

        Parameters
        ----------
        qcc_input : str
            Input string to visualize
        """
        import mbuild as mb

        if single:
            qcc_input = self.chromophores[i].qcc_input
        else:
            qcc_input = self.qcc_pairs[i][1]
        comp = mb.Compound()
        for line in qcc_input.split(";")[:-1]:
            atom, x, y, z = line.split()
            xyz = np.array([x,y,z], dtype=float)
            # Angstrom -> nm
            xyz /= 10
            comp.add(mb.Particle(name=atom, pos=xyz))
        comp.visualize().show()

    def visualize_system(self):
        """Visualize the system snapshot."""
        if self._comp is None:
            self._comp = self._make_comp()
        self._comp.visualize().show()

    def visualize_chromophores(self):
        """Visualize the system snapshot."""
        import mbuild as mb

        if self._comp is None:
            self._comp = self._make_comp()
        ccomp = mb.clone(self._comp)
        for i,p in enumerate(ccomp):
            if self._ainds:
                if i in np.hstack(self._ainds):
                    p.name = "Br"
            if self._dinds:
                if i in np.hstack(self._dinds):
                    p.name = "I"
        ccomp.visualize().show()

    def _make_comp(self):
        """Convert the system Snapshot to an mbuild Compound.

        Returns
        -------
        comp : mb.Compound
        """
        import mbuild as mb

        comp = mb.Compound()
        snap = self.snap
        bond_array = snap.bonds.group
        n_atoms = snap.particles.N

        box = snap.configuration.box[:3]
        # to_hoomdsnapshot shifts the coords, this will keep consistent
        shift = box/2

        unwrap_pos = snap.particles.position + snap.particles.image * box
        # angstrom -> nm
        unwrap_pos *= 0.1
        unwrap_pos += shift
        types = [
            self.conversion_dict[i].symbol for i in snap.particles.types
        ]

        # Add particles
        for i in range(n_atoms):
            name = types[snap.particles.typeid[i]]
            xyz = unwrap_pos[i]
            atom = mb.Particle(name=name, pos=xyz)
            comp.add(atom, label=str(i))

        # Add bonds
        for i in range(bond_array.shape[0]):
            atom1 = int(bond_array[i][0])
            atom2 = int(bond_array[i][1])
            comp.add_bond([comp[atom1], comp[atom2]])

        return comp
