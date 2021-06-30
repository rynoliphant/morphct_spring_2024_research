import os

import gsd, gsd.hoomd

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
        self.qcc_pairs = None

    @property
    def chromophores(self):
        """"""
        return self._chromophores

    def add_chromophores(self, indices, species, charge=0):
        """

        Parameters
        ----------

        """
        start = len(self.chromophores)
        for i, ind in enumerate(indices):
            self._chromophores.append(
                Chromophore(
                    i+start,
                    self._snap,
                    ind,
                    species,
                    self._conversion_dict,
                    charge=charge
                )
            )

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

        s_filename = os.path.join(outpath, "singles_energies.txt")
        d_filename = os.path.join(outpath, "dimer_energies.txt")

        print("Starting singles energy calculation...")
        data = singles_homolumo(self.chromophores, s_filename)
        print(f"Finished. Output written to {s_filename}.")

        print("Starting dimer energy calculation...")
        dimer_data = dimer_homolumo(
            self.qcc_pairs, self.chromophores, d_filename
        )
        print(f"Finished. Output written to {d_filename}.")

    def set_energies(self, s_filename, d_filename):
        """

        Parameters
        ----------

        """
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
        carrier_kwargs={}
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
            carrier_kwargs=carrier_kwargs
        )

        kmc_analyze.main(data, temp, self.chromophores, self.snap, kmc_dir)
