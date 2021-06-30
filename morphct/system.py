import os

import gsd, gsd.hoomd

from morphct.chromophores import Chromophore, set_neighbors_voronoi
from morphct.execute_qcc import (
    singles_homolumo, dimer_homolumo, set_energyvalues
)

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

    def compute_energies(dcut=None):
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


