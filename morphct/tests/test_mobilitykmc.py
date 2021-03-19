import numpy as np
import pytest

from base_test import BaseTest


class TestKMC(BaseTest):
    def test_init_carrier(self, p3ht_chromo_list_energies):
        from morphct.mobility_kmc import Carrier

        chromo_list = p3ht_chromo_list_energies
        n = len(chromo_list)
        chromo = chromo_list[0]
        box = np.array([85.18963, 85.18963, 85.18963])
        carrier = Carrier(chromo, 1e-13, 0, box, 300, n)

        assert carrier.displacement == 0

        np.random.seed(42)
        carrier.calculate_hop(chromo_list)

        assert carrier.n_hops == 1
        assert carrier.current_chromo.id == 1
        assert carrier.current_time == 7.685088279604407e-15

        carrier.update_displacement()

        assert carrier.displacement == 9.193875570863462
