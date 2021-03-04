import numpy as np
from base_test import BaseTest

class TestEQCC(BaseTest):

    def test_singles_homolumo(self, p3ht_chromo_list):
        from morphct.execute_qcc import singles_homolumo

        chromo = p3ht_chromo_list[0]
        data = singles_homolumo([chromo])
        assert np.allclose(
                data,
                np.array([[-9.01337182, -8.5404688, 0.17193304, 0.86523495]])
                )

    def test_dimer_homolumo(self, p3ht_qcc_pairs):
        from morphct.execute_qcc import dimer_homolumo

        qcc_pair = p3ht_qcc_pairs[0]
        pair, energies = dimer_homolumo([qcc_pair])[0]

        assert pair == (0,1)
        assert np.allclose(
                energies,
                np.array([-8.70917208, -8.21756382, -0.26434042, 0.35184321])
                )
