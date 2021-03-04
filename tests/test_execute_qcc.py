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
