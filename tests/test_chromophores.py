import numpy as np
from base_test import BaseTest

class TestChromophores(BaseTest):

    def test_chromos_from_smiles(self, p3ht_snap):
        from morphct.chromophores import get_chromo_ids_smiles, conversion_dict

        p3ht_smarts = "[#6]1[#6][#16][#6][#6]1CCCCCC"
        aaids = get_chromo_ids_smiles(p3ht_snap, p3ht_smarts, conversion_dict)
        assert len(aaids) == 29
        assert np.all([len(i)  == 11 for i in aaids])
