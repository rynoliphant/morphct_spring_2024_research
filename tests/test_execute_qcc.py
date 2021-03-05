import numpy as np
from base_test import BaseTest

class TestEQCC(BaseTest):

    def test_pyscf(self):
        import pyscf
        from pysscf.semiempirical import MINDO3

        mol = pyscf.M(atom="H 0 0 0; F 0.9 0 0;")
        mf = MINDO3(mol).run(verbose=5)

