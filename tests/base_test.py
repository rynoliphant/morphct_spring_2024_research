import os
import pytest


class BaseTest:
    test_dir = os.path.dirname(__file__)

    @pytest.fixture
    def p3ht_snap(self):
        import gsd.hoomd

        filepath = os.path.join(test_dir, "assets/p3ht_2_15mers.gsd")
        with gsd.hoomd.open(name=filepath, mode='rb') as f:
            snap = f[0]

        return snap

    @pytest.fixture
    def p3ht_chromo_list(self):
        import pickle

        filepath = os.path.join(test_dir, "assets/chromo_list.pkl")
        with open(name=filepath, mode='rb') as f:
            chromo_list = pickle.load(f)
        return chromo_list

    @pytest.fixture
    def p3ht_chromo_list_neighbors(self):
        import pickle

        filepath = os.path.join(test_dir, "assets/chromo_list_neighbors.pkl")
        with open(name=filepath, mode='rb') as f:
            chromo_list = pickle.load(f)
        return chromo_list

    @pytest.fixture
    def p3ht_qcc_pairs(self):
        import pickle

        filepath = os.path.join(test_dir, "assets/qcc_pairs.pkl")
        with open(name=filepath, mode='rb') as f:
            qcc_pairs = pickle.load(f)
        return qcc_pairs

    @pytest.fixture
    def p3ht_singlesdata(self):
        from morphct.execute_qcc import get_singlesdata

        filepath = os.path.join(test_dir, "assets/singles_energies.txt")
        data = get_singlesdata(filepath)
        return data

    @pytest.fixture
    def p3ht_dimerdata(self):
        from morphct.execute_qcc import get_dimerdata

        filepath = os.path.join(test_dir, "assets/dimer_energies.txt")
        data = get_dimerdata(filepath)
        return data
