import os
import pytest


class BaseTest:

    @pytest.fixture
    def p3ht_snap(self):
        import gsd.hoomd

        test_dir = os.path.dirname(__file__)
        rel_path = "assets/p3ht_2_15mers.gsd"
        filepath = os.path.join(main_script_dir, rel_path)
        with gsd.hoomd.open(name=filepath, mode='rb') as f:
            snap = f[0]

        return snap


