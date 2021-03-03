
class BaseTest:

    @pytest.fixture
    def p3ht_snap(self):
        import gsd.hoomd

        with gsd.hoomd.open(name='assets/p3ht_2_15mers.gsd', mode='rb') as f:
            snap = f[0]

        return snap


