import csv
import os
import pytest
import shutil
import subprocess
from morphct import run_MorphCT
from morphct.definitions import TEST_ROOT
from testing_tools import TestCommand
from morphct.code import helper_functions as hf


@pytest.fixture(scope="module")
def run_simulation(request):
    # ---==============================================---
    # ---=============== Setup Prereqs ================---
    # ---==============================================---

    output_dir = os.path.join(TEST_ROOT, "output_AEB")

    try:
        shutil.rmtree(output_dir)
    except OSError:
        pass
    # os.makedirs(os.path.join(output_dir, 'figures'))
    shutil.copytree(
        os.path.join(TEST_ROOT, "assets", "add_error_bars"), os.path.join(output_dir)
    )




    command = ["addErrorBars", "-c", "{'morph_1': ['output_AEB/morph_1*'], 'morph_2': ['output_AEB/morph_2*'], 'morph_3': ['output_AEB/morph_3*']}", "-x", "$\\psi^{\\prime}$ (Arb. U.)", "-s", "0.33,0.25,0.17", "-p" "hole_mobility", "-o", os.path.join(output_dir, "hole_mobility.pdf")]
    subprocess.Popen(command).communicate()


# ---==============================================---
# ---================= Run Tests ==================---
# ---==============================================---


class TestCompareOutputs(TestCommand):
    def test_check_output_exists(self, run_simulation):
        print([file_name for file_name in os.listdir(os.path.join(TEST_ROOT, "output_AEB"))])
        self.confirm_file_exists(
            os.path.join(TEST_ROOT, "output_AEB", "hole_mobility.pdf")
        )


def teardown_module():
    shutil.rmtree(os.path.join(TEST_ROOT, "output_AEB"))


if __name__ == "__main__":

    class parameters:
        def __init__(self, param):
            self.param = param

    run_simulation(parameters(""))
