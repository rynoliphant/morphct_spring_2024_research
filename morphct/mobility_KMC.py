import glob
import os
import math
import pickle
import sys
import numpy as np
import subprocess as sp
from morphct import helper_functions as hf


def main(
    AA_morphology_dict,
    CG_morphology_dict,
    CG_to_AAID_master,
    param_dict,
    chromophore_list,
):
    # Get the random seed now for all the child processes
    if param_dict["random_seed_override"] is not None:
        np.random.seed(param_dict["random_seed_override"])
    try:
        if param_dict["use_average_hop_rates"]:
            print("Be advised: use_average_hop_rates is set to {}".format(
                        repr(param_dict["use_average_hop_rates"])))
            print("Average Intra-molecular hop rate: {}".format(
                param_dict["average_intra_hop_rate"]))
            print("Average Inter-molecular hop rate: {}".format(
                param_dict["average_inter_hop_rate"]))
    except KeyError:
        pass
    # Determine the maximum simulation times based on the parameter dictionary
    simulation_times = param_dict["simulation_times"]
    carrier_list = []
    # Modification: Rather than being clever here with the carriers, I'm just
    # going to create the master list of jobs that need running and then
    # randomly shuffle it. This will hopefully permit a similar number of holes
    # and electrons and lifetimes to be run simultaneously providing adequate
    # statistics more quickly
    for lifetime in simulation_times:
        for carrier_no in range(param_dict["number_of_holes_per_simulation_time"]):
            carrier_list.append([carrier_no, lifetime, "hole"])
        for carrier_no in range(
            param_dict["number_of_electrons_per_simulation_time"]
        ):
            carrier_list.append([carrier_no, lifetime, "electron"])
    np.random.shuffle(carrier_list)
    proc_IDs = param_dict["proc_IDs"]
    step = math.ceil(len(carrier_list) / len(proc_IDs))
    jobs_list = [
        carrier_list[i : i + step] for i in range(0, len(carrier_list), step)
    ]
    return jobs_list


if __name__ == "__main__":
    pass
