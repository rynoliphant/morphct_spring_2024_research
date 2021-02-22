import glob
import os
import pickle
import shutil
import sys
import numpy as np
import subprocess as sp
from morphct import helper_functions as hf


def calculate_delta_E(chromo1, chromo2):
    chromo1_e = chromo1.get_MO_energy()
    chromo2_e = chromo2.get_MO_energy()
    return chromo2_e - chromo1_e


def calculate_TI(orbital_splitting, delta_e):
    # Use the energy splitting in dimer method to calculate the electronic
    # transfer integral in eV
    if delta_e ** 2 > orbital_splitting ** 2:
        # Avoid an imaginary TI by returning zero.
        # (Could use KOOPMAN'S APPROXIMATION here if desired)
        TI = 0
    else:
        TI = 0.5 * np.sqrt((orbital_splitting ** 2) - (delta_e ** 2))
    return TI




def scale_energies(chromophore_list, parameter_dict):
    # Shorter chromophores have significantly deeper HOMOs because they are
    # treated as small molecules instead of chain segments. To rectify this,
    # find the average energy level for each chromophore and then map that
    # average to the literature value.
    # First, get the energy level data
    chromophore_species = {k: [] for k in parameter_dict["chromophore_species"].keys()}
    chromophore_MO_info = {k: {} for k in parameter_dict["chromophore_species"].keys()}
    for chromo in chromophore_list:
        chromophore_species[chromo.sub_species].append(chromo.get_MO_energy())

    for sub_species, chromo_energy in chromophore_species.items():
        lit_DOS_std = parameter_dict["chromophore_species"][sub_species][
            "target_DOS_std"
        ]
        lit_MO = parameter_dict["chromophore_species"][sub_species]["literature_MO"]
        chromophore_MO_info[sub_species]["target_DOS_std"] = lit_DOS_std
        chromophore_MO_info[sub_species]["av_MO"] = np.average(chromo_energy)
        chromophore_MO_info[sub_species]["std_MO"] = np.std(chromo_energy)
        chromophore_MO_info[sub_species]["E_shift"] = (
            lit_MO - chromophore_MO_info[sub_species]["av_MO"]
        )

    for chromo in chromophore_list:
        E_shift = chromophore_MO_info[chromo.sub_species]["E_shift"]
        target_DOS_std = chromophore_MO_info[chromo.sub_species]["target_DOS_std"]
        std_MO = chromophore_MO_info[chromo.sub_species]["std_MO"]
        av_MO = chromophore_MO_info[chromo.sub_species]["av_MO"]

        chromo.HOMO_1 += E_shift
        chromo.HOMO += E_shift
        chromo.LUMO += E_shift
        chromo.LUMO_1 += E_shift

        if (target_DOS_std is not None) and (target_DOS_std < std_MO):
            # Determine how many sigmas away from the mean this datapoint is
            sigma = (chromo.get_MO_energy() - av_MO) / std_MO
            # Calculate the new deviation from the mean based on the target
            # STD and sigma
            new_deviation = target_DOS_std * sigma
            # Work out the change in energy to be applied to meet this target
            # energy level
            delta_E = (av_MO + new_deviation) - chromo.get_MO_energy()
            # Apply the energy level displacement
            chromo.HOMO_1 += delta_E
            chromo.HOMO += delta_E
            chromo.LUMO += delta_E
            chromo.LUMO_1 += delta_E
    return chromophore_list


if __name__ == "__main__":
    pass
