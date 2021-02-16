import multiprocessing as mp
import numpy as np

import pyscf
from pyscf.semiempirical import MINDO3

from morphct import helper_functions as hf
from morphct import transfer_integrals as ti


def get_homolumo(molstr, verbose=False, tol=1e-6, send_end=None):
    mol = pyscf.M(atom=molstr)
    mf = MINDO3(mol).run(verbose=verbose, conv_tol=tol)
    occ = mf.get_occ()
    i_lumo = np.argmax(occ<1)
    energies = mf.mo_energy[i_lumo-2:i_lumo+2]
    energies *= 27.2114 # convert Eh to eV
    if send_end is not None:
        send_end.send(energies)
        return
    return energies


def singles_homolumo(chromo_list, filename=None, nprocs=None):
    if nprocs is not None:
        nprocs = mp.cpu_count()
    p = mp.Pool(processes=nprocs)
    data = p.map(get_homolumo, [i.qcc_input for i in chromo_list])
    p.close()
    data = np.stack(data)
    if filename is not None:
        np.savetxt(filename, data)
    return data


def dimer_homolumo(qcc_pairs, filename=None, nprocs=None):
    if nprocs is not None:
        nprocs = mp.cpu_count()

    p = mp.Pool(processes=nprocs)
    data = p.map(eqcc.get_homolumo, [qcc_input for pair,qcc_input in qcc_pairs])
    p.close()

    dimer_data = [i for i in zip([pair for pair,qcc_input in qcc_pairs],data)]
    if filename is not None:
        with open(filename, "w") as f:
            f.writelines(
                f"{pair[0]} {pair[1]} {en[0]} {en[1]} {en[2]} {en[3]}\n"
                for pair,en in dimer_data
            )
    return dimer_data


def get_dimerdata(filename):
    dimer_data = []
    with open(filename, "r") as f:
        for i in f.readlines():
            a,b,c,d,e,f = i.split()
            dimer_data.append(
                ((int(a),int(b)),(float(c),float(d),float(e),float(f)))
            )
    return dimer_data


def get_singlesdata(filename):
    return np.loadtxt(filename)


def set_energyvalues(chromo_list, s_filename, d_filename):
    s_data = get_singlesdata(s_filename)
    d_data = get_dimerdata(d_filename)

    for i,chromo in enumerate(chromo_list):
        chromo.HOMO_1, chromo.HOMO, chromo.LUMO, chromo.LUMO_1 = s_data[i]

    for (i,j), (HOMO_1, HOMO, LUMO, LUMO_1) in d_data:
        chromo1 = chromo_list[i]
        chromo2 = chromo_list[j]
        neighborind1 = [i[0] for i in chromo1.neighbors].index(j)
        neighborind2 = [i[0] for i in chromo2.neighbors].index(i)
        deltaE = ti.calculate_delta_E(chromo1,chromo2)
        chromo1.neighbors_delta_E[neighborind1] = deltaE
        chromo2.neighbors_delta_E[neighborind2] = -deltaE

        assert chromo1.species == chromo2.species
        if chromo1.species.lower() == "donor":
            TI = ti.calculate_TI(HOMO - HOMO_1, deltaE)
        else:
            TI = ti.calculate_TI(LUMO - LUMO_1, deltaE)
        chromo1.neighbors_TI[neighborind1] = TI
        chromo2.neighbors_TI[neighborind2] = TI


def create_inputs(chromo_list, AA_morphdict, param_dict):
    # Singles first
    for chromophore in chromo_list:
        # Include the molecule terminating units on the required atoms of the
        # chromophore
        if chromophore.terminate is True:
            terminating_group_positions = terminate_monomers(
                chromophore, param_dict, AA_morphdict
            )
            terminating_group_images = [chromophore.image] * len(
                terminating_group_positions
            )
        else:
            terminating_group_positions = None
            terminating_group_images = None
        chromophore.qcc_input = write_qcc_inp(
            AA_morphdict,
            chromophore.AAIDs,
            [chromophore.image] * len(chromophore.AAIDs),
            terminating_group_positions,
            terminating_group_images,
        )
    # Determine how many pairs there are first:
    n_pairs = np.sum([len(chromo.neighbors) for chromo in chromo_list])
    print(f"There are {n_pairs // 2} total neighbor pairs to consider.")
    # /2 because the forwards and backwards hops are identical
    # Then consider each chromophore against every other chromophore
    qcc_pairs = []
    for chromo1 in chromo_list:
        neighbors_ID = [neighbor[0] for neighbor in chromo1.neighbors]
        neighbors_image = [neighbor[1] for neighbor in chromo1.neighbors]
        for chromo2 in chromo_list:
            # Skip if chromo2 is not one of chromo1's neighbors
            # Also skip if chromo2's ID is < chromo1's ID to prevent
            # duplicates
            if (chromo2.ID not in neighbors_ID) or (chromo2.ID < chromo1.ID):
                continue
            # Update the qcc input name
            pair = (chromo1.ID, chromo2.ID)
            # Find the correct relative image for the neighbor chromophore
            chromo2_rel_image = neighbors_image[neighbors_ID.index(chromo2.ID)]
            chromo2_transformation = list(
                np.array(chromo1.image)
                - np.array(chromo2.image)
                + np.array(chromo2_rel_image)
            )
            # Find the dimer AAIDs and relative images for each atom
            AAIDs = chromo1.AAIDs + chromo2.AAIDs
            images = [[0, 0, 0] for i in range(len(chromo1.AAIDs))] + [
                chromo2_transformation for i in range(len(chromo2.AAIDs))
            ]
            # Now add the terminating groups to both chromophores
            # Note that we would only ever expect both chromophores to require
            # termination or neither
            if chromo1.terminate is True:
                term_group_pos1 = terminate_monomers(
                    chromo1, param_dict, AA_morphdict
                )
                term_group_pos2 = terminate_monomers(
                    chromo2, param_dict, AA_morphdict
                )
                # We don't want to add the terminating hydrogens for adjacent
                # monomers, so remove the ones that are within a particular
                # distance
                term_group_pos1, term_group_pos2 = remove_adjacent_terminators(
                    term_group_pos1, term_group_pos2
                )
                terminating_group_images1 = [
                    [0, 0, 0] for i in range(len(term_group_pos1))
                ]
                terminating_group_images2 = [
                    chromo2_transformation for i in range(len(term_group_pos2))
                ]
                # Write the dimer input file
                qcc_input = write_qcc_inp(
                    AA_morphdict,
                    AAIDs,
                    images,
                    term_group_pos1 + term_group_pos2,
                    terminating_group_images1 + terminating_group_images2,
                )
            else:
                # Write the dimer input file
                qcc_input = write_qcc_inp(
                    AA_morphdict,
                    AAIDs,
                    images,
                    None,
                    None,
                )
            qcc_pairs.append((pair,qcc_input))
    return qcc_pairs


def remove_adjacent_terminators(group1, group2):
    pop_list = [[], []]
    for index1, terminal_hydrogen1 in enumerate(group1):
        for index2, terminal_hydrogen2 in enumerate(group2):
            sep = np.linalg.norm(terminal_hydrogen2 - terminal_hydrogen1)
            if sep < 1.2:
                pop_list[0].append(index1)
                pop_list[1].append(index2)
    for group_no, group in enumerate(pop_list):
        for index in sorted(group, reverse=True):
            try:
                [group1, group2][group_no].pop(index)
            except IndexError:
                raise SystemError(
                    """
                    Tried to pop a termination group that does not exist...
                    are you sure this is an atomistic morphology?
                    """
                )
    return group1, group2


def write_qcc_inp(
    AA_morphdict,
    AAIDs,
    images,
    terminal_pos,
    terminal_images,
):
    qcc_lines = []
    all_atom_types = []
    all_positions = []
    numstr = "0123456789"
    lxyz = [AA_morphdict["lx"], AA_morphdict["ly"], AA_morphdict["lz"]]
    # Format the atom positions ready for qcc
    for index, atom_ID in enumerate(AAIDs):
        # Cut the integer bit and capitalize to allow two letter atom types
        atom_type = AA_morphdict["type"][atom_ID].strip(numstr).capitalize()
        all_atom_types.append(atom_type)
        # Add in the correct periodic images to the position
        all_positions.append(
            AA_morphdict["unwrapped_position"][atom_ID]
            + np.array([(images[index][i] * lxyz[i]) for i in range(3)])
        )
    # Now add in the terminating hydrogens if necessary
    if terminal_pos is not None:
        for ind, pos in enumerate(terminal_pos):
            all_atom_types.append("H")
            # Add in the correct periodic images to the position
            all_positions.append(
                    pos + np.array(
                        [terminal_images[ind][i] * lxyz[i] for i in range(3)]
                        )
                    )
    # Now geometrically centralize all of the atoms that are to be included in
    # this input file to make it easier on qcc
    central_position = np.array(
        [
            np.average(np.array(all_positions)[:, 0]),
            np.average(np.array(all_positions)[:, 1]),
            np.average(np.array(all_positions)[:, 2]),
        ]
    )
    # Create the lines to be written in the input file
    for index, position in enumerate(all_positions):
        qcc_lines.append(
            "{0:s}  {1:.5f}  {2:.5f}  {3:.5f};".format(
                all_atom_types[index],
                position[0] - central_position[0],
                position[1] - central_position[1],
                position[2] - central_position[2],
            )
        )
    qcc_input = " ".join(qcc_lines)
    return qcc_input

def terminate_monomers(chromophore, param_dict, AA_morphdict):
    # No CG morphology, so we will use the UA -> AA code definition of which
    # atoms need to have hydrogens added to them.
    new_hydrogen_positions = []
    for atom_index_chromo, atom_index_morph in enumerate(chromophore.AAIDs):
        atom_type = AA_morphdict["type"][atom_index_morph]
        if atom_type not in param_dict[
                "molecule_terminating_connections"
                ].keys():
            continue
        bonded_AAIDs = []
        # Iterate over all termination connections defined for this atomType (in
        # case we are trying to do something mega complicated)
        for connection_info in param_dict["molecule_terminating_connections"][
            atom_type
        ]:
            for [bond_name, AAID1, AAID2] in chromophore.bonds:
                if AAID1 == atom_index_morph:
                    if AAID2 not in bonded_AAIDs:
                        bonded_AAIDs.append(AAID2)
                elif AAID2 == atom_index_morph:
                    if AAID1 not in bonded_AAIDs:
                        bonded_AAIDs.append(AAID1)
            if len(bonded_AAIDs) != connection_info[0]:
                continue
            new_hydrogen_positions += hf.get_terminating_positions(
                AA_morphdict["unwrapped_position"][atom_index_morph],
                [
                    AA_morphdict["unwrapped_position"][bonded_AAID]
                    for bonded_AAID in bonded_AAIDs
                ],
                1,
            )
    # Return terminatingGroups (positions of those hydrogens to be added to the
    # qcc input)
    return new_hydrogen_positions


if __name__ == "__main__":
    pass
