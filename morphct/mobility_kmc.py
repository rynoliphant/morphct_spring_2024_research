import math
import multiprocessing as mp
import os
import pickle
import signal
import time
import traceback

import numpy as np
from scipy.sparse import lil_matrix

from morphct import helper_functions as hf


log_file = None


class Carrier:
    def __init__(
        self,
        chromo,
        lifetime,
        carrier_no,
        box,
        temp,
        n,
        hop_limit=None,
        record_history=True,
        mol_id_dict=None,
        use_average_hop_rates = False,
        average_intra_hop_rate = None,
        average_inter_hop_rate = None,
        use_koopmans = False,
        boltz_penalty = False,
        use_VRH = False,
        hopping_prefactor = 1.0
    ):
        self.id = carrier_no
        self.image = np.array([0, 0, 0])
        self.initial_chromo = chromo
        self.current_chromo = chromo
        self.lambda_ij = self.current_chromo.reorganization_energy
        self.hop_limit = hop_limit
        self.temp = temp
        self.lifetime = lifetime
        self.current_time = 0.0

        self.hole_history = None
        self.electron_history = None
        if self.current_chromo.species == "donor":
            self.c_type = "hole"
            if record_history:
                self.hole_history = lil_matrix((n, n), dtype=int)
        elif self.current_chromo.species == "acceptor":
            self.c_type = "electron"
            if record_history:
                self.electron_history = lil_matrix((n, n), dtype=int)

        self.n_hops = 0
        self.box = box
        self.displacement = 0
        self.mol_id_dict = mol_id_dict

        self.use_average_hop_rates = use_average_hop_rates
        self.average_intra_hop_rate = average_intra_hop_rate
        self.average_inter_hop_rate = average_inter_hop_rate

        # Set the use of Koopmans' approximation to false if the key does not
        # exist in the parameter dict
        self.use_koopmans = use_koopmans
        # Are we using a simple Boltzmann penalty?
        self.boltz_penalty = boltz_penalty
        # Are we applying a distance penalty beyond the transfer integral?
        self.use_VRH = use_VRH
        if self.use_VRH:
            self.VRH_delocalisation = (self.current_chromo.VRH_delocalisation)

        self.hopping_prefactor = hopping_prefactor

    def update_displacement(self):
        init_pos = self.initial_chromo.center
        final_pos = self.current_chromo.center
        displacement = final_pos - init_pos + self.image * self.box
        self.displacement = np.linalg.norm(displacement)

    def calculate_hop(self, chromo_list, verbose=False):
        # Terminate if the next hop would be more than the termination limit
        if self.hop_limit is not None:
            if self.n_hops + 1 > self.hop_limit:
                return False
        # Determine the hop times to all possible neighbors
        hop_times = []
        if self.use_average_hop_rates:
            # Use the average hop values given in the parameter dict to pick a
            # hop
            for i,img in self.current_chromo.neighbors:
                neighbor = chromo_list[i]
                assert neighbor.id == i

                current_mol = self.mol_id_dict[self.current_chromo.id]
                neighbor_mol = self.mol_id_dict[neighbor.id]
                if current_mol == neighbor_mol:
                    hop_rate = self.average_intra_hop_rate
                else:
                    hop_rate = self.average_inter_hop_rate
                hop_time = hf.determine_event_tau(hop_rate)
                # Keep track of the chromophoreid and the corresponding tau
                hop_times.append([neighbor.id, hop_time, img])
        else:
            # Obtain the reorganisation energy in J (from eV)
            for i_neighbor, ti in enumerate(self.current_chromo.neighbors_ti):
                # Ignore any hops with a NoneType transfer integral
                if ti is None:
                    continue
                # index of the neighbor in the main list
                n_ind = self.current_chromo.neighbors[i_neighbor][0]
                delta_E_ij = self.current_chromo.neighbors_delta_e[i_neighbor]
                # Load the specified hopping prefactor
                prefactor = self.hopping_prefactor
                # Get the relative image so we can update the carrier image
                # after the hop
                rel_img = self.current_chromo.neighbors[i_neighbor][1]
                # All of the energies are in eV currently, so convert them to J
                if self.use_VRH is True:
                    neighbor_chromo = chromo_list[n_ind]
                    neighbor_pos = neighbor_chromo.center + rel_img * self.box

                    # Chromophore separation needs converting to m
                    sep = np.linalg.norm(
                            self.current_chromo.center - neighbor_pos
                            ) * 1e-10

                    hop_rate = hf.calculate_carrier_hop_rate(
                        self.lambda_ij,
                        ti,
                        delta_E_ij,
                        prefactor,
                        self.temp,
                        use_VRH=True,
                        rij=separation,
                        VRH_delocalisation=self.VRH_delocalisation,
                        boltz_pen=self.boltz_penalty,
                    )
                else:
                    hop_rate = hf.calculate_carrier_hop_rate(
                        self.lambda_ij,
                        ti,
                        delta_E_ij,
                        prefactor,
                        self.temp,
                        boltz_pen=self.boltz_penalty,
                    )
                hop_time = hf.determine_event_tau(hop_rate)
                # Keep track of the chromophoreid and the corresponding tau
                hop_times.append([n_ind, hop_time, rel_img])
        # Sort by ascending hop time
        hop_times.sort(key=lambda x: x[1])
        if len(hop_times) == 0:
            # We are trapped here, so create a dummy hop with time 1E99
            hop_times = [[self.current_chromo.id, 1e99, [0, 0, 0]]]
        # As long as we're not limiting by the number of hops:
        if self.hop_limit is None:
            # Ensure that the next hop does not put the carrier over its
            # lifetime
            if (self.current_time + hop_times[0][1]) > self.lifetime:
                # Send the termination signal to singleCoreRunKMC.py
                return False
        # Move the carrier and send the contiuation signal to
        # singleCoreRunKMC.py
        # Take the quickest hop
        n_ind, hop_time, rel_img = hop_times[0]
        if verbose:
            print("\thop_times:")
            print(*[f"\t{i} {j:.2e} {k}" for (i,j,k) in hop_times], sep="\n")
            print(f"\tHopping to {n_ind}")
        self.perform_hop(chromo_list[n_ind], hop_time, rel_img)
        return True

    def perform_hop(self, destination_chromo, hop_time, rel_image):
        init_id = self.current_chromo.id
        dest_id = destination_chromo.id
        self.image += rel_image
        # Carrier image now sorted, so update its current position
        self.current_chromo = destination_chromo
        # Increment the simulation time
        self.current_time += hop_time
        # Increment the hop counter
        self.n_hops += 1
        # Now update the sparse history matrix
        if self.c_type == "hole" and self.hole_history is not None:
            self.hole_history[init_id, dest_id] += 1
        elif self.c_type == "electron" and self.electron_history is not None:
            self.electron_history[init_id, dest_id] += 1


class termination_signal:
    kill_sent = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.catch_kill)
        signal.signal(signal.SIGTERM, self.catch_kill)

    def catch_kill(self, signum, frame):
        self.kill_sent = True


class terminate(Exception):
    """This class is raised to terminate a KMC simulation"""

    def __init__(self, string):
        self.string = string

    def __str__(self):
        return self.string


def save_pickle(save_data, save_pickle_name):
    with open(save_pickle_name, "wb+") as pickle_file:
        pickle.dump(save_data, pickle_file)
    hf.write_to_file(
        log_file, [f"Pickle file saved successfully as {save_pickle_name}"]
    )




def initialise_save_data(n_chromos, seed):
    return {
        "seed": seed,
        "id": [],
        "image": [],
        "lifetime": [],
        "current_time": [],
        "n_hops": [],
        "displacement": [],
        "hole_history": lil_matrix((n_chromos, n_chromos), dtype=int),
        "electron_history": lil_matrix((n_chromos, n_chromos), dtype=int),
        "initial_position": [],
        "final_position": [],
        "c_type": [],
    }


def split_molecules(input_dictionary, chromo_list):
    # Split the full morphology into individual molecules
    # Create a lookup table `neighbor list' for all connected atoms called
    # {bondedAtoms}
    bonded_atoms = hf.obtain_bonded_list(input_dictionary["bond"])
    molecule_list = [i for i in range(len(input_dictionary["type"]))]
    # Recursively add all atoms in the neighbor list to this molecule
    for mol_id in range(len(molecule_list)):
        molecule_list = update_molecule(mol_id, molecule_list, bonded_atoms)
    # Here we have a list of len(atoms) where each index gives the molid
    mol_id_dict = {}
    for chromo in chromo_list:
        AAID_to_check = chromo.AAIDs[0]
        mol_id_dict[chromo.id] = molecule_list[AAID_to_check]
    return mol_id_dict


def update_molecule(atom_id, molecule_list, bonded_atoms):
    # Recursively add all neighbors of atom number atomid to this molecule
    try:
        for bonded_atom in bonded_atoms[atom_id]:
            # If the moleculeid of the bonded atom is larger than that of the
            # current one, update the bonded atom's id to the current one's to
            # put it in this molecule, then iterate through all of the bonded
            # atom's neighbors
            if molecule_list[bonded_atom] > molecule_list[atom_id]:
                molecule_list[bonded_atom] = molecule_list[atom_id]
                molecule_list = update_molecule(
                    bonded_atom, molecule_list, bonded_atoms
                )
            # If the moleculeid of the current atom is larger than that of the
            # bonded one, update the current atom's id to the bonded one's to
            # put it in this molecule, then iterate through all of the current
            # atom's neighbors
            elif molecule_list[bonded_atom] < molecule_list[atom_id]:
                molecule_list[atom_id] = molecule_list[bonded_atom]
                molecule_list = update_molecule(
                    atom_id, molecule_list, bonded_atoms
                )
            # Else: both the current and the bonded atom are already known to
            # be in this molecule, so we don't have to do anything else.
    except KeyError:
        # This means that there are no bonded CG sites
        # (i.e. it's a single molecule)
        pass
    return molecule_list


def run_single_kmc(
        jobs,
        KMC_directory,
        chromo_list,
        box,
        temp,
        carrier_kwargs={},
        cpu_rank=None,
        seed=None,
        send_end=None,
        ):
    print(f"Found {len(jobs):d} jobs to run")
    try:
        use_avg_hop_rates = carrier_kwargs["use_avg_hop_rates"]
    except KeyError:
        use_avg_hop_rates = False

    if use_avg_hop_rates:
        # Chosen to split hopping by inter-intra molecular hops, so get
        # molecule data
        mol_id_dict = split_molecules(AA_morphdict, chromo_list)
        # molidDict is a dictionary where the keys are the chromoids, and
        # the vals are the molids
    else:
        mol_id_dict = None
    t0 = time.perf_counter()
    carrier_list = []
    for i_job, [carrier_no, lifetime, ctype] in enumerate(jobs):
        print(f"starting job {i_job}")
        t1 = time.perf_counter()
        # Find a random position to start the carrier in
        while True:
            i = np.random.randint(0, len(chromo_list) - 1)
            i_chromo = chromo_list[i]
            if (ctype == "electron") and (i_chromo.species != "acceptor"):
                continue
            elif (ctype == "hole") and (i_chromo.species != "donor"):
                continue
            break
        # Create the carrier instance
        i_carrier = Carrier(
            i_chromo,
            lifetime,
            carrier_no,
            box,
            temp,
            len(chromo_list),
            mol_id_dict=mol_id_dict,
        )
        continue_simulation = True
        i=0
        while continue_simulation:
            continue_simulation = i_carrier.calculate_hop(chromo_list)
            i += 1
            if i > 100:
                continue_simulation = False
        # Now the carrier has finished hopping, let's calculate its vitals
        i_carrier.update_displacement()

        t2 = time.perf_counter()
        elapsed_time = float(t2) - float(t1)
        time_str = hf.time_units(elapsed_time)

        print("\t{} hopped {} times, over {:.2e} seconds, ".format(
            i_carrier.c_type,
            i_carrier.n_hops,
            i_carrier.current_time
            )
        )
        print(f"\tinto image {i_carrier.image}")
        print("\tfor a displacement of {:.2f} (took walltime {})".format(
            i_carrier.displacement,
            time_str
            )
        )
        carrier_list.append(i_carrier)
    t3 = time.perf_counter()
    elapsed_time = float(t3) - float(t0)
    time_str = hf.time_units(elapsed_time)
    if send_end is not None:
        send_end.send(save_data)
    return carrier_list


def get_jobslist(sim_times, n_holes=0, n_elec=0, nprocs=None, seed=None):
    # Get the random seed now for all the child processes
    if seed is not None:
        np.random.seed(seed)
    if nprocs is None:
        nprocs = mp.cpu_count()
    # Determine the maximum simulation times based on the parameter dictionary
    carriers = []
    # Modification: Rather than being clever here with the carriers, I'm just
    # going to create the master list of jobs that need running and then
    # randomly shuffle it. This will hopefully permit a similar number of holes
    # and electrons and lifetimes to be run simultaneously providing adequate
    # statistics more quickly
    for lifetime in sim_times:
        for carrier_no in range(n_holes):
            carriers.append([carrier_no, lifetime, "hole"])
        for carrier_no in range(n_elec):
            carriers.append([carrier_no, lifetime, "electron"])
    np.random.shuffle(carriers)
    step = math.ceil(len(carriers) / nprocs)
    jobs_list = [carriers[i : i+step] for i in range(0, len(carriers), step)]
    return jobs_list


def run_kmc(
        jobs_list,
        KMC_directory,
        AA_morphdict,
        param_dict,
        chromo_list
        ):
    running_jobs = []
    pipes = []

    for proc_id, jobs in enumerate(jobs_list):
        child_seed = np.random.randint(0, 2 ** 32)

        recv_end, send_end = mp.Pipe(False)
        p = mp.Process(target=run_single_kmc, args=(
            jobs,
            KMC_directory,
            AA_morphdict,
            param_dict,
            chromo_list,
            proc_id,
            child_seed,
            send_end
        ))
        running_jobs.append(p)
        pipes.append(recv_end)
        p.start()

    # wait for all jobs to finish
    for p in running_jobs:
        p.join()

    carrier_data_list = [x.recv() for x in pipes]

    # Now combine the carrier data
    print("All KMC jobs completed!")
    if param_dict["combine_KMC_results"] is True:
        print("Combining outputs...")
        combined_data = {}
        for carrier_data in carrier_data_list:
            for key, val in carrier_data.items():
                    if key not in combined_data:
                        combined_data[key] = val
                    else:
                        combined_data[key] += val
        return combined_data
    return carrier_data_list
