from collections import defaultdict
import copy
import csv
import glob
import itertools
import os
import pickle
import shutil
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.sparse import lil_matrix
from scipy.stats import linregress

from morphct import helper_functions as hf


plt = None
p3 = None
elem_chrg = 1.60217657e-19  # C
kB = 1.3806488e-23  # m^{2} kg s^{-2} K^{-1}
hbar = 1.05457173e-34  # m^{2} kg s^{-1}
temperature = 290  # K


def load_KMC_results_pickle(path):
    KMC_pickle = os.path.join(path, "KMC", "KMC_results.pickle")
    try:
        with open(KMC_pickle, "rb") as pickle_file:
            carrier_data = pickle.load(pickle_file)
    except FileNotFoundError:
        print(
            "No final KMC_results.pickle found. ",
            "Creating it from incomplete parts..."
        )
        create_results_pickle(path)
        with open(KMC_pickle, "rb") as pickle_file:
            carrier_data = pickle.load(pickle_file)
    except UnicodeDecodeError:
        with open(KMC_pickle, "rb") as pickle_file:
            carrier_data = pickle.load(pickle_file, encoding="latin1")
    return carrier_data


def split_carriers_by_type(carrier_data):
    # If only one carrier type has been given, call the carriers holes and
    # skip the electron calculations
    list_variables = [
        "current_time",
        "ID",
        "no_hops",
        "displacement",
        "lifetime",
        "final_position",
        "image",
        "initial_position",
    ]
    try:
        carrier_data_holes = {
            "carrier_history_matrix": carrier_data["hole_history_matrix"],
            "seed": carrier_data["seed"],
        }
        carrier_data_electrons = {
            "carrier_history_matrix": carrier_data["electron_history_matrix"],
            "seed": carrier_data["seed"],
        }
        for list_var in list_variables:
            carrier_data_holes[list_var] = []
            carrier_data_electrons[list_var] = []
            for carrier_i, charge_type in enumerate(
                carrier_data["carrier_type"]
            ):
                if charge_type.lower() == "hole":
                    carrier_data_holes[list_var].append(
                        carrier_data[list_var][carrier_i]
                    )
                elif charge_type.lower() == "electron":
                    carrier_data_electrons[list_var].append(
                        carrier_data[list_var][carrier_i]
                    )
    except KeyError:
        print("This is an old-style pickle!")
        print(
            "Multiple charge carriers not found, ",
            "assuming donor material and holes only."
        )
        try:
            carrier_data_holes = {
                "carrier_history_matrix": carrier_data[
                    "carrier_history_matrix"
                ],
                "seed": carrier_data["seed"],
            }
        except KeyError:
            carrier_data_holes = {
                "carrier_history_matrix": carrier_data[
                    "carrier_history_matrix"
                ],
                "seed": 0,
            }
        carrier_data_electrons = None
        for list_var in list_variables:
            carrier_data_holes[list_var] = []
            for carrier_i, carrier_ID in enumerate(carrier_data["ID"]):
                carrier_data_holes[list_var].append(
                    carrier_data[list_var][carrier_i]
                )
    return carrier_data_holes, carrier_data_electrons


def get_carrier_data(carrier_data):
    try:
        carrier_history = carrier_data["carrier_history_matrix"]
    except:
        carrier_history = None
    total_data_points = 0
    total_data_points_averaged_over = 0
    squared_disps = {}
    actual_times = {}
    for carrier_i, displacement in enumerate(carrier_data["displacement"]):
        if (
            (
                carrier_data["current_time"][carrier_i]
                > carrier_data["lifetime"][carrier_i] * 2
            )
            or (
                carrier_data["current_time"][carrier_i]
                < carrier_data["lifetime"][carrier_i] / 2.0
            )
            or (carrier_data["no_hops"][carrier_i] == 1)
        ):
            total_data_points += 1
            continue
        carrier_key = str(carrier_data["lifetime"][carrier_i])
        if carrier_key not in squared_disps:
            squared_disps[carrier_key] = [
                (carrier_data["displacement"][carrier_i] * 1e-10) ** 2
            ]
            # A -> m
            actual_times[carrier_key] = [
                carrier_data["current_time"][carrier_i]
            ]
        else:
            squared_disps[carrier_key].append(
                (carrier_data["displacement"][carrier_i] * 1e-10) ** 2
            )  # A -> m
            actual_times[carrier_key].append(
                carrier_data["current_time"][carrier_i]
            )
        # Also keep track of whether each carrier is a hole or an electron
        total_data_points_averaged_over += 1
        total_data_points += 1
    if total_data_points > total_data_points_averaged_over:
        print(
            "Notice: The data from",
            total_data_points - total_data_points_averaged_over,
            "carriers were discarded",
            "due to the carrier lifetime being more than double ",
            "(or less than half of) the specified carrier lifetime.",
        )
    times = []
    MSDs = []
    time_stderr = []
    MSD_stderr = []
    for time, disps in squared_disps.items():
        times.append(float(time))
        time_stderr.append(
            np.std(actual_times[time]) / len(actual_times[time])
        )
        MSDs.append(np.average(disps))
        MSD_stderr.append(np.std(disps) / len(disps))
    return (
        carrier_history,
        times,
        MSDs,
        time_stderr,
        MSD_stderr,
    )


def plot_displacement_dist(carrier_data, path, carrier_type):
    carrier_types = ["hole", "electron"]
    plt.figure()
    plt.hist(np.array(carrier_data["displacement"]) * 0.1, bins=60, color="b")
    plt.xlabel("".join([carrier_type, "Displacement (nm)"]))
    plt.ylabel("Frequency (Arb. U.)")
    # 30 for hole displacement dist, 31 for electron displacement dist
    filename = "{:02}_{}_displacement_dist.png".format(
            30 + carrier_types.index(carrier_type),
            carrier_type
            )
    filepath = os.path.join(path, "figures", filename)
    plt.savefig(filepath, dpi=300)
    print(f"Figure saved as {filepath}")
    plt.close()


def plot_cluster_size_dist(cluster_freqs, path):
    carrier_types = ["hole", "electron"]
    for carrier_type_i, carrier_type in enumerate(carrier_types):
        try:
            sizes = list(cluster_freqs[carrier_type_i].values())
            sizes = [np.log10(size) for size in sizes if size > 5]
            if len(sizes) == 0:
                raise IndexError
        except IndexError:
            return
        plt.figure()
        try:
            plt.hist(
                sizes,
                bins=np.logspace(0, np.ceil(np.max(sizes)), 20),
                color="b",
            )
            plt.xscale("log")
            plt.xlim([1, 10 ** (np.ceil(np.max(sizes)))])
        except ValueError:
            print(
                "EXCEPTION: No clusters found. ",
                "Are you sure the cluster criteria are correct?"
            )
        plt.xlabel("Cluster Size (Arb. U.)")
        plt.ylabel("Frequency (Arb. U.)")
        # 32 for hole cluster size dist, 33 for electron cluster size dist
        filename = "{:02}_{}_cluster_dist.png".format(
                32 + carrier_types.index(carrier_type),
                carrier_type
                )
        filepath = os.path.join(path, "figures", filename)
        plt.savefig(filepath, dpi=300)
        print(f"Figure saved as {filepath}")
        plt.close()


def create_array_for_plot_connections(chromo_list, carrier_history, sim_dims):
    """
    Function to create an array of with a starting point, a vector
    and the number of hops that occured.
    Requires:
        chromo_list,
        carrier_history
        sim_dims
    Returns:
        7xN array
    """
    # Create an "empty" array to store data.
    connections_array = np.zeros(7)
    # Iterate through the chromo_list
    for i, chromo in enumerate(chromo_list):
        # Iterate through the neighbors of the chromophore
        for neighbor in zip(chromo.neighbors):
            # index of the neighbor
            index = neighbor[0][0]
            image = neighbor[0][1]
            # check to see if they are in the same rel image
            # Only consider one direction.
            if i < index:
                # Get the vector between the two chromophores.
                if not np.count_nonzero(image):
                    vector = chromo_list[index].pos - chromo.pos
                # Account for periodic boundary conditions if not in same relative image.
                else:
                    vector = chromo_list[index].pos - chromo.pos
                    vector += image * np.array(
                        [
                            2 * sim_dims[0][1],
                            2 * sim_dims[1][1],
                            2 * sim_dims[2][1],
                        ]
                    )

                # Get the net number of times the path was travelled.
                forward = carrier_history[index, i]
                reverse = carrier_history[i, index]
                times_travelled = abs(forward - reverse)

                # Append the array if the net times travelled is greater than 0
                if times_travelled > 0:
                    datum = np.hstack(
                        (
                            chromo.pos,
                            vector,
                            np.array([np.log10(times_travelled)]),
                        )
                    )
                    connections_array = np.vstack((connections_array, datum))
    # Return the array excluding the zeros first line.
    return connections_array[1:]


def plot_connections(
    chromo_list, sim_dims, carrier_history, path, carrier_type
):
    # A complicated function that shows connections between carriers in 3D
    # that carriers prefer to hop between.
    # Connections that are frequently used are highlighted in black, whereas
    # rarely used connections are more white.
    # Import matplotlib color modules to set up color bar.
    import matplotlib.colors
    import matplotlib.cm as cmx

    # Create a figure class
    fig = plt.figure(figsize=(7, 6))
    # Make a 3D subplot
    ax = fig.add_subplot(111, projection="3d")

    # Create the array for all the chromophore connections
    connections_array = create_array_for_plot_connections(
        chromo_list, carrier_history, sim_dims
    )

    # Determine the smallest, non-zero number of times two chromophores
    # are connected.
    vmin = np.min(np.array(connections_array)[:, 6])
    # Determine the max number of times two chormophores are connected.
    vmax = np.max(np.array(connections_array)[:, 6])

    # Set up the color bar.
    colour_map = plt.get_cmap("inferno")
    c_norm = matplotlib.colors.Normalize(
        vmin=np.floor(vmin), vmax=np.ceil(vmax)
    )
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=colour_map)
    hop_colors = scalar_map.to_rgba(connections_array[:, 6])

    # Set up the intensity for the hops so more travelled paths are more intense
    alphas = connections_array[:, 6] / vmax
    hop_colors[:, 3] = alphas

    # Plot the vectors between two chromophores
    ax.quiver(
        connections_array[:, 0],
        connections_array[:, 1],
        connections_array[:, 2],
        connections_array[:, 3],
        connections_array[:, 4],
        connections_array[:, 5],
        color=hop_colors,
        arrow_length_ratio=0,
        linewidth=0.7,
    )

    # Draw boxlines
    # Varying X
    ax.plot(
        [sim_dims[0][0], sim_dims[0][1]],
        [sim_dims[1][0], sim_dims[1][0]],
        [sim_dims[2][0], sim_dims[2][0]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][0], sim_dims[0][1]],
        [sim_dims[1][1], sim_dims[1][1]],
        [sim_dims[2][0], sim_dims[2][0]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][0], sim_dims[0][1]],
        [sim_dims[1][0], sim_dims[1][0]],
        [sim_dims[2][1], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][0], sim_dims[0][1]],
        [sim_dims[1][1], sim_dims[1][1]],
        [sim_dims[2][1], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    # Varying Y
    ax.plot(
        [sim_dims[0][0], sim_dims[0][0]],
        [sim_dims[1][0], sim_dims[1][1]],
        [sim_dims[2][0], sim_dims[2][0]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][1], sim_dims[0][1]],
        [sim_dims[1][0], sim_dims[1][1]],
        [sim_dims[2][0], sim_dims[2][0]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][0], sim_dims[0][0]],
        [sim_dims[1][0], sim_dims[1][1]],
        [sim_dims[2][1], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][1], sim_dims[0][1]],
        [sim_dims[1][0], sim_dims[1][1]],
        [sim_dims[2][1], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    # Varying Z
    ax.plot(
        [sim_dims[0][0], sim_dims[0][0]],
        [sim_dims[1][0], sim_dims[1][0]],
        [sim_dims[2][0], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][0], sim_dims[0][0]],
        [sim_dims[1][1], sim_dims[1][1]],
        [sim_dims[2][0], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][1], sim_dims[0][1]],
        [sim_dims[1][0], sim_dims[1][0]],
        [sim_dims[2][0], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][1], sim_dims[0][1]],
        [sim_dims[1][1], sim_dims[1][1]],
        [sim_dims[2][0], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )

    # Make the colour bar
    scalar_map.set_array(connections_array[:, 6])
    tick_location = np.arange(0, vmax, 1)
    # Plot the colour bar
    cbar = plt.colorbar(
        scalar_map, ticks=tick_location, shrink=0.8, aspect=20
    )
    cbar.ax.set_yticklabels([rf"10$^{{{x}}}$" for x in tick_location])

    # Name and save the figure.
    carrier_types = ["hole", "electron"]
    species = ["donor", "acceptor"]
    carrier_i = carrier_types.index(carrier_type)
    figure_title = "{} ({}) Network".format(
            species[carrier_i].capitalize(),
            carrier_type.capitalize()
            )
    # 01 for donor 3d network, 02 for acceptor 3d network
    filename = f"{1 + carrier_i:02}_3d_{carrier_type}.png"
    filepath = os.path.join(path, "figures", filename),
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    print(f"Figure saved as {filepath}")

    plt.clf()


def calc_mobility(lin_fit_X, lin_fit_Y, av_time_error, av_MSD_error):
    # YVals have a std error avMSDError associated with them
    # XVals have a std error avTimeError assosciated with them
    numerator = lin_fit_Y[-1] - lin_fit_Y[0]
    denominator = lin_fit_X[-1] - lin_fit_X[0]
    # Diffusion coeff D = d(MSD)/dt * 1/2n (n = 3 = number of dimensions)
    # Ref: Carbone2014a (Carbone and Troisi)
    diffusion_coeff = numerator / (6 * denominator)
    # The error in the mobility is proportionally the same as the error in the
    # diffusion coefficient as the other variables are constants without error
    diff_error = diffusion_coeff * np.sqrt(
        (av_MSD_error / numerator) ** 2 + (av_time_error / denominator) ** 2
    )
    # Use Einstein-Smoluchowski relation
    # This is in m^{2} / Vs
    mobility = (elem_chrg * diffusion_coeff / (kB * temperature))
    # Convert to cm^{2}/ Vs
    mobility *= 100 ** 2
    mob_error = (diff_error / diffusion_coeff) * mobility
    return mobility, mob_error


def plot_MSD(
    times,
    MSDs,
    time_stderr,
    MSD_stderr,
    path,
    carrier_type,
):
    carrier_types = ["hole", "electron"]
    fit_X = np.linspace(np.min(times), np.max(times), 100)
    gradient, intercept, r_val, p_val, std_err = linregress(times, MSDs)
    print("Standard Error", std_err)
    print("Fitting r_val =", r_val)
    fit_Y = (fit_X * gradient) + intercept
    mobility, mob_error = calc_mobility(
        fit_X,
        fit_Y,
        np.average(time_stderr),
        np.average(MSD_stderr),
    )
    plt.plot(times, MSDs)
    plt.errorbar(times, MSDs, xerr=time_stderr, yerr=MSD_stderr)
    plt.plot(fit_X, fit_Y, "r")
    plt.xlabel("Time (s)")
    plt.ylabel(r"MSD (m$^{2}$)")
    plt.title(
            rf"$\mu_{{0, {carrier_type[0]}}}$ = {mobility:.3e} cm$^{2}$/Vs",
            y=1.1
            )
    # 18 for hole linear MSD, 19 for electron linear MSD
    filename = "{:02}_lin_MSD_{}.png".format(
            18 + carrier_types.index(carrier_type),
            carrier_type
            )
    filepath = os.path.join(path, "figures", filename)
    plt.savefig(filepath, dpi=300)
    plt.clf()
    print(f"Figure saved as {filepath}")
    plt.semilogx(times, MSDs)
    plt.errorbar(times, MSDs, xerr=time_stderr, yerr=MSD_stderr)
    plt.semilogx(fit_X, fit_Y, "r")
    plt.xlabel("Time (s)")
    plt.ylabel(r"MSD (m$^{2}$)")
    plt.title(
            rf"$\mu_{{0, {carrier_type[0]}}}$ = {mobility:.3e} cm$^{2}$/Vs",
            y=1.1
            )
    # 20 for hole semilog MSD, 21 for electron semilog MSD
    filename = "{:02}_semi_log_MSD_{}.png".format(
            20 + carrier_types.index(carrier_type),
            carrier_type
            )
    filepath = os.path.join(path, "figures", filename)
    plt.savefig(filepath, dpi=300)
    plt.clf()
    print(f"Figure saved as {filepath}")
    plt.plot(times, MSDs)
    plt.errorbar(times, MSDs, xerr=time_stderr, yerr=MSD_stderr)
    plt.plot(fit_X, fit_Y, "r")
    plt.xlabel("Time (s)")
    plt.ylabel(r"MSD (m$^{2}$)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(
        rf"$\mu_{{0, {carrier_type[0]}}}$ = {mobility:.3e} cm$^{{2}}$/Vs",
        y=1.1
        )
    # 22 for hole semilog MSD, 23 for electron semilog MSD
    filename = "{:02}_log_MSD_{}.png".format(
            22 + carrier_types.index(carrier_type),
            carrier_type
            )
    filepath = os.path.join(path, "figures", filename)
    plt.savefig(filepath, dpi=300)
    plt.clf()
    print(f"Figure saved as {filepath}")
    return mobility, mob_error, r_val ** 2


def calculate_anisotropy(xvals, yvals, zvals):
    # First calculate the `centre of position' for the particles
    centre = [np.mean(xvals), np.mean(yvals), np.mean(zvals)]
    # First calculate the gyration tensor:
    sxx = 0
    sxy = 0
    sxz = 0
    syy = 0
    syz = 0
    szz = 0
    for carrier_ID, raw_xval in enumerate(xvals):
        xval = raw_xval - centre[0]
        yval = yvals[carrier_ID] - centre[1]
        zval = zvals[carrier_ID] - centre[2]
        sxx += xval * xval
        sxy += xval * yval
        sxz += xval * zval
        syy += yval * yval
        syz += yval * zval
        szz += zval * zval
    S = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    eigenvalues, eigenvectors = np.linalg.eig(S)
    # Diagonalisation of S is the diagonal matrix of the eigenvalues in ascending order
    # diagonalMatrix = np.diag(sorted(eigenValues))
    # We only need the eigenvalues though, no more matrix multiplication
    diagonal = sorted(eigenvalues)
    # Then calculate the relative shape anisotropy (kappa**2)
    anisotropy = (3 / 2) * (
        ((diagonal[0] ** 2) + (diagonal[1] ** 2) + (diagonal[2] ** 2))
        / ((diagonal[0] + diagonal[1] + diagonal[2]) ** 2)
    ) - (1 / 2)
    return anisotropy


def plot_hop_vectors(
    carrier_data,
    AA_morphdict,
    chromo_list,
    path,
    sim_dims,
    carrier_type,
    plot3D_graphs,
):
    import matplotlib.colors
    import matplotlib.cm as cmx

    if not plot3D_graphs:
        return

    box_lengths = np.array([axis[1] - axis[0] for axis in sim_dims])
    carrier_history = carrier_data["carrier_history_matrix"]
    non_zero_coords = list(zip(*carrier_history.nonzero()))
    hop_vectors = []
    intensities = []
    for (chromo1, chromo2) in non_zero_coords:
        # Find the normal of chromo1 plane (first three atoms)
        triplet_pos = [
            np.array(AA_morphdict["position"][AAID])
            for AAID in chromo_list[chromo1].AAIDs[:3]
        ]
        AB = triplet_pos[1] - triplet_pos[0]
        AC = triplet_pos[2] - triplet_pos[0]
        normal = np.cross(AB, AC)
        normal /= np.linalg.norm(normal)
        # Calculate rotation matrix required to map this onto baseline (0, 0, 1)
        rotation_matrix = hf.get_rotation_matrix(normal, np.array([0, 0, 1]))
        # Find the vector from chromo1 to chromo2 (using the correct rleative image)
        neighbor_IDs = [
            neighbor[0] for neighbor in chromo_list[chromo1].neighbors
        ]
        nlist_i = neighbor_IDs.index(chromo2)
        relative_image = np.array(
            chromo_list[chromo1].neighbors[nlist_i][1]
        )
        chromo1_pos = chromo_list[chromo1].pos
        chromo2_pos = chromo_list[chromo2].pos + (
            relative_image * box_lengths
        )
        unrotated_hop_vector = chromo2_pos - chromo1_pos
        # Apply rotation matrix to chromo1-chromo2 vector
        hop_vector = np.matmul(rotation_matrix, unrotated_hop_vector)
        # Store the hop vector and the intensity
        intensity = np.log10(carrier_history[chromo1, chromo2])
        hop_vectors.append(hop_vector)
        intensities.append(intensity)

    # Convert hop_vectors to an np.array
    hop_vectors = np.array(hop_vectors)
    # Create a figure class
    fig = plt.figure(figsize=(7, 6))
    # Make a 3D subplot
    ax = fig.add_subplot(111, projection="3d")
    # Determine the smallest, non-zero number of times two chromophores are connected.
    I_min = np.min(intensities)
    # Determine the max number of times two chormophores are connected.
    I_max = np.max(intensities)

    # Set up the color bar.
    colour_map = plt.get_cmap("Greys")
    c_norm = matplotlib.colors.Normalize(
        vmin=np.floor(I_min), vmax=np.ceil(I_max)
    )
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=colour_map)
    hop_colors = scalar_map.to_rgba(intensities)

    # # Set up the intensity for the hops so more travelled paths are more intense
    # alphas = intensities / I_max
    # hop_colors[:, 3] = alphas

    # Plot the vectors between two chromophores
    ax.quiver(
        # X-vals for start of each arrow
        np.array([0] * len(hop_colors)),
        # Y-vals for start of each arrow
        np.array([0] * len(hop_colors)),
        # Z-vals for start of each arrow
        np.array([0] * len(hop_colors)),
        # X-vecs for direction of each arrow
        hop_vectors[:, 0],
        # Y-vecs for direction of each arrow
        hop_vectors[:, 1],
        # Z-vecs for direction of each arrow
        hop_vectors[:, 2],
        color=hop_colors,
        arrow_length_ratio=0,
        linewidth=0.7,
    )
    ax.set_xlim([-np.max(hop_vectors[:, 0]), np.max(hop_vectors[:, 0])])
    ax.set_ylim([-np.max(hop_vectors[:, 1]), np.max(hop_vectors[:, 1])])
    ax.set_zlim([-np.max(hop_vectors[:, 2]), np.max(hop_vectors[:, 2])])

    # Make the colour bar
    scalar_map.set_array(intensities)
    # tick_location = np.arange(0, int(np.ceil(vmax)) + 1, 1)
    tick_location = np.arange(0, I_max, 1)
    # Plot the colour bar
    cbar = plt.colorbar(scalar_map, ticks=tick_location, shrink=0.8, aspect=20)
    cbar.ax.set_yticklabels([rf"10$^{{{x}}}$" for x in tick_location])

    # Name and save the figure.
    carrier_types = ["hole", "electron"]
    species = ["donor", "acceptor"]
    carrier_i = carrier_types.index(carrier_type)
    figure_title = "{} ({}) Network".format(
            species[carrier_i].capitalize(), carrier_type.capitalize()
            )
    # 36 for donor 3d network, 37 for acceptor 3d network
    filename = "{:02}_hop_vec_{}.png".format(36 + carrier_i, carrier_type)
    filepath = os.path.join(path, "figures", filename),
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    print(f"Figure saved as {filepath}")
    plt.clf()


def plot_anisotropy(
    carrier_data, path, sim_dims, carrier_type, plot3D_graphs
):
    sim_extent = [value[1] - value[0] for value in sim_dims]
    xvals = []
    yvals = []
    zvals = []
    colours = []
    sim_dims_nm = list(map(list, np.array(sim_dims) / 10.0))
    # Get the indices of the carriers that travelled the furthest
    if len(carrier_data["final_position"]) <= 1000:
        carrier_indices_to_use = range(len(carrier_data["final_position"]))
    else:
        displacements = copy.deepcopy(np.array(carrier_data["displacement"]))
        carrier_indices_to_use = displacements.argsort()[-1000:][::-1]
    for carrier_no in carrier_indices_to_use:
        pos = carrier_data["final_position"][carrier_no]
        position = [0.0, 0.0, 0.0]
        for axis in range(len(pos)):
            position[axis] = (
                carrier_data["image"][carrier_no][axis] * sim_extent[axis]
            ) + pos[axis]
        xvals.append(position[0] / 10.0)
        yvals.append(position[1] / 10.0)
        zvals.append(position[2] / 10.0)
        colours.append("b")
    anisotropy = calculate_anisotropy(xvals, yvals, zvals)
    if not plot3D_graphs:
        return anisotropy
    print("----------------------------------------")
    print(
        "{0:s} charge transport anisotropy calculated as {1:f}".format(
            carrier_type.capitalize(), anisotropy
        )
    )
    print("----------------------------------------")
    # Reduce number of plot markers
    fig = plt.gcf()
    ax = p3.Axes3D(fig)
    if len(xvals) > 1000:
        xvals = xvals[0 : len(xvals) : len(xvals) // 1000]
        yvals = yvals[0 : len(yvals) : len(yvals) // 1000]
        zvals = zvals[0 : len(zvals) : len(zvals) // 1000]
    plt.scatter(xvals, yvals, zs=zvals, c=colours, s=20)
    plt.scatter(0, 0, zs=0, c="r", s=50)
    # Draw boxlines
    # Varying X
    ax.plot(
        [sim_dims_nm[0][0], sim_dims_nm[0][1]],
        [sim_dims_nm[1][0], sim_dims_nm[1][0]],
        [sim_dims_nm[2][0], sim_dims_nm[2][0]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims_nm[0][0], sim_dims_nm[0][1]],
        [sim_dims_nm[1][1], sim_dims_nm[1][1]],
        [sim_dims_nm[2][0], sim_dims_nm[2][0]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims_nm[0][0], sim_dims_nm[0][1]],
        [sim_dims_nm[1][0], sim_dims_nm[1][0]],
        [sim_dims_nm[2][1], sim_dims_nm[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims_nm[0][0], sim_dims_nm[0][1]],
        [sim_dims_nm[1][1], sim_dims_nm[1][1]],
        [sim_dims_nm[2][1], sim_dims_nm[2][1]],
        c="k",
        linewidth=1.0,
    )
    # Varying Y
    ax.plot(
        [sim_dims_nm[0][0], sim_dims_nm[0][0]],
        [sim_dims_nm[1][0], sim_dims_nm[1][1]],
        [sim_dims_nm[2][0], sim_dims_nm[2][0]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims_nm[0][1], sim_dims_nm[0][1]],
        [sim_dims_nm[1][0], sim_dims_nm[1][1]],
        [sim_dims_nm[2][0], sim_dims_nm[2][0]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims_nm[0][0], sim_dims_nm[0][0]],
        [sim_dims_nm[1][0], sim_dims_nm[1][1]],
        [sim_dims_nm[2][1], sim_dims_nm[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims_nm[0][1], sim_dims_nm[0][1]],
        [sim_dims_nm[1][0], sim_dims_nm[1][1]],
        [sim_dims_nm[2][1], sim_dims_nm[2][1]],
        c="k",
        linewidth=1.0,
    )
    # Varying Z
    ax.plot(
        [sim_dims_nm[0][0], sim_dims_nm[0][0]],
        [sim_dims_nm[1][0], sim_dims_nm[1][0]],
        [sim_dims_nm[2][0], sim_dims_nm[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims_nm[0][0], sim_dims_nm[0][0]],
        [sim_dims_nm[1][1], sim_dims_nm[1][1]],
        [sim_dims_nm[2][0], sim_dims_nm[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims_nm[0][1], sim_dims_nm[0][1]],
        [sim_dims_nm[1][0], sim_dims_nm[1][0]],
        [sim_dims_nm[2][0], sim_dims_nm[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims_nm[0][1], sim_dims_nm[0][1]],
        [sim_dims_nm[1][1], sim_dims_nm[1][1]],
        [sim_dims_nm[2][0], sim_dims_nm[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.set_xlabel("X (nm)", fontsize=20, labelpad=40)
    ax.set_ylabel("Y (nm)", fontsize=20, labelpad=40)
    ax.set_zlabel("Z (nm)", fontsize=20, labelpad=40)
    maximum = max([max(xvals), max(yvals), max(zvals)])
    ax.set_xlim([-maximum, maximum])
    ax.set_ylim([-maximum, maximum])
    ax.set_zlim([-maximum, maximum])
    for tick in (
        ax.xaxis.get_major_ticks()
        + ax.yaxis.get_major_ticks()
        + ax.zaxis.get_major_ticks()
    ):
        tick.label.set_fontsize(16)
    ax.dist = 11
    # 08 for hole anisotropy, 09 for electron anisotropy
    if carrier_type == "hole":
        figure_i = "08"
    elif carrier_type == "electron":
        figure_i = "09"
    plt.title(f"Anisotropy ({carrier_type.capitalize()})", y=1.1)
    filename = f"{figure_i}_anisotropy_{carrier_type}.png"
    filepath = os.path.join(path, "figures", filename)
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.clf()
    print(f"Figure saved as {filepath}")
    return anisotropy


def get_temp_val(string):
    return string.split("-")[-2][2:]


def get_frame_val(string):
    return string.split("-")[0][1:]


def plot_temperature_progression(
    temp_data, mobility_data, anisotropy_data, carrier_type, x_label
):
    plt.gcf()
    plt.clf()
    xvals = temp_data
    yvals = list(np.array(mobility_data)[:, 0])
    yerrs = list(np.array(mobility_data)[:, 1])
    plt.xlabel(x_label)
    plt.ylabel(r"Mobility (cm$^{2}$ / Vs)")
    plt.errorbar(xvals, yvals, xerr=0, yerr=yerrs)
    plt.yscale("log")
    filename = f"mobility_{carrier_type}.png"
    plt.savefig(filename, dpi=300)
    plt.clf()
    print(f"Figure saved as {filename}")

    plt.plot(temp_data, anisotropy_data, c="r")
    plt.xlabel(x_label)
    plt.ylabel(r"$\kappa$ (Arb. U)")
    filename = f"anisotropy_{carrier_type}.png"
    plt.savefig(filename, dpi=300)
    plt.clf()
    print(f"Figure saved as {filename}")


def calculate_lambda_ij(chromo_length):
    # The equation for the internal reorganisation energy was obtained from
    # the data given in
    # Johansson, E and Larsson, S; 2004, Synthetic Metals 144: 183-191.
    # External reorganisation energy obtained from
    # Liu, T and Cheung, D. L. and Troisi, A; 2011, Phys. Chem. Chem. Phys.
    # 13: 21461-21470
    lambda_external = 0.11  # eV
    if chromo_length < 12:
        lambda_internal = 0.20826 - (chromo_length * 0.01196)
    else:
        lambda_internal = 0.06474
    lambdae_V = lambda_external + lambda_internal
    return lambdae_V


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_fit(data):
    mean = np.mean(data)
    std = np.std(data)
    hist, bin_edges = np.histogram(data, bins=100)
    try:
        fit_args, fit_conv = curve_fit(
                gaussian, bin_edges[:-1], hist, p0=[1, mean, std]
        )
    except RuntimeError:
        fit_args = None
    return bin_edges, fit_args, mean, std


def split_molecules(input_dictionary):
    # Split the full morphology into individual molecules
    molecule_AAIDs = []
    molecule_lengths = []
    # Create a lookup table `neighbor list' for all connected atoms called
    # {bondedAtoms}
    bonded_atoms = hf.obtain_bonded_list(input_dictionary["bond"])
    molecule_list = [i for i in range(len(input_dictionary["type"]))]
    # Recursively add all atoms in the neighbor list to this molecule
    for mol_ID in range(len(molecule_list)):
        molecule_list = update_molecule(mol_ID, molecule_list, bonded_atoms)
    # Create a dictionary of the molecule data
    molecule_data = {}
    for atom_ID in range(len(input_dictionary["type"])):
        if molecule_list[atom_ID] not in molecule_data:
            molecule_data[molecule_list[atom_ID]] = [atom_ID]
        else:
            molecule_data[molecule_list[atom_ID]].append(atom_ID)
    # Return the list of AAIDs and the lengths of the molecules
    for molecule_ID in list(molecule_data.keys()):
        molecule_AAIDs.append(sorted(molecule_data[molecule_ID]))
        molecule_lengths.append(len(molecule_data[molecule_ID]))
    return molecule_AAIDs, molecule_lengths


def update_molecule(atom_ID, molecule_list, bonded_atoms):
    # Recursively add all neighbors of atom number atomID to this molecule
    try:
        for bonded_atom in bonded_atoms[atom_ID]:
            # If the moleculeID of the bonded atom is larger than that of the
            # current one, update the bonded atom's ID to the current one's
            # to put it in this molecule, then iterate through all of the
            # bonded atom's neighbors
            if molecule_list[bonded_atom] > molecule_list[atom_ID]:
                molecule_list[bonded_atom] = molecule_list[atom_ID]
                molecule_list = update_molecule(
                    bonded_atom, molecule_list, bonded_atoms
                )
            # If the moleculeID of the current atom is larger than that of the bonded one,
            # update the current atom's ID to the bonded one's to put it in this molecule,
            # then iterate through all of the current atom's neighbors
            elif molecule_list[bonded_atom] < molecule_list[atom_ID]:
                molecule_list[atom_ID] = molecule_list[bonded_atom]
                molecule_list = update_molecule(
                    atom_ID, molecule_list, bonded_atoms
                )
            # Else: both the current and the bonded atom are already known to be in this
            # molecule, so we don't have to do anything else.
    except KeyError:
        # This means that there are no bonded CG sites (i.e. it's a single molecule)
        pass
    return molecule_list


def plot_neighbor_hist(
    chromo_list,
    chromo_to_mol_ID,
    morphology_shape,
    output_dir,
    sep_cut_donor,
    sep_cut_acceptor,
):
    separation_dist_donor = []
    separation_dist_acceptor = []
    for chromo1 in chromo_list:
        for chromo2_details in chromo1.neighbors:
            if (chromo2_details is None) or (
                chromo1.ID == chromo_list[chromo2_details[0]].ID
            ):
                continue
            chromo2 = chromo_list[chromo2_details[0]]
            # Skip any chromophores that are part of the same molecule
            if chromo_to_mol_ID[chromo1.ID] == chromo_to_mol_ID[chromo2.ID]:
                continue
            separation = np.linalg.norm(
                (
                    np.array(chromo2.pos)
                    + (
                        np.array(chromo2_details[1])
                        * np.array(morphology_shape)
                    )
                )
                - chromo1.pos
            )
            if chromo1.species == "donor":
                separation_dist_donor.append(separation)
            elif chromo1.species == "acceptor":
                separation_dist_acceptor.append(separation)
    material = ["donor", "acceptor"]
    sep_cuts = [sep_cut_donor, sep_cut_acceptor]
    for material_type, separation_dist in enumerate(
        [separation_dist_donor, separation_dist_acceptor]
    ):
        if len(separation_dist) == 0:
            continue
        plt.figure()
        n, bin_edges, _ = plt.hist(separation_dist, bins=40, color="b")
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        bin_centres = np.insert(bin_centres, 0, 0)
        n = np.insert(n, 0, 0)
        smoothed_n = gaussian_filter(n, 1.0)
        plt.plot(bin_centres, smoothed_n, color="r")
        if (sep_cuts[material_type] is not None) and (
            sep_cuts[material_type].lower() == "auto"
        ):
            sep_cuts[material_type] = calculate_cut_off_from_dist(
                bin_centres,
                smoothed_n,
                minimum_i=0,
                value_at_least=100,
                logarithmic=False,
            )
        if sep_cuts[material_type] is not None:
            print(
                "Cluster cut-off based on",
                material[material_type],
                "chromophore separation set to",
                sep_cuts[material_type],
            )
            plt.axvline(float(sep_cuts[material_type]), c="k")
        plt.xlabel(
            rf"{material[material_type].capitalize()} r$_{{i,j}}$ (\AA)"
        )
        plt.ylabel("Frequency (Arb. U.)")
        # 04 for donor neighbor hist, 05 for acceptor neighbor hist
        filename = "{:02}_neighbor_hist_{}.png".format(
                4 + material_type,
                material[material_type].lower(),
                )
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Neighbour histogram figure saved as {filepath}")
        plt.close()
    return sep_cuts[0], sep_cuts[1]


def plot_orientation_hist(
    chromo_list,
    chromo_to_mol_ID,
    orientations_data,
    output_dir,
    o_cut_donor,
    o_cut_acceptor,
):
    orientation_dist_donor = []
    orientation_dist_acceptor = []
    for chromo1 in chromo_list:
        orientation_1 = orientations_data[chromo1.ID]
        for chromo2_details in chromo1.neighbors:
            if (chromo2_details is None) or (
                chromo1.ID == chromo_list[chromo2_details[0]].ID
            ):
                continue
            chromo2 = chromo_list[chromo2_details[0]]
            # Skip any chromophores that are part of the same molecule
            if chromo_to_mol_ID[chromo1.ID] == chromo_to_mol_ID[chromo2.ID]:
                continue
            orientation_2 = orientations_data[chromo2_details[0]]
            dot_product = np.dot(orientation_1, orientation_2)
            separation_angle = np.arccos(np.abs(dot_product)) * 180 / np.pi
            if chromo1.species == "donor":
                orientation_dist_donor.append(separation_angle)
            elif chromo1.species == "acceptor":
                orientation_dist_acceptor.append(separation_angle)
    material = ["donor", "acceptor"]
    o_cuts = [o_cut_donor, o_cut_acceptor]
    for material_type, orientation_dist in enumerate(
        [orientation_dist_donor, orientation_dist_acceptor]
    ):
        if len(orientation_dist) == 0:
            continue
        plt.figure()
        n, bin_edges, _ = plt.hist(orientation_dist, bins=40, color="b")
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        bin_centres = np.insert(bin_centres, 0, 0)
        n = np.insert(n, 0, 0)
        smoothed_n = gaussian_filter(n, 1.0)
        plt.plot(bin_centres, smoothed_n, color="r")
        if (o_cuts[material_type] is not None) and (
            o_cuts[material_type].lower() == "auto"
        ):
            o_cuts[material_type] = calculate_cut_off_from_dist(
                bin_centres,
                smoothed_n,
                maximum_i=0,
                value_at_least=100,
                logarithmic=False,
            )
        if o_cuts[material_type] is not None:
            print(
                "Cluster cut-off based on",
                material[material_type],
                "relative chromophore orientations set to",
                o_cuts[material_type],
            )
            plt.axvline(float(o_cuts[material_type]), c="k")
            plt.xlabel(
                f"{material[material_type].capitalize()} Orientations (Deg)"
                )
        plt.xlim([0, 90])
        plt.xticks(np.arange(0, 91, 15))
        plt.ylabel("Frequency (Arb. U.)")
        # 34 for donor neighbor hist, 35 for acceptor neighbor hist
        filename = "{:02}_orientation_hist_{}.png".format(
                34 + material_type,
                material[material_type].lower()
                )
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Orientation histogram figure saved as {filepath}")
    return o_cuts[0], o_cuts[1]


def create_cut_off_dict(
    sep_cut_donor,
    sep_cut_acceptor,
    o_cut_donor,
    o_cut_acceptor,
    ti_cut_donor,
    ti_cut_acceptor,
    freq_cut_donor,
    freq_cut_acceptor,
):
    cut_off_dict = {
        "separation": [sep_cut_donor, sep_cut_acceptor],
        "orientation": [o_cut_donor, o_cut_acceptor],
        "TI": [ti_cut_donor, ti_cut_acceptor],
        "freq": [freq_cut_donor, freq_cut_acceptor],
    }
    # Convert all cuts to floats (leaving NoneType intact)
    for cut_type, cuts in cut_off_dict.items():
        for material_i, cut in enumerate(cuts):
            try:
                cut_off_dict[cut_type][material_i] = float(cut)
            except TypeError:
                pass
            except ValueError:
                # Most likely both donor and acceptor cuts have been specified
                # for this KMCA job, but only one type of carrier is available in
                # this instance, so we can safely set the opposing carrier type cut
                # to None.
                cut_off_dict[cut_type][material_i] = None
    return cut_off_dict


def get_clusters(
    chromo_list,
    carrier_history_dict,
    orientations_all,
    cut_off_dict,
    AA_morphdict,
):
    freud_box = [[AA_morphdict[coord] for coord in ["lx", "ly", "lz"]]]
    materials_to_check = ["donor", "acceptor"]
    carriers_to_check = ["hole", "electron"]
    cluster_freqs = [{}, {}]
    cluster_dicts = [{}, {}]
    clusters_total = [0, 0]
    clusters_large = [0, 0]
    clusters_biggest = [0, 0]
    species_psi = [0, 0]
    clusters_cutoffs = [[], []]
    if any(cut is not None for cut in cut_off_dict["separation"]):
        separations = get_separations(chromo_list, AA_morphdict)
    else:
        separations = None
    for type_i, material_type in enumerate(materials_to_check):
        material_chromos = [
            c for c in chromo_list if c.species == material_type
        ]
        print("Examining the", material_type, "material...")
        positions = np.array([chromo.pos for chromo in material_chromos])
        if len(positions) == 0:
            print("No material found. Continuing...")
            continue
        print("Obtaining orientations of each chromophore...")
        if cut_off_dict["orientation"][type_i] is not None:
            orientations = [orientations_all[c.ID] for c in material_chromos]
        else:
            orientations = None
        print("Calculating clusters...")
        n_list = get_n_list(
            chromo_list,
            carrier_history_dict[carriers_to_check[type_i]],
            separations=separations,
            r_cut=cut_off_dict["separation"][type_i],
            orientations=orientations,
            o_cut=cut_off_dict["orientation"][type_i],
            ti_cut=cut_off_dict["TI"][type_i],
            freq_cut=cut_off_dict["freq"][type_i],
            sim_dims=freud_box,
        )
        clusters_list = make_clusters(n_list)
        cluster_dict = {}
        for chromo_ID, cluster_ID in enumerate(clusters_list):
            cluster_dict[chromo_ID] = cluster_ID
        cluster_freq = {}
        for cluster_ID in set(clusters_list):
            cluster_freq[cluster_ID] = clusters_list.count(cluster_ID)
        species_psi[type_i] = sum(
                [p for p in cluster_freq.values() if p > 6]
        ) / len(clusters_list)
        clusters_total[type_i] = len(cluster_freq.keys())
        clusters_large[type_i] = len(
            [k for k, v in cluster_freq.items() if v > 6]
        )
        clusters_biggest[type_i] = np.max(list(cluster_freq.values()))
        clusters_cutoffs[type_i] = [
            cut_off_dict[cut_type][type_i]
            for cut_type in ["separation", "orientation", "TI", "freq"]
        ]
        print("----------------------------------------")
        print(f"{material_type}: Detected {clusters_total[type_i]} total")
        print(f"and large {clusters_large[type_i]} clusters (size > 6).")
        print(f"Largest cluster size ={clusters_biggest[type_i]} chromophores.")
        print(
            f'Ratio in "large" clusters: {species_psi[type_i]:.3f}'
            )
        print("----------------------------------------")
        cluster_dicts[type_i] = cluster_dict
        cluster_freqs[type_i] = cluster_freq
    return (
        cluster_dicts,
        cluster_freqs,
        clusters_total,
        clusters_large,
        clusters_biggest,
        clusters_cutoffs,
    )


def make_clusters(n_list):
    """
    Function to call for the creation
    of turning the neighbor list into a
    cluster list.
    Requires:
        n_list - neighbor list
    Returns:
        c_list - cluster list
    """
    sys.setrecursionlimit(int(5e4))
    c_list = [i for i in range(len(n_list))]
    for i in range(len(c_list)):
        n_list, c_list = update_neighbors(i, c_list, n_list)
    return c_list


def update_neighbors(particle, cluster_list, neighbor_list):
    """Recursive function to convert neighborlist into cluster list"""
    for n in neighbor_list[particle]:
        if cluster_list[n] > cluster_list[particle]:
            cluster_list[n] = cluster_list[particle]
            neighbor_list, cluster_list = update_neighbors(
                n, cluster_list, neighbor_list
            )
        elif cluster_list[n] < cluster_list[particle]:
            cluster_list[particle] = cluster_list[n]
            neighbor_list, cluster_list = update_neighbors(
                particle, cluster_list, neighbor_list
            )
    return neighbor_list, cluster_list


def get_n_list(
    chromo_list,
    carrier_history,
    separations,
    r_cut,
    orientations,
    o_cut,
    ti_cut,
    freq_cut,
    sim_dims,
):
    if o_cut is not None:
        # o_cut is currently an angle in degrees.
        # Need to calculate the corresponding dot-product cut off for this
        # angle
        dotcut = np.cos(o_cut * np.pi / 180)
    n_list = []
    for chromo in chromo_list:
        n_list.append([neighbor[0] for neighbor in chromo.neighbors])
    printing = False
    for chromo_ID, neighbors in enumerate(n_list):
        # if chromo_ID == 47:
        #    printing = True
        remove_list = []
        if printing is True:
            print(chromo_ID)
            print(chromo_list[chromo_ID].neighbors, "==", n_list[chromo_ID])
            print(chromo_list[chromo_ID].neighbors_TI)
        for neighbor_i, neighbor_ID in enumerate(neighbors):
            ti = chromo_list[chromo_ID].neighbors_TI[neighbor_i]
            if printing is True:
                print(
                    "Examining neighbor_index",
                    neighbor_i,
                    "which corresponds to chromo ID",
                    neighbor_ID,
                )
                print("TI =", ti)
            # Check the cut_offs in order of ease of calculation
            # Simple TI lookup first
            if (ti_cut is not None) and (ti < ti_cut):
                remove_list.append(neighbor_ID)
                if printing is True:
                    print("Adding", neighbor_ID, "to remove list as TI =", ti)
                    print("Remove list now =", remove_list)
                continue
            # Simple hop_frequency cutoff lookup in the carrier history
            if (freq_cut is not None) and (carrier_history is not None):
                total_hops = (
                    carrier_history[chromo_ID, neighbor_ID]
                    + carrier_history[neighbor_ID, chromo_ID]
                )
                if total_hops < freq_cut:
                    remove_list.append(neighbor_ID)
                    continue
            # Separation cutoff lookup in the separations matrix depends on the chromo_IDs
            if (r_cut is not None) and (separations is not None):
                if chromo_ID < neighbor_ID:
                    separation = separations[chromo_ID, neighbor_ID]
                else:
                    separation = separations[neighbor_ID, chromo_ID]
                if separation > r_cut:
                    remove_list.append(neighbor_ID)
                    continue
            # Some dot product manipulation is required to get the orientations right
            if (o_cut is not None) and (orientations is not None):
                chromo1_normal = orientations[chromo_ID]
                chromo2_normal = orientations[neighbor_ID]
                rotation_dot_product = abs(
                    np.dot(chromo1_normal, chromo2_normal)
                )
                if rotation_dot_product < dotcut:
                    remove_list.append(neighbor_ID)
                    if printing is True:
                        print(
                            "Adding",
                            neighbor_ID,
                            "to remove list as dot_product =",
                            rotation_dot_product,
                        )
                        print("Remove list now =", remove_list)
                    continue
        if printing is True:
            print("n_list_current =", n_list[chromo_ID])
            print("Remove list final =", remove_list)
        for neighbor_ID in remove_list:
            n_list[chromo_ID].remove(neighbor_ID)
            ##Will need to check the reverse remove.
            # n_list[neighbor_ID].remove(chromo_ID)
        if printing is True:
            print("n_list_after remove =", n_list[chromo_ID])
            exit()
        # input("".join(map(str, [chromo_ID, n_list[chromo_ID]])))
    return n_list


def get_orientations(
    chromo_list,
    CG_morphdict,
    AA_morphdict,
    CGtoAAID_list,
    param_dict,
):
    orientations = []
    for i, chromo in enumerate(chromo_list):
        positions = get_electronic_atom_positions(
            chromo,
            CG_morphdict,
            AA_morphdict,
            CGtoAAID_list,
            param_dict,
        )
        # There is a really cool way to do this with single value composition
        # but on the time crunch I didn't have time to learn how to implement
        # it properly. Check https://goo.gl/jxuhvJ for more details.
        plane = calculate_plane(positions)
        orientations.append(np.array(plane) / np.linalg.norm(plane))
    return orientations


def calculate_plane(positions):
    ## See https://goo.gl/jxuhvJ for details on this methodology.
    vec1 = hf.find_axis(positions[0], positions[1])
    vec2 = hf.find_axis(positions[0], positions[2])
    return np.cross(vec1, vec2)


def get_separations(chromo_list, AA_morphdict):
    box_dims = [AA_morphdict[axis] for axis in ["lx", "ly", "lz"]]
    n = len(chromo_list)
    separations = lil_matrix((n, n), dtype=float)
    for chromo_ID, chromo in enumerate(chromo_list):
        for neighbor_details in chromo.neighbors:
            neighbor_ID = neighbor_details[0]
            relative_image = neighbor_details[1]
            neighbor_pos = chromo_list[neighbor_ID].pos
            neighbor_chromo_pos = neighbor_pos + (
                np.array(relative_image) * np.array(box_dims)
            )
            separation = hf.calculate_separation(
                chromo.pos, neighbor_chromo_pos
            )
            if chromo_ID < neighbor_ID:
                separations[chromo_ID, neighbor_ID] = separation
            else:
                separations[neighbor_ID, chromo_ID] = separation
    return separations


def get_electronic_atom_positions(
    chromo,
    CG_morphdict,
    AA_morphdict,
    CGtoAAID_list,
    param_dict,
):
    # We don't save this in the chromophore info so we'll have to calculate it
    # again.
    # Determine whether this chromophore is a donor or an acceptor, as well
    # as the site types that have been defined as the electronically active
    # in the chromophore
    if CG_morphdict is not None:
        # Normal operation
        CG_types = sorted(
            list(set([CG_morphdict["type"][CGID] for CGID in chromo.CGIDs]))
        )
        active_CG_sites, _ = obtain_electronic_species(
            chromo.CGIDs,
            CG_morphdict["type"],
            param_dict["CG_site_species"],
        )
        # CGtoAAID_list is a list of dictionaries where each list
        # element corresponds to a new molecule. Firstly, flatten this out
        # so that it becomes a single CG:AAID dictionary
        flattened_CGtoAAID_list = {
            dict_key: dict_val[1]
            for dictionary in CGtoAAID_list
            for dict_key, dict_val in dictionary.items()
        }
        # By using active_CG_sites, determine the AAIDs for
        # the electrically active proportion of the chromophore, so that we
        # can calculate its proper position. Again each element corresponds
        # to each CG site so the list needs to be flattened afterwards.
        electronically_active_AAIDs = [
            AAID
            for AAIDs in [
                flattened_CGtoAAID_list[CGID] for CGID in active_CG_sites
            ]
            for AAID in AAIDs
        ]
    else:
        # No fine-graining has been performed by MorphCT, so we know that
        # the input morphology is already atomistic.
        if len(param_dict["CG_site_species"]) == 1:
            # If the morphology contains only a single type of electronic
            # species, then the param_dict['CG_site_species'] should
            # only have one entry, and we can set all chromophores to be
            # this species.
            active_CG_sites = chromo.CGIDs
            electronically_active_AAIDs = chromo.CGIDs
        elif (len(param_dict["CG_site_species"]) == 0) and (
            len(param_dict["AA_rigid_body_species"]) > 0
        ):
            # If the CG_site_species have not been specified, then look to
            # the AA_rigid_body_species dictionary to determine which rigid
            # bodies are donors and which are acceptors
            electronically_active_AAIDs = []
            for AAID in chromo.CGIDs:
                if AA_morphdict["body"][AAID] != -1:
                    electronically_active_AAIDs.append(AAID)
        else:
            raise SystemError(
                "Multiple electronic species defined, but no way to map them"
                " without a coarse-grained morphology "
                "(no CG morph has been given)"
            )
    # The position of the chromophore can be calculated easily. Note that
    # here, the `self.image' is the periodic image that the
    # unwrapped_position of the chromophore is located in, relative to the
    # original simulation volume.
    electronically_active_unwrapped_poss = [
        AA_morphdict["unwrapped_position"][AAID]
        for AAID in electronically_active_AAIDs
    ]
    return electronically_active_unwrapped_poss


def obtain_electronic_species(
    chromo_CG_sites, CG_site_types, CG_to_species
):
    electronically_active_sites = []
    current_chromo_species = None
    for CG_site_ID in chromo_CG_sites:
        site_type = CG_site_types[CG_site_ID]
        site_species = CG_to_species[site_type]
        if site_species.lower() != "none":
            if (current_chromo_species is not None) and (
                current_chromo_species != site_species
            ):
                raise SystemError(
                    "Problem - multiple electronic species defined in the same "
                    " chromophore. Please modify the chromophore generation "
                    "code  to fix this issue for your molecule!"
                )
            else:
                current_chromo_species = site_species
                electronically_active_sites.append(CG_site_ID)
    return electronically_active_sites, current_chromo_species


def update_cluster(atom_ID, cluster_list, neighbor_dict):
    try:
        for neighbor in neighbor_dict[atom_ID]:
            if cluster_list[neighbor] > cluster_list[atom_ID]:
                cluster_list[neighbor] = cluster_list[atom_ID]
                cluster_list = update_cluster(
                    neighbor, cluster_list, neighbor_dict
                )
            elif cluster_list[neighbor] < cluster_list[atom_ID]:
                cluster_list[atom_ID] = cluster_list[neighbor]
                cluster_list = update_cluster(
                    neighbor, cluster_list, neighbor_dict
                )
    except KeyError:
        pass
    return cluster_list


def write_cluster_tcl_script(output_dir, cluster_dict, large_cluster):
    """
    Create a tcl script for each identified cluster.
    """
    # Obtain the IDs of the cluster sizes, sorted by largest first
    print("Sorting the clusters by size...")
    cluster_order = list(zip(*sorted(
        zip([len(v) for v in cluster_dict.values()], cluster_dict.keys(),),
        reverse=True,
        )))[1]
    colors = list(range(int(1e6)))
    count = 0

    print("Creating tcl header...")
    tcl_text = ["mol delrep 0 0;"]
    tcl_text += ["pbc wrap -center origin;"]
    tcl_text += ["pbc box -color black -center origin -width 4;"]
    tcl_text += ["display resetview;"]
    tcl_text += ["color change rgb 9 1.0 0.29 0.5;"]
    tcl_text += ["color change rgb 16 0.25 0.25 0.25;"]

    for i, cluster_ID in enumerate(cluster_order):
        print(
            f"Creating tcl commands for cluster {i+1:d}/{len(cluster_order):d}",
            end=" ",
            )
        chromos = cluster_dict[cluster_ID]
        chromo_IDs = [c.ID for c in chromos if c.species == "donor"]
        if (len(chromo_IDs) > large_cluster):
            # Only make clusters that are ``large''
            inclust = ""
            for chromo in chromo_IDs:
                inclust += f"{inclust}{chromo:d} "
            tcl_text += ["mol material AOEdgy;"]  # Use AOEdgy if donor
            # The +1 makes the largest cluster red rather than blue (looks
            # better with AO, DoF, shadows)
            tcl_text += [f"mol color ColorID {colors[count + 1 % 32]:d};"]
            # VMD has 32 unique colors
            tcl_text += ["mol representation VDW 4.0 8.0;"]
            tcl_text += [f"mol selection resid {inclust:s};"]
            tcl_text += ["mol addrep 0;"]
            count += 1
        chromo_IDs = [c.ID for c in chromos if c.species == "acceptor"]
        if len(chromo_IDs) > large_cluster:
            inclust = ""
            for chromo in chromo_IDs:
                inclust += f"{inclust}{chromo:d} "
            tcl_text += ["mol material Glass2;"]  # Use Glass2 if acceptor
            # The +1 makes the largest cluster red rather than blue (looks
            # better with AO, DoF, shadows)
            tcl_text += [
                "mol color ColorID {:d};".format(colors[count + 1 % 32])
            ]
            tcl_text += ["mol representation VDW 4.0 8.0;"]
            tcl_text += ["mol selection resid {:s};".format(inclust)]
            tcl_text += ["mol addrep 0;"]
            count += 1
    tcl_file_path = os.path.join(
        output_dir.replace("figures", "morphology"), "cluster_colors.tcl"
    )
    with open(tcl_file_path, "w+") as tcl_file:
        tcl_file.writelines("".join(tcl_text))
    print("\nClusters coloring written to {:s}".format(tcl_file_path))


def generate_lists_for_3d_clusters(cluster_dict, colours, large_cluster):
    data = []
    for cluster_ID, chromos in cluster_dict.items():
        if len(chromos) > large_cluster:
            for chromo in chromos:
                if chromo.species == "donor":
                    data.append(
                        [
                            chromo.pos[0],
                            chromo.pos[1],
                            chromo.pos[2],
                            "w",
                            colours[cluster_ID % 7],
                        ]
                    )
                if chromo.species == "acceptor":
                    data.append(
                        [
                            chromo.pos[0],
                            chromo.pos[1],
                            chromo.pos[2],
                            colours[cluster_ID % 7],
                            "none",
                        ]
                    )
    # Needs a sort so that final image is layered properly
    data = list(sorted(data, key=lambda x: x[0]))
    # Split up list into sublists
    xs = [row[0] for row in data]
    ys = [row[1] for row in data]
    zs = [row[2] for row in data]
    face_colors = [row[3] for row in data]
    edge_colors = [row[4] for row in data]
    return xs, ys, zs, face_colors, edge_colors


def plot_clusters_3D(
    output_dir, chromo_list, cluster_dicts, sim_dims, generate_tcl
):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    colours = ["r", "g", "b", "c", "m", "y", "k"]
    large_cluster = 6
    cluster_dict = {}
    for dictionary in cluster_dicts:
        if dictionary is not None:
            cluster_dict.update(dictionary)
    cluster_dict = {}
    for chromo_ID, cluster_ID in cluster_dict.items():
        if cluster_ID not in cluster_dict.keys():
            cluster_dict[cluster_ID] = []
        else:
            cluster_dict[cluster_ID].append(chromo_list[chromo_ID])
    if generate_tcl:
        write_cluster_tcl_script(output_dir, cluster_dict, large_cluster)

    xs, ys, zs, face_colors, edge_colors = generate_lists_for_3d_clusters(
        cluster_dict, colours, large_cluster
    )
    ax.scatter(
        xs,
        ys,
        zs,
        facecolors=face_colors,
        edgecolors=edge_colors,
        alpha=0.6,
        s=40,
    )
    # Draw boxlines
    # Varying X
    ax.plot(
        [sim_dims[0][0], sim_dims[0][1]],
        [sim_dims[1][0], sim_dims[1][0]],
        [sim_dims[2][0], sim_dims[2][0]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][0], sim_dims[0][1]],
        [sim_dims[1][1], sim_dims[1][1]],
        [sim_dims[2][0], sim_dims[2][0]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][0], sim_dims[0][1]],
        [sim_dims[1][0], sim_dims[1][0]],
        [sim_dims[2][1], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][0], sim_dims[0][1]],
        [sim_dims[1][1], sim_dims[1][1]],
        [sim_dims[2][1], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    # Varying Y
    ax.plot(
        [sim_dims[0][0], sim_dims[0][0]],
        [sim_dims[1][0], sim_dims[1][1]],
        [sim_dims[2][0], sim_dims[2][0]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][1], sim_dims[0][1]],
        [sim_dims[1][0], sim_dims[1][1]],
        [sim_dims[2][0], sim_dims[2][0]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][0], sim_dims[0][0]],
        [sim_dims[1][0], sim_dims[1][1]],
        [sim_dims[2][1], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][1], sim_dims[0][1]],
        [sim_dims[1][0], sim_dims[1][1]],
        [sim_dims[2][1], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    # Varying Z
    ax.plot(
        [sim_dims[0][0], sim_dims[0][0]],
        [sim_dims[1][0], sim_dims[1][0]],
        [sim_dims[2][0], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][0], sim_dims[0][0]],
        [sim_dims[1][1], sim_dims[1][1]],
        [sim_dims[2][0], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][1], sim_dims[0][1]],
        [sim_dims[1][0], sim_dims[1][0]],
        [sim_dims[2][0], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.plot(
        [sim_dims[0][1], sim_dims[0][1]],
        [sim_dims[1][1], sim_dims[1][1]],
        [sim_dims[2][0], sim_dims[2][1]],
        c="k",
        linewidth=1.0,
    )
    ax.set_xlim([sim_dims[0][0], sim_dims[0][1]])
    ax.set_ylim([sim_dims[1][0], sim_dims[1][1]])
    ax.set_zlim([sim_dims[2][0], sim_dims[2][1]])
    # 03 for clusters (material agnostic)
    plt.savefig(
        os.path.join(output_dir, "03_clusters.png"),
        bbox_inches="tight",
        dpi=300,
    )
    print(
        "3D cluster figure saved as",
        os.path.join(output_dir, "03_clusters.png"),
    )
    plt.close()


def determine_molecule_IDs(
        CGtoAAID_list, AA_morphdict, param_dict, chromo_list
):
    print("Determining molecule IDs...")
    CGID_to_mol_ID = {}
    if CGtoAAID_list is not None:
        # Normal operation with a CGMorphology defined
        # (fine-graining was performed)
        for mol_ID, mol_dict in enumerate(CGtoAAID_list):
            for CGID in list(mol_dict.keys()):
                CGID_to_mol_ID[CGID] = mol_ID
    elif len(param_dict["CG_site_species"]) == 1 and (
        "AA_rigid_body_species" not in param_dict
        or param_dict["AA_rigid_body_species"]
    ):
        print(
                "Small-molecule system detected, assuming each chromophore is",
                " its own molecule..."
                )
        # When CGMorphology doesn't exist, and no rigid body species have been
        # specified, then every chromophore is its own molecule)
        for i, chromo in enumerate(chromo_list):
            for CGID in chromo.CGIDs:
                CGID_to_mol_ID[CGID] = chromo.ID
    else:
        # No CGMorphology, but not small molecules either, so determine
        # molecules based on bonds
        print(
                "Polymeric system detected, determining molecules based on AA",
                " bonds(slow calculation)..."
                )
        molecule_AAIDs, molecule_lengths = split_molecules(AA_morphdict)
        for i, molecule_AAID_list in enumerate(molecule_AAIDs):
            for AAID in molecule_AAID_list:
                CGID_to_mol_ID[AAID] = i
    # Convert to chromo_to_mol_ID
    chromo_to_mol_ID = {}
    for chromo in chromo_list:
        first_CGID = chromo.CGIDs[0]
        chromo_to_mol_ID[chromo.ID] = CGID_to_mol_ID[first_CGID]
    return chromo_to_mol_ID


def plot_energy_levels(output_dir, chromo_list, data_dict):
    HOMO_levels = []
    LUMO_levels = []
    donor_delta_E_ij = []
    acceptor_delta_E_ij = []
    donor_lambda_ij = None
    acceptor_lambda_ij = None
    for chromo in chromo_list:
        if chromo.species == "donor":
            HOMO_levels.append(chromo.HOMO)
            for neighbor_ind, delta_E_ij in enumerate(chromo.neighbors_delta_E):
                if (delta_E_ij is not None) and (
                    chromo.neighbors_TI[neighbor_ind] is not None
                ):
                    donor_delta_E_ij.append(delta_E_ij)
                if "reorganisation_energy" in chromo.__dict__:
                    donor_lambda_ij = chromo.reorganisation_energy
        else:
            LUMO_levels.append(chromo.LUMO)
            for neighbor_ind, delta_E_ij in enumerate(chromo.neighbors_delta_E):
                if (delta_E_ij is not None) and (
                    chromo.neighbors_TI[neighbor_ind] is not None
                ):
                    acceptor_delta_E_ij.append(delta_E_ij)
                if "reorganisation_energy" in chromo.__dict__:
                    acceptor_lambda_ij = chromo.reorganisation_energy
    if len(donor_delta_E_ij) > 0:
        (donor_bin_edges,
         donor_fit_args,
         donor_mean,
         donor_std) = gauss_fit(donor_delta_E_ij)
        data_dict["donor_delta_E_ij_mean"] = donor_mean
        data_dict["donor_delta_E_ij_std"] = donor_std
        data_dict["donor_delta_E_ij_err"] = donor_std / np.sqrt(
            len(donor_delta_E_ij)
        )
        HOMO_av = np.average(HOMO_levels)
        HOMO_std = np.std(HOMO_levels)
        HOMO_err = HOMO_std / np.sqrt(len(HOMO_levels))
        data_dict["donor_frontier_MO_mean"] = HOMO_av
        data_dict["donor_frontier_MO_std"] = HOMO_std
        data_dict["donor_frontier_MO_err"] = HOMO_err
        print(f"Donor HOMO Level = {HOMO_av:.3f} +/- {HOMO_err:.3f}")
        print("Donor Delta E_ij stats: mean = {:.3f} +/- {:.3f}".format(
            donor_mean,
            donor_std / np.sqrt(len(donor_delta_E_ij))
            ))
        # 06 for donor delta Eij
        plot_delta_E_ij(
            donor_delta_E_ij,
            donor_bin_edges,
            donor_fit_args,
            "donor",
            os.path.join(output_dir, "06_donor_delta_E_ij.png"),
            donor_lambda_ij,
        )
    if len(acceptor_delta_E_ij) > 0:
        (acceptor_bin_edges,
         acceptor_fit_args,
         acceptor_mean,
         acceptor_std) = gauss_fit(acceptor_delta_E_ij)
        data_dict["acceptor_delta_E_ij_mean"] = acceptor_mean
        data_dict["acceptor_delta_E_ij_std"] = acceptor_std
        data_dict["acceptor_delta_E_ij_err"] = acceptor_std / np.sqrt(
            len(acceptor_delta_E_ij)
        )
        LUMO_av = np.average(LUMO_levels)
        LUMO_std = np.std(LUMO_levels)
        LUMO_err = LUMO_std / np.sqrt(len(LUMO_levels))
        data_dict["acceptor_frontier_MO_mean"] = LUMO_av
        data_dict["acceptor_frontier_MO_std"] = LUMO_std
        data_dict["acceptor_frontier_MO_err"] = LUMO_err
        print("Acceptor LUMO Level =", LUMO_av, "+/-", LUMO_err)
        print(
            "Acceptor Delta E_ij stats: mean =",
            acceptor_mean,
            "+/-",
            acceptor_std / np.sqrt(len(acceptor_delta_E_ij)),
        )
        # 07 for acceptor delta Eij
        plot_delta_E_ij(
            acceptor_delta_E_ij,
            acceptor_bin_edges,
            acceptor_fit_args,
            "acceptor",
            os.path.join(output_dir, "07_acceptor_delta_E_ij.png"),
            acceptor_lambda_ij,
        )
    return data_dict


def plot_delta_E_ij(
    delta_E_ij, gauss_bins, fit_args, data_type, filename, lambda_ij
):
    plt.figure()
    n, bins, patches = plt.hist(
        delta_E_ij,
        np.arange(np.min(delta_E_ij), np.max(delta_E_ij), 0.05),
        color=["b"],
    )
    if fit_args is not None:
        gauss_Y = gaussian(gauss_bins[:-1], *fit_args)
        scale_factor = max(n) / max(gauss_Y)
        plt.plot(gauss_bins[:-1], gauss_Y * scale_factor, "ro:")
    else:
        print("No Gaussian found (probably zero-width delta function)")
    if lambda_ij is not None:
        plt.axvline(-float(lambda_ij), c="k")
    plt.ylabel("Frequency (Arb. U.)")
    plt.xlabel(rf"{data_type.capitalize()} $\Delta E_{{i,j}}$ (eV)")
    plt.savefig(filename, dpi=300)
    plt.close()
    print("Figure saved as", filename)


def plot_mixed_hopping_rates(
    output_dir,
    chromo_list,
    param_dict,
    cluster_dicts,
    chromo_to_mol_ID,
    data_dict,
    AA_morphdict,
    cut_off_dict,
):
    # Create all the empty lists we need
    hop_types = ["intra_", "inter_"]
    hop_targets = ["c", "m"]
    # hop_properties = ["r", "T"]
    chromo_species = ["d", "a"]
    prop_lists = defaultdict(list)
    T = 290
    for chromo in chromo_list:
        mol1ID = chromo_to_mol_ID[chromo.ID]
        for i, T_ij in enumerate(chromo.neighbors_TI):
            if (T_ij is None) or (T_ij == 0):
                continue
            chromo2 = chromo_list[chromo.neighbors[i][0]]
            mol2ID = chromo_to_mol_ID[chromo2.ID]
            delta_E = chromo.neighbors_delta_E[i]
            if chromo.sub_species == chromo2.sub_species:
                lambda_ij = chromo.reorganisation_energy
            else:
                lambda_ij = (
                    chromo.reorganisation_energy + chromo2.reorganisation_energy
                ) / 2
            # Now take into account the various behaviours we can have from
            # the parameter file
            prefactor = 1.0
            # Apply the koopmans prefactor
            try:
                use_koop = param_dict["use_koopmans_approximation"]
                if use_koop:
                    prefactor *= param_dict["koopmans_hopping_prefactor"]
            except KeyError:
                pass
            # Apply the simple energetic penalty model
            try:
                boltz_pen = param_dict["use_simple_energetic_penalty"]
            except KeyError:
                boltz_pen = False
            # Apply the distance penalty due to VRH
            try:
                VRH = param_dict["use_VRH"]
                if VRH is True:
                    VRH_delocalisation = 1.0 / chromo.VRH_delocalisation
            except KeyError:
                VRH = False
            if VRH is True:
                relative_image = chromo.neighbors[i][1]
                neighbor_chromo_pos = chromo2.pos + (
                    np.array(relative_image)
                    * np.array(
                        [AA_morphdict[axis] for axis in ["lx", "ly", "lz"]]
                    )
                )
                chromo_separation = (
                    hf.calculate_separation(chromo.pos, neighbor_chromo_pos)
                    * 1e-10
                )
                rate = hf.calculate_carrier_hop_rate(
                    lambda_ij * elem_chrg,
                    T_ij * elem_chrg,
                    delta_E * elem_chrg,
                    prefactor,
                    T,
                    use_VRH=VRH,
                    rij=chromo_separation,
                    VRH_delocalisation=VRH_delocalisation,
                    boltz_pen=boltz_pen,
                )
            else:
                rate = hf.calculate_carrier_hop_rate(
                    lambda_ij * elem_chrg,
                    T_ij * elem_chrg,
                    delta_E * elem_chrg,
                    prefactor,
                    T,
                    boltz_pen=boltz_pen,
                )
            if chromo2.ID < chromo.ID:
                continue
            # Do intra- / inter- clusters
            if chromo.species == "acceptor":
                if (
                    cluster_dicts[1][chromo.ID]
                    == cluster_dicts[1][chromo.neighbors[i][0]]
                ):
                    prop_lists["intra_cra"].append(rate)
                    prop_lists["intra_cTa"].append(T_ij)
                else:
                    prop_lists["inter_cra"].append(rate)
                    prop_lists["inter_cTa"].append(T_ij)
            else:
                if (
                    cluster_dicts[0][chromo.ID]
                    == cluster_dicts[0][chromo.neighbors[i][0]]
                ):
                    prop_lists["intra_crd"].append(rate)
                    prop_lists["intra_cTd"].append(T_ij)
                else:
                    prop_lists["inter_crd"].append(rate)
                    prop_lists["inter_cTd"].append(T_ij)
            # Now do intra- / inter- molecules
            if mol1ID == mol2ID:
                if chromo.species == "acceptor":
                    prop_lists["intra_mra"].append(rate)
                    prop_lists["intra_mTa"].append(T_ij)
                else:
                    prop_lists["intra_mrd"].append(rate)
                    prop_lists["intra_mTd"].append(T_ij)
            else:
                if chromo.species == "acceptor":
                    prop_lists["inter_mra"].append(rate)
                    prop_lists["inter_mTa"].append(T_ij)
                else:
                    prop_lists["inter_mrd"].append(rate)
                    prop_lists["inter_mTd"].append(T_ij)
    # 12 for the donor cluster TI, 13 for the acceptor cluster TI, 14 for
    # the donor mol kij, 15 for the acceptor mol kij, 16 for the donor
    # cluster kij, 17 for the acceptor cluster kij
    # Donor cluster Plots:
    if prop_lists["intra_crd"]:
        val = prop_lists["intra_crd"]
        mean = np.mean(val)
        avg_std = np.std(val) / len(val)
        print(f"Mean intra-cluster donor rate: {mean:.3e}+/-{avg_std:.3e}")

    if prop_lists["inter_crd"]:
        val = prop_lists["inter_crd"]
        mean = np.mean(val)
        avg_std = np.std(val) / len(val)
        print(f"Mean inter-cluster donor rate: {mean:.3e}+/-{avg_std:.3e}")

    if prop_lists["intra_crd"] or prop_lists["inter_crd"]:
        plot_stacked_hist_rates(
            prop_lists["intra_crd"],
            prop_lists["inter_crd"],
            ["Intra-cluster", "Inter-cluster"],
            "donor",
            os.path.join(output_dir, "16_donor_hopping_rate_clusters.png"),
        )
        plot_stacked_hist_TIs(
            prop_lists["intra_cTd"],
            prop_lists["inter_cTd"],
            ["Intra-cluster", "Inter-cluster"],
            "donor",
            os.path.join(output_dir, "12_donor_transfer_integral_clusters.png"),
            cut_off_dict["TI"][0],
        )
    # Acceptor cluster Plots:
    if prop_lists["intra_cra"]:
        val = prop_lists["intra_cra"]
        mean = np.mean(val)
        avg_std = np.std(val) / len(val)
        print(f"Mean intra-cluster acceptor rate: {mean:.3e}+/-{avg_std:.3e}")

    if prop_lists["inter_cra"]:
        val = prop_lists["inter_cra"]
        mean = np.mean(val)
        avg_std = np.std(val) / len(val)
        print(f"Mean inter-cluster acceptor rate: {mean:.3e}+/-{avg_std:.3e}")

    if prop_lists["intra_cra"] or prop_lists["inter_cra"]:
        plot_stacked_hist_rates(
            prop_lists["intra_cra"],
            prop_lists["inter_cra"],
            ["Intra-cluster", "Inter-cluster"],
            "acceptor",
            os.path.join(output_dir, "17_acceptor_hopping_rate_clusters.png"),
        )
        plot_stacked_hist_TIs(
            prop_lists["intra_cTa"],
            prop_lists["inter_cTa"],
            ["Intra-cluster", "Inter-cluster"],
            "acceptor",
            os.path.join(
                output_dir, "13_acceptor_transfer_integral_clusters.png"
            ),
            cut_off_dict["TI"][1],
        )
    # Donor Mol Plots:
    if prop_lists["intra_mrd"]:
        val = prop_lists["intra_mrd"]
        mean = np.mean(val)
        avg_std = np.std(val) / len(val)
        print(f"Mean intra-molecular donor rate: {mean:.3e}+/-{avg_std:.3e}")

    if prop_lists["inter_mrd"]:
        val = prop_lists["inter_mrd"]
        mean = np.mean(val)
        avg_std = np.std(val) / len(val)
        print(f"Mean inter-molecular donor rate: {mean:.3e}+/-{avg_std:.3e}")

    if prop_lists["intra_mrd"] or prop_lists["inter_mrd"]:
        plot_stacked_hist_rates(
            prop_lists["intra_mrd"],
            prop_lists["inter_mrd"],
            ["Intra-mol", "Inter-mol"],
            "donor",
            os.path.join(output_dir, "14_donor_hopping_rate_mols.png"),
        )
    # Acceptor Mol Plots:
    if prop_lists["intra_mra"]:
        val = prop_lists["intra_mra"]
        mean = np.mean(val)
        avg_std = np.std(val) / len(val)
        print(f"Mean intra-molecular acceptor rate: {mean:.3e}+/-{avg_std:.3e}")

    if prop_lists["inter_mra"]:
        val = prop_lists["inter_mra"]
        mean = np.mean(val)
        avg_std = np.std(val) / len(val)
        print(f"Mean inter-molecular acceptor rate: {mean:.3e}+/-{avg_std:.3e}")

    if prop_lists["intra_mra"] or prop_lists["inter_mra"]:
        plot_stacked_hist_rates(
            prop_lists["intra_mra"],
            prop_lists["inter_mra"],
            ["Intra-mol", "Inter-mol"],
            "acceptor",
            os.path.join(output_dir, "15_acceptor_hopping_rate_mols.png"),
        )
    # Update the dataDict
    for sp in chromo_species:
        for hop_type in hop_types:
            for target in hop_targets:
                hop_name = f"{hop_type}_{target}r{sp}"
                n_hops = len(prop_lists[hop_name])
                if n_hops == 0:
                    continue

                other_hop = hop_types[hop_types.index(hop_type) * -1 + 1]
                other_hop_name = f"{other_hop}_{target}r{sp}"

                total_hops = n_hops + len(prop_lists[other_hop_name])
                proportion = n_hops / total_hops

                mean_rate = np.mean(prop_lists[hop_name])
                dev_rate = np.std(prop_lists[hop_name])
                data_dict[f"{sp}_{hop_type}_{target}_hops"]= n_hops
                data_dict[f"{sp}_{hop_type}_{target}_proportion"] = proportion
                data_dict[f"{sp}_{hop_type}_{target}_rate_mean"] = mean_rate
                data_dict[f"{sp}_{hop_type}_{target}_rate_std"] = dev_rate
    return data_dict


def plot_stacked_hist_rates(data1, data2, labels, data_type, filename):
    plt.figure()
    (n, bins, patches) = plt.hist(
        [data1, data2],
        bins=np.logspace(1, 18, 40),
        stacked=True,
        color=["r", "b"],
        label=labels,
    )
    plt.ylabel("Frequency (Arb. U.)")
    plt.xlabel(rf"{data_type.capitalize()} k$_{{i,j}}$ (s$^{-1}$)")
    plt.xlim([1, 1e18])
    plt.xticks([1e0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18])
    plt.ylim([0, np.max(n) * 1.02])
    plt.legend(loc=2, prop={"size": 18})
    plt.gca().set_xscale("log")
    plt.savefig(filename, dpi=300)
    plt.close()
    print("Figure saved as", filename)


def plot_stacked_hist_TIs(data1, data2, labels, data_type, filename, cut_off):
    plt.figure()
    (n, bins, patches) = plt.hist(
        [data1, data2],
        bins=np.linspace(0, 1.2, 20),
        stacked=True,
        color=["r", "b"],
        label=labels,
    )
    plt.ylabel("Frequency (Arb. U.)")
    plt.xlabel(rf"{data_type.capitalize()} J$_{{i,j}}$ (eV)")
    # plt.xlim([0, 1.2])
    plt.ylim([0, np.max(n) * 1.02])
    if cut_off is not None:
        plt.axvline(float(cut_off), c="k")
    plt.legend(loc=0, prop={"size": 18})
    plt.savefig(filename, dpi=300)
    plt.close()
    print("Figure saved as", filename)


def write_CSV(data_dict, path):
    CSV_filename = os.path.join(path, "results.csv")
    with open(CSV_filename, "w+") as CSV_file:
        CSV_writer = csv.writer(CSV_file)
        for key in sorted(data_dict.keys()):
            CSV_writer.writerow([key, data_dict[key]])
    print("CSV file written to {:s}".format(CSV_filename))


def create_results_pickle(path):
    cores_list = []
    for filename in glob.glob(os.path.join(path, "KMC", "*")):
        if "log" not in filename:
            continue
        try:
            cores_list.append(
                os.path.split(filename)[1].split(".")[0].split("_")[-1]
            )
        except IndexError:
            pass
    cores_list = sorted(list(set(cores_list)))
    results_pickles_list = []
    keep_list = []
    for core in cores_list:
        # Check if there is already a finished KMC_results pickle
        main = os.path.join(path, "KMC", f"KMC_results_{core:02d}.pickle")
        if os.path.exists(main):
            results_pickles_list.append(main)
            keep_list.append(None)
            continue
        # If not, find the slot1 and slot2 pickle that is most recent
        slot1 = os.path.join(
            path, "KMC", f"KMC_slot1_results_{core:02d}.pickle",
        )
        slot2 = os.path.join(
            path, "KMC", f"KMC_slot2_results_{core:02d}.pickle"
        )
        if os.path.exists(slot1) and not os.path.exists(slot2):
            keep_list.append(slot1)
        elif os.path.exists(slot2) and not os.path.exists(slot1):
            keep_list.append(slot2)
        elif os.path.getsize(slot1) >= os.path.getsize(slot2):
            keep_list.append(slot1)
        else:
            keep_list.append(slot2)
    print(f"{len(keep_list):d} pickle files found to combine!")
    print("Combining", keep_list)
    for keeper in zip(cores_list, keep_list):
        # Skip this core if we already have a finished KMC_results for it
        if keeper[1] is None:
            continue
        new_name = os.path.join(
                path, "KMC", f"KMC_results_{keeper[0]}.pickle"
        )
        shutil.copyfile(str(keeper[1]), new_name)
        results_pickles_list.append(new_name)
    combine_results_pickles(path, results_pickles_list)


def combine_results_pickles(path, pickle_files):
    combined_data = {}
    pickle_files = sorted(pickle_files)
    for filename in pickle_files:
        # The pickle was repeatedly dumped to, in order to save time.
        # Each dump stream is self-contained, so iteratively unpickle
        # to add the new data.
        with open(filename, "rb") as pickle_file:
            pickled_data = pickle.load(pickle_file)
            for key, val in pickled_data.items():
                if val is None:
                    continue
                if key not in combined_data:
                    combined_data[key] = val
                else:
                    combined_data[key] += val
    # Write out the combined data
    print("Writing out the combined pickle file...")
    combined_file_loc = os.path.join(path, "KMC", "KMC_results.pickle")
    with open(combined_file_loc, "wb+") as pickle_file:
        pickle.dump(combined_data, pickle_file)
    print("Complete data written to", combined_file_loc)


def calculate_cut_off_from_dist(
    bin_centres,
    frequencies,
    minimum_i=None,
    maximum_i=None,
    value_at_least=100,
    logarithmic=False,
):
    try:
        if minimum_i is not None:
            # Looking for minima
            minima = argrelextrema(frequencies, np.less)[0]
            if minimum_i < 0:
                # Sometimes a tiny minimum at super RHS breaks this
                cut_off = 0.0
                while True:
                    selected_minimum = minima[minimum_i]
                    cut_off = bin_centres[selected_minimum]
                    if frequencies[selected_minimum] > value_at_least:
                        break
                    minimum_i -= 1
            else:
                # Sometimes a tiny maximum at super LHS breaks this
                cut_off = 0.0
                while True:
                    selected_minimum = minima[minimum_i]
                    cut_off = bin_centres[selected_minimum]
                    if frequencies[selected_minimum] > value_at_least:
                        break
                    minimum_i += 1
        elif maximum_i is not None:
            # Looking for maxima
            maxima = argrelextrema(frequencies, np.greater)[0]
            if maximum_i < 0:
                # Sometimes a tiny maximum at super RHS breaks this
                cut_off = 0.0
                while True:
                    selected_maximum = maxima[maximum_i]
                    cut_off = bin_centres[selected_maximum]
                    if frequencies[selected_maximum] > value_at_least:
                        break
                    maximum_i -= 1
            else:
                # Sometimes a tiny maximum at super LHS breaks this
                cut_off = 0.0
                while True:
                    selected_maximum = maxima[maximum_i]
                    cut_off = bin_centres[selected_maximum]
                    if frequencies[selected_maximum] > value_at_least:
                        break
                    maximum_i += 1
        if logarithmic is True:
            cut_off = 10 ** cut_off
        # Return as string, as it will be converted to a float later
        return str(cut_off)
    except IndexError:
        print(
            "EXCEPTION: No minima found in frequency distribution.",
            " Setting cut_off to None."
        )
        return None


def plot_TI_hist(
    chromo_list,
    chromo_to_mol_ID,
    output_dir,
    TI_cut_donor,
    TI_cut_acceptor,
):
    # TI_dist [[DONOR], [ACCEPTOR]]
    TI_dist_intra = [[], []]
    TI_dist_inter = [[], []]
    material = ["donor", "acceptor"]
    labels = ["Intra-mol", "Inter-mol"]
    TI_cuts = [TI_cut_donor, TI_cut_acceptor]
    for material_i, material_type in enumerate(material):
        for chromo1 in chromo_list:
            for neighbor_i, chromo2_details in enumerate(chromo1.neighbors):
                chromo2 = chromo_list[chromo2_details[0]]
                if (chromo2_details is None) or (chromo1.ID >= chromo2.ID):
                    continue
                if chromo_to_mol_ID[chromo1.ID] == chromo_to_mol_ID[chromo2.ID]:
                    TI_dist_intra[material_i].append(
                        chromo1.neighbors_TI[neighbor_i]
                    )
                else:
                    TI_dist_inter[material_i].append(
                        chromo1.neighbors_TI[neighbor_i]
                    )
        if not TI_dist_intra[material_i] and TI_dist_inter[material_i]:
            continue
        plt.figure()
        maxTI = np.max(
                TI_dist_intra[material_i] + TI_dist_inter[material_i]
                )
        n, bin_edges, _ = plt.hist(
            [TI_dist_intra[material_i], TI_dist_inter[material_i]],
            bins=np.linspace(0, maxTI, 20),
            color=["r", "b"],
            stacked=True,
            label=labels,
        )
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        smoothed_n = gaussian_filter(n[0] + n[1], 1.0)
        plt.plot(bin_centres, smoothed_n, color="r")
        if TI_cuts[material_i] is not None and TI_cuts[material_i] == "auto":
            TI_cuts[material_i] = calculate_cut_off_from_dist(
                bin_centres, smoothed_n, minimum_i=-1, value_at_least=100
            )
        if TI_cuts[material_i] is not None:
            print(
                f"Cluster cut-off based on {material[material_i]} ",
                f"transfer integrals set to {TI_cuts[material_i]}"
            )
            plt.axvline(float(TI_cuts[material_i]), c="k")
        plt.xlim(
            [0, np.max(TI_dist_intra[material_i] + TI_dist_inter[material_i])]
        )
        plt.ylim([0, np.max(n) * 1.02])
        plt.ylabel("Frequency (Arb. U.)")
        plt.xlabel(rf"{material[material_i].capitalize()} J$_{{i,j}}$ (eV)")
        plt.legend(loc=1, prop={"size": 18})
        # 10 for donor TI mols dist, 11 for acceptor TI mols dist,
        filename = "{:02}_{}_transfer_integral_mols.png".format(
                10 + material_i,
                material_type
                )
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Figure saved as {filepath}")
    return TI_cuts[0], TI_cuts[1]


def plot_frequency_dist(path, carrier_type, carrier_history, cut_off):
    carrier_types = ["hole", "electron"]
    non_zero_indices = carrier_history.nonzero()
    coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
    frequencies = []
    for coords in coordinates:
        if coords[1] < coords[0]:
            # Only consider hops in one direction
            continue
        frequency = carrier_history[coords]
        frequencies.append(np.log10(frequency))
    plt.figure()
    n, bin_edges, _ = plt.hist(frequencies, bins=60, color="b")
    bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    smoothed_n = gaussian_filter(n, 1.0)
    plt.plot(bin_centres, smoothed_n, color="r")
    if (cut_off is not None) and (cut_off.lower() == "auto"):
        print("DYNAMIC CUT")
        cut_off = calculate_cut_off_from_dist(
            bin_centres,
            smoothed_n,
            minimum_i=-1,
            value_at_least=100,
            logarithmic=True,
        )
    if cut_off is not None:
        print("Cluster cut-off based on hop frequency set to", cut_off)
        plt.axvline(np.log10(float(cut_off)), c="k")
    plt.xlabel("".join(["Total ", carrier_type, " hops (Arb. U.)"]))
    ax = plt.gca()
    # tick_labels = np.arange(0, 7, 1)
    # plt.xlim([0, 6])
    tick_labels = np.arange(0, np.ceil(np.max(frequencies)) + 1, 1)
    plt.xlim([0, np.ceil(np.max(frequencies))])
    plt.xticks(
        tick_labels, [r"10$^{{{}}}$".format(int(x)) for x in tick_labels]
    )
    plt.ylabel("Frequency (Arb. U.)")
    # 24 for hole hop frequency dist, 25 for electron hop frequency dist
    filename = "".join(
        [
            "{:02}_total_hop_freq_".format(
                24 + carrier_types.index(carrier_type)
            ),
            carrier_type,
            ".png",
        ]
    )
    plt.savefig(os.path.join(path, "figures", filename), dpi=300)
    plt.close()
    print("Figure saved as", os.path.join(path, "figures", filename))
    return cut_off


def plot_net_frequency_dist(path, carrier_type, carrier_history):
    carrier_types = ["hole", "electron"]
    non_zero_indices = carrier_history.nonzero()
    coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
    frequencies = []
    for coords in coordinates:
        if coords[1] < coords[0]:
            # Only consider hops in one direction
            continue
        frequency = np.abs(
            carrier_history[coords] - carrier_history[coords[::-1]]
        )
        if frequency > 0:
            frequencies.append(np.log10(frequency))
    plt.figure()
    plt.hist(frequencies, bins=60, color="b")
    plt.xlabel("".join(["Net ", carrier_type, " hops (Arb. U.)"]))
    ax = plt.gca()
    tick_labels = np.arange(0, np.ceil(np.max(frequencies)) + 1, 1)
    plt.xlim([0, np.ceil(np.max(frequencies))])
    plt.xticks(tick_labels, [rf"10$^{{{x}}}$" for x in tick_labels])
    plt.ylabel("Frequency (Arb. U.)")
    # 26 for hole hop frequency dist, 27 for electron hop frequency dist
    filename = "{:02}_net_hop_freq_{}.png".format(
            26 + carrier_types.index(carrier_type),
            carrier_type
            )
    filepath = os.path.join(path, "figures", filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Figure saved as {filepath}")


def plot_discrepancy_frequency_dist(path, carrier_type, carrier_history):
    carrier_types = ["hole", "electron"]
    non_zero_indices = carrier_history.nonzero()
    coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
    frequencies = []
    net_equals_total = 0
    net_near_total = 0
    for coords in coordinates:
        if coords[1] < coords[0]:
            # Only consider hops in one direction
            continue
        total_hops = carrier_history[coords] + carrier_history[coords[::-1]]
        net_hops = np.abs(
            carrier_history[coords] - carrier_history[coords[::-1]]
        )
        frequency = total_hops - net_hops
        if frequency == 0:
            net_equals_total += 1
            net_near_total += 1
        else:
            if frequency < 10:
                net_near_total += 1
            frequencies.append(np.log10(frequency))
    plt.figure()
    plt.hist(frequencies, bins=60, color="b")
    plt.xlabel("Discrepancy (Arb. U.)")
    ax = plt.gca()
    tick_labels = np.arange(0, np.ceil(np.max(frequencies)) + 1, 1)
    plt.xlim([0, np.ceil(np.max(frequencies))])
    plt.xticks(tick_labels, [rf"10$^{{{x}}}$" for x in tick_labels])
    plt.ylabel("Frequency (Arb. U.)")
    # 28 for hole hop frequency dist, 29 for electron hop frequency dist
    filename = "{:02}_hop_discrepancy_{}.png".format(
            28 + carrier_types.index(carrier_type),
            carrier_type
            )
    filepath = os.path.join(path, "figures", filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(
        "There are",
        net_equals_total,
        "paths in this morphology with one-way transport.",
    )
    print(
        "There are",
        net_near_total,
        "paths in this morphology with total - net < 10.",
    )
    print(f"Figure saved as {filepath}")


def calculate_mobility(
    path,
    current_carrier_type,
    times,
    MSDs,
    time_stderr,
    MSD_stderr,
):
    # Create the first figure that will be replotted each time
    plt.figure()
    times, MSDs = hf.parallel_sort(times, MSDs)
    mobility, mob_error, r_squared = plot_MSD(
        times,
        MSDs,
        time_stderr,
        MSD_stderr,
        path,
        current_carrier_type,
    )
    print("----------------------------------------")
    print(
        current_carrier_type.capitalize(),
        "mobility for",
        path,
        "= {0:.2E} +- {1:.2E}".format(mobility, mob_error),
        "cm^{2} V^{-1} s^{-1}",
    )
    print("----------------------------------------")
    plt.close()
    return mobility, mob_error, r_squared


def main(
        AA_morphdict,
        CG_morphdict,
        CGtoAAID_list,
        param_dict,
        chromo_list,
        carrier_data_list,
        path,
        three_D=False,
        freq_cut_donor=None,
        freq_cut_acceptor=None,
        sep_cut_donor=None,
        sep_cut_acceptor=None,
        o_cut_donor=None,
        o_cut_acceptor=None,
        ti_cut_donor=None,
        ti_cut_acceptor=None,
        generate_tcl=False,
        sequence_donor=None,
        sequence_acceptor=None,
        xlabel="Temperature (Arb. U.)",
        backend=None,
        ):
    # Load the matplotlib backend and the plotting subroutines
    global plt
    global p3
    if backend is not None:
        import matplotlib

        matplotlib.use(backend.strip())
    import matplotlib.pyplot as plt

    try:
        import mpl_toolkits.mplot3d as p3
    except ImportError:
        print(
            "Could not import 3D plotting engine, calling the plotMolecule3D ",
            "function will result in an error!"
        )
    hole_mobility_data = []
    hole_anisotropy_data = []
    electron_mobility_data = []
    electron_anisotropy_data = []
    # Create the figures path if it doesn't already exist
    os.makedirs(os.path.join(path, "figures"), exist_ok=True)
    # Load in all the required data
    data_dict = {}
    print("\n")
    print("---------- KMC_ANALYSE ----------")
    print(path)
    print("---------------------------------")
    for carrier_data in carrier_data_list:
        # Now need to split up the carrierData into both electrons and holes
        (carrier_data_holes,
         carrier_data_electrons) = split_carriers_by_type(carrier_data)
        if "unwrapped_position" not in AA_morphdict.keys():
            AA_morphdict = hf.add_unwrapped_positions(AA_morphdict)
        morphology_shape = np.array(
            [AA_morphdict[axis] for axis in ["lx", "ly", "lz"]]
        )
        sim_dims = [
            [-AA_morphdict[axis] / 2.0, AA_morphdict[axis] / 2.0]
            for axis in ["lx", "ly", "lz"]
        ]

        # Calculate the mobilities
        complete_carrier_types = []
        complete_carrier_data = []
        if (carrier_data_holes is not None) and (
            len(carrier_data_holes["ID"]) > 0
        ):
            complete_carrier_types.append("hole")
            complete_carrier_data.append(carrier_data_holes)
        if (carrier_data_electrons is not None) and (
            len(carrier_data_electrons["ID"]) > 0
        ):
            complete_carrier_types.append("electron")
            complete_carrier_data.append(carrier_data_electrons)
        carrier_history_dict = {}
        for carrier_type_i, carrier_data in enumerate(complete_carrier_data):
            current_carrier_type = complete_carrier_types[carrier_type_i]
            print(f"Considering the transport of {current_carrier_type}...")
            print("Obtaining mean squared displacements...")
            (carrier_history,
             times,
             MSDs,
             time_stderr,
             MSD_stderr) = get_carrier_data(carrier_data)
            carrier_history_dict[current_carrier_type] = carrier_history
            print("Plotting distribution of carrier displacements")
            plot_displacement_dist(carrier_data, path, current_carrier_type)
            print("Calculating mobility...")
            mobility, mob_error, r_squared = calculate_mobility(
                path,
                current_carrier_type,
                times,
                MSDs,
                time_stderr,
                MSD_stderr,
            )
            print("Plotting hop vector distribution")
            plot_hop_vectors(
                carrier_data,
                AA_morphdict,
                chromo_list,
                path,
                sim_dims,
                current_carrier_type,
                three_D,
            )
            print("Calculating carrier trajectory anisotropy...")
            anisotropy = plot_anisotropy(
                carrier_data,
                path,
                sim_dims,
                current_carrier_type,
                three_D
            )
            print("Plotting carrier hop frequency distribution...")
            if current_carrier_type == "hole":
                freq_cut_donor = plot_frequency_dist(
                    path,
                    current_carrier_type,
                    carrier_history,
                    freq_cut_donor,
                )
            else:
                freq_cut_acceptor = plot_frequency_dist(
                    path,
                    current_carrier_type,
                    carrier_history,
                    freq_cut_acceptor,
                )
            print("Plotting carrier net hop frequency distribution...")
            plot_net_frequency_dist(path, current_carrier_type, carrier_history)
            print("Plotting (total - net hops) discrepancy distribution...")
            plot_discrepancy_frequency_dist(
                path, current_carrier_type, carrier_history
            )
            if (carrier_history is not None) and three_D:
                print(
                    "Determining carrier hopping connections (network graph)..."
                )
                plot_connections(
                    chromo_list,
                    sim_dims,
                    carrier_history,
                    path,
                    current_carrier_type,
                )
            if current_carrier_type == "hole":
                hole_anisotropy_data.append(anisotropy)
                hole_mobility_data.append([mobility, mob_error])
            elif current_carrier_type == "electron":
                electron_anisotropy_data.append(anisotropy)
                electron_mobility_data.append([mobility, mob_error])
            lower_carrier_type = current_carrier_type.lower()
            data_dict["name"] = os.path.split(path)[1]
            data_dict[f"{lower_carrier_type}_anisotropy"] = anisotropy
            data_dict[f"{lower_carrier_type}_mobility"] = mobility
            data_dict[f"{lower_carrier_type}_mobility_r_squared"] = r_squared
        # Now plot the distributions!
        temp_dir = os.path.join(path, "figures")
        chromo_to_mol_ID = determine_molecule_IDs(
            CGtoAAID_list,
            AA_morphdict,
            param_dict,
            chromo_list
        )
        data_dict = plot_energy_levels(temp_dir, chromo_list, data_dict)
        orientations = get_orientations(
            chromo_list,
            CG_morphdict,
            AA_morphdict,
            CGtoAAID_list,
            param_dict,
        )
        sep_cut_donor, sep_cut_acceptor = plot_neighbor_hist(
            chromo_list,
            chromo_to_mol_ID,
            morphology_shape,
            temp_dir,
            sep_cut_donor,
            sep_cut_acceptor,
        )
        o_cut_donor, o_cut_acceptor = plot_orientation_hist(
            chromo_list,
            chromo_to_mol_ID,
            orientations,
            temp_dir,
            o_cut_donor,
            o_cut_acceptor,
        )
        ti_cut_donor, ti_cut_acceptor = plot_TI_hist(
            chromo_list,
            chromo_to_mol_ID,
            temp_dir,
            ti_cut_donor,
            ti_cut_acceptor,
        )
        cut_off_dict = create_cut_off_dict(
            sep_cut_donor,
            sep_cut_acceptor,
            o_cut_donor,
            o_cut_acceptor,
            ti_cut_donor,
            ti_cut_acceptor,
            freq_cut_donor,
            freq_cut_acceptor,
        )
        print("Cut-offs specified (value format: [donor, acceptor])")
        print(*[f"\t{i}" for i in cut_off_dict.items()], sep="\n")
        (cluster_dicts,
         cluster_freqs,
         clusters_total,
         clusters_large,
         clusters_biggest,
         clusters_cutoffs) = get_clusters(
             chromo_list,
             carrier_history_dict,
             orientations,
             cut_off_dict,
             AA_morphdict
             )
        if clusters_total[0] > 0:
            data_dict["donor_clusters_total"] = clusters_total[0]
            data_dict["donor_clusters_large"] = clusters_large[0]
            data_dict["donor_clusters_biggest"] = clusters_biggest[0]
            data_dict["donor_clusters_separation_cut"] = repr(
                clusters_cutoffs[0][0]
            )
            data_dict["donor_clusters_orientation_cut"] = repr(
                clusters_cutoffs[0][1]
            )
            data_dict["donor_clusters_transfer_integral_cut"] = repr(
                clusters_cutoffs[0][2]
            )
            data_dict["donor_clusters_hop_freq_cut"] = repr(
                clusters_cutoffs[0][3]
            )
        if clusters_total[1] > 0:
            data_dict["acceptor_clusters_total"] = clusters_total[1]
            data_dict["acceptor_clusters_large"] = clusters_large[1]
            data_dict["acceptor_clusters_biggest"] = clusters_biggest[1]
            data_dict["acceptor_clusters_separation_cut"] = repr(
                clusters_cutoffs[1][0]
            )
            data_dict["acceptor_clusters_orientation_cut"] = repr(
                clusters_cutoffs[1][1]
            )
            data_dict["acceptor_clusters_transfer_integral_cut"] = repr(
                clusters_cutoffs[1][2]
            )
            data_dict["acceptor_clusters_hop_freq_cut"] = repr(
                    clusters_cutoffs[1][3]
                    )
        if three_D:
            print("Plotting 3D cluster location plot...")
            plot_clusters_3D(
                temp_dir,
                chromo_list,
                cluster_dicts,
                sim_dims,
                generate_tcl
            )
        data_dict = plot_mixed_hopping_rates(
            temp_dir,
            chromo_list,
            param_dict,
            cluster_dicts,
            chromo_to_mol_ID,
            data_dict,
            AA_morphdict,
            cut_off_dict,
        )
        print("Plotting cluster size distribution...")
        plot_cluster_size_dist(cluster_freqs, path)
        print("Writing CSV Output File...")
        write_CSV(data_dict, path)
    print("Plotting Mobility and Anisotropy progressions...")
    if sequence_donor is not None:
        if len(hole_anisotropy_data) > 0:
            plot_temperature_progression(
                sequence_donor,
                hole_mobility_data,
                hole_anisotropy_data,
                "hole",
                xlabel,
            )
    if sequence_acceptor is not None:
        if len(electron_anisotropy_data) > 0:
            plot_temperature_progression(
                sequence_acceptor,
                electron_mobility_data,
                electron_anisotropy_data,
                "electron",
                xlabel,
            )
    else:
        print("Skipping plotting mobility evolution.")
