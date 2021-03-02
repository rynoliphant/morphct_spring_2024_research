from collections import defaultdict
import copy
import csv
import glob
import itertools
import os
import pickle
import shutil
import sys

import freud
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.sparse import lil_matrix
from scipy.stats import linregress

from morphct import helper_functions as hf


plt = None
p3 = None
temperature = 290  # K


def split_carriers(combined_data):
    hole_inds = np.where(np.array(combined_data["c_type"]) == "hole")[0]
    elec_inds = np.where(np.array(combined_data["c_type"]) == "electron")[0]
    hole_data = {}
    elec_data = {}
    for key, val in combined_data.items():
        hole_data[key] = [val[i] for i in hole_inds]
        elec_data[key] = [val[i] for i in elec_inds]
    return hole_data, elec_data


def get_times_msds(carrier_data):
    total = 0
    total_averaged = 0
    squared_disps = defaultdict(list)
    actual_times = defaultdict(list)
    for i, displacement in enumerate(carrier_data["displacement"]):
        if (
            carrier_data["current_time"][i] > carrier_data["lifetime"][i] * 2
            or
            carrier_data["current_time"][i] < carrier_data["lifetime"] / 2
            or
            carrier_data["n_hops"][i] == 1
            ):
            total += 1
            continue
        key = carrier_data["lifetime"][i]
        # A -> m
        squared_disps[key] += [(carrier_data["displacement"][i] * 1e-10) ** 2]
        actual_times[key] += [carrier_data["current_time"][i]]

        # Also keep track of whether each carrier is a hole or an electron
        total_averaged += 1
        total += 1
    if total > total_averaged:
        print(
        f"Notice: The data from {total - total_averaged} carriers were ",
        "discarded due to the carrier lifetime being more than double (or ",
        "less than half of) the specified carrier lifetime."
        )
    times = []
    msds = []
    time_stderr = []
    msd_stderr = []
    for lt, disps in squared_disps.items():
        times.append(lt)
        time_stderr.append(np.std(actual_times[lt]) / len(actual_times[lt]))
        msds.append(np.average(disps))
        msd_stderr.append(np.std(disps) / len(disps))
    return times, msds, time_stderr, msd_stderr


def plot_displacement_dist(carrier_data, c_type, path):
    plt.figure()
    plt.hist(np.array(carrier_data["displacement"]) * 0.1, bins=60, color="b")
    plt.xlabel(f"{c_type.capitalize()} Displacement (nm)")
    plt.ylabel("Frequency (Arb. U.)")
    filename = f"{c_type}_displacement_dist.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300)
    print(f"Figure saved as {filepath}")
    plt.close()


def plot_cluster_size_dist(clusters, path):
    species = ["donor", "acceptor"]
    for i, cl in enumerate(clusters):
        if cl is not None:
            sizes = [np.log10(len(c)) for c in cl.cluster_keys if len(c) > 5]
        if sizes:
            plt.figure()
            plt.hist(
                sizes,
                bins=np.logspace(0, np.ceil(np.max(sizes)), 20),
                color="b",
            )
            plt.xscale("log")
            plt.xlim([1, 10 ** np.ceil(np.max(sizes))])
            plt.xlabel("Cluster Size (Arb. U.)")
            plt.ylabel("Frequency (Arb. U.)")
            filename = f"{species[i]}_cluster_dist.png"
            filepath = os.path.join(path, filename)
            plt.savefig(filepath, dpi=300)
            print(f"Figure saved as {filepath}")
            plt.close()


def get_connections(chromo_list, carrier_history, box):
    """
    Function to create an array of with a starting point, a vector
    and the number of hops that occured.
    Requires:
        chromo_list,
        carrier_history
    Returns:
        7xN array
    """
    # Create an "empty" array to store data.
    connections = np.zeros(7)
    # Iterate through the chromo_list
    for i, chromo in enumerate(chromo_list):
        # Iterate through the neighbors of the chromophore
        for j, image in chromo.neighbors:
            # Only consider one direction.
            if i < j:
                chromoj = chromo_list[j]
                # Get the vector between the two chromophores.
                if np.array_equal(image, [0,0,0]):
                    vector = chromoj.center - chromo.center
                # Account for pbc if not in same relative image.
                else:
                    vector = chromoj.center - chromo.center + image * box

                # Get the net number of times the path was travelled.
                forward = carrier_history[j, i]
                reverse = carrier_history[i, j]
                times = abs(forward - reverse)

                # Append the array if the net times travelled is greater than 0
                if times > 0:
                    datum = np.hstack(
                        (chromo.center, vector, np.array([np.log10(times)]))
                    )
                    connections = np.vstack((connections, datum))
    # Return the array excluding the zeros first line.
    return connections[1:]


def plot_connections(chromo_list, carrier_history, c_type, path):
    # A complicated function that shows connections between carriers in 3D
    # that carriers prefer to hop between.
    # Connections that are frequently used are highlighted in black, whereas
    # rarely used connections are more white.
    # Import matplotlib color modules to set up color bar.
    from matplotlib import colors
    import matplotlib.cm as cmx

    # Create a figure class
    fig = plt.figure(figsize=(7, 6))
    # Make a 3D subplot
    ax = fig.add_subplot(111, projection="3d")

    # Create the array for all the chromophore connections
    connections = get_connections(chromo_list, carrier_history, box)

    # Determine the smallest, non-zero number of times two chromophores
    # are connected.
    vmin = np.min(np.array(connections)[:, 6])
    # Determine the max number of times two chormophores are connected.
    vmax = np.max(np.array(connections)[:, 6])

    # Set up the color bar.
    color_map = plt.get_cmap("inferno")
    c_norm = colors.Normalize(vmin=np.floor(vmin), vmax=np.ceil(vmax))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=color_map)
    hop_colors = scalar_map.to_rgba(connections[:, 6])

    # Set up the intensity for the hops so more travelled paths are more intense
    alphas = connections[:, 6] / vmax
    hop_colors[:, 3] = alphas

    # Plot the vectors between two chromophores
    ax.quiver(
        connections[:, 0],
        connections[:, 1],
        connections[:, 2],
        connections[:, 3],
        connections[:, 4],
        connections[:, 5],
        color=hop_colors,
        arrow_length_ratio=0,
        linewidth=0.7,
    )

    # Draw boxlines
    box_pts = hf.box_points(box)
    ax.plot(box_pts[:,0], box_pts[:,1], box_pts[:,2], c="k", linewidth=1)

    # Make the color bar
    scalar_map.set_array(connections[:, 6])
    tick_location = np.arange(0, vmax, 1)
    # Plot the color bar
    cbar = plt.colorbar(scalar_map, ticks=tick_location, shrink=0.8, aspect=20)
    cbar.ax.set_yticklabels([rf"10$^{{{x}}}$" for x in tick_location])

    # Name and save the figure.
    if c_type == "hole":
        species = "Donor"
    else:
        species = "Acceptor"
    fig.title = f"{species} ({c_type.capitalize()}) Network"

    filename = f"3d_{c_type}_network.png"
    filepath = os.path.join(path, filename),
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    print(f"Figure saved as {filepath}")

    plt.clf()


def calc_mobility(lin_fit_X, lin_fit_Y, time_err, msd_err, temp):
    # YVals have a std error avmsdError associated with them
    # XVals have a std error avTimeError assosciated with them
    numerator = lin_fit_Y[-1] - lin_fit_Y[0]
    denominator = lin_fit_X[-1] - lin_fit_X[0]

    # Diffusion coeff D = d(MSD)/dt * 1/2n (n = 3 = number of dimensions)
    # Ref: Carbone2014a (Carbone and Troisi)
    diff_coeff = numerator / (6 * denominator)

    # The error in the mobility is proportionally the same as the error in the
    # diffusion coefficient as the other variables are constants without error
    diff_err = diff_coeff * np.sqrt(
            (msd_err / numerator) ** 2 + (time_err / denominator) ** 2
    )

    # Use Einstein-Smoluchowski relation
    # This is in m^{2} / Vs
    mobility = (hf.elem_chrg * diff_coeff / (hf.k_B * temp))

    # Convert to cm^{2}/ Vs
    mobility *= 100 ** 2
    mob_err = (diff_err / diff_coeff) * mobility
    return mobility, mob_err


def plot_msd(times, msds, time_stderr, msd_stderr, c_type, path):
    fit_X = np.linspace(np.min(times), np.max(times), 100)
    gradient, intercept, r_val, p_val, std_err = linregress(times, msds)
    print(f"Standard Error {std_err}")
    print(f"Fitting r_val = {r_val}")
    fit_Y = (fit_X * gradient) + intercept
    mobility, mob_error = calc_mobility(
        fit_X,
        fit_Y,
        np.average(time_stderr),
        np.average(msd_stderr),
    )
    plt.plot(times, msds)
    plt.errorbar(times, msds, xerr=time_stderr, yerr=msd_stderr)
    plt.plot(fit_X, fit_Y, "r")
    plt.xlabel("Time (s)")
    plt.ylabel(r"MSD (m$^{2}$)")
    plt.title(rf"$\mu_{{0, {c_type[0]}}}$ = {mobility:.3e} cm$^{2}$/Vs", y=1.1)
    filename = f"lin_MSD_{c_type}.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300)
    plt.clf()
    print(f"Figure saved as {filepath}")

    plt.semilogx(times, msds)
    plt.errorbar(times, msds, xerr=time_stderr, yerr=msd_stderr)
    plt.semilogx(fit_X, fit_Y, "r")
    plt.xlabel("Time (s)")
    plt.ylabel(r"MSD (m$^{2}$)")
    plt.title(rf"$\mu_{{0, {c_type[0]}}}$ = {mobility:.3e} cm$^{2}$/Vs", y=1.1)
    filename = f"semi_log_MSD_{c_type}.png"
    filepath = os.path.join(path, "figures", filename)
    plt.savefig(filepath, dpi=300)
    plt.clf()
    print(f"Figure saved as {filepath}")

    plt.plot(times, msds)
    plt.errorbar(times, msds, xerr=time_stderr, yerr=msd_stderr)
    plt.plot(fit_X, fit_Y, "r")
    plt.xlabel("Time (s)")
    plt.ylabel(r"MSD (m$^{2}$)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(rf"$\mu_{{0,{c_type[0]}}}$ = {mobility:.3e} cm$^{{2}}$/Vs", y=1.1)
    filename = f"log_MSD_{c_type}.png"
    filepath = os.path.join(path, "figures", filename)
    plt.savefig(filepath, dpi=300)
    plt.clf()
    print(f"Figure saved as {filepath}")
    return mobility, mob_error, r_val ** 2


#TODO
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


def plot_hop_vectors(carrier_data, chromo_list, snap, c_type, path,):
    from matplotlib import colors
    import matplotlib.cm as cmx

    box = snap.configuration.box[:3]
    if c_type == "hole":
        carrier_history = carrier_data["hole_history"]
    else:
        carrier_history = carrier_data["electron_history"]

    nonzero = list(zip(*carrier_history.nonzero()))
    hop_vectors = []
    intensities = []
    for i,j in nonzero:
        # Find the normal of chromo1 plane (first three atoms)
        ichromo_i = chromo_list[i]
        jchromo_j = chromo_list[j]
        normal = hf.get_chromo_normvec(ichromo, snap)

        # Calculate rotation matrix required to map this onto (0, 0, 1)
        rotation_matrix = hf.get_rotation_matrix(normal, np.array([0, 0, 1]))

        # Find the vector from chromoi to chromoj (using correct relative image)
        relative_image = [img for ind,img in ichromo.neighbors if ind == j][0]
        jchromo_center = jchromo.center + relative_image * box

        unrotated_hop_vector = jchromo_center - ichromo_center

        # Apply rotation matrix to chromo1-chromo2 vector
        hop_vector = np.matmul(rotation_matrix, unrotated_hop_vector)

        # Store the hop vector and the intensity
        intensity = np.log10(carrier_history[i, j])
        hop_vectors.append(hop_vector)
        intensities.append(intensity)

    # Convert hop_vectors to an np.array
    hop_vectors = np.array(hop_vectors)
    # Create a figure class
    fig = plt.figure(figsize=(7, 6))
    # Make a 3D subplot
    ax = fig.add_subplot(111, projection="3d")
    # Determine the smallest, non-zero n_times two chromophores are connected.
    I_min = np.min(intensities)
    # Determine the max number of times two chormophores are connected.
    I_max = np.max(intensities)

    # Set up the color bar.
    color_map = plt.get_cmap("Greys")
    c_norm = colors.Normalize(vmin=np.floor(I_min), vmax=np.ceil(I_max))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=color_map)
    hop_colors = scalar_map.to_rgba(intensities)

    n = len(hop_colors)
    # Plot the vectors between two chromophores
    # X-vals for start of each arrow
    # Y-vals for start of each arrow
    # Z-vals for start of each arrow
    # X-vecs for direction of each arrow
    # Y-vecs for direction of each arrow
    # Z-vecs for direction of each arrow
    ax.quiver(
        np.zeros(n), np.zeros(n), np.zeros(n),
        hop_vectors[:, 0], hop_vectors[:, 1], hop_vectors[:, 2],
        color=hop_colors, arrow_length_ratio=0, linewidth=0.7,
    )
    ax.set_xlim([-np.max(hop_vectors[:, 0]), np.max(hop_vectors[:, 0])])
    ax.set_ylim([-np.max(hop_vectors[:, 1]), np.max(hop_vectors[:, 1])])
    ax.set_zlim([-np.max(hop_vectors[:, 2]), np.max(hop_vectors[:, 2])])

    # Make the color bar
    scalar_map.set_array(intensities)
    # tick_location = np.arange(0, int(np.ceil(vmax)) + 1, 1)
    tick_location = np.arange(0, I_max, 1)
    # Plot the color bar
    cbar = plt.colorbar(scalar_map, ticks=tick_location, shrink=0.8, aspect=20)
    cbar.ax.set_yticklabels([rf"10$^{{{x}}}$" for x in tick_location])

    # Name and save the figure.
    if c_type == "hole":
        species = "donor"
    else:
        species = "acceptor"
    figure_title = f"{species.capitalize()} ({c_type.capitalize()}) Network"

    filename = f"hop_vec_{c_type}.png"
    filepath = os.path.join(path, filename),
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    print(f"Figure saved as {filepath}")
    plt.clf()


def plot_anisotropy(carrier_data, c_type, path):
    box = carrier_data["box"][0]
    xyz_vals = []
    # Get the indices of the carriers that travelled the furthest
    # only do the first 1000, in case there's a lot
    for i,pos in enumerate(carrier_data["current_position"][:1000]):
        image = combined_data["image"][i]
        position = image * box + pos
        xyz_vals.append(position / 10.0)

    colors = ['b'] * len(xyz_vals)
    xyz_vals = np.array(xyz_vals)

    anisotropy = get_anisotropy(xyz_vals[:,0], xyz_vals[:,1], xyz_vals[:,2])
    print(
            "----------------------------------------\n",
            f"{c_type.capitalize()} charge transport anisotropy ",
            f"calculated as {anisotrophy}\n",
            "----------------------------------------"
            )
    # Reduce number of plot markers
    fig = plt.gcf()
    ax = p3.Axes3D(fig)
    plt.scatter(xyz_vals[:,0], xyz_vals[:,1], zs=xyz_vals[:,2], c=colors, s=20)
    plt.scatter(0, 0, zs=0, c="r", s=50)

    # Draw boxlines
    box_pts = hf.box_points(box)
    ax.plot(box_pts[:,0], box_pts[:,1], box_pts[:,2], c="k", linewidth=1)

    ax.set_xlabel("X (nm)", fontsize=20, labelpad=40)
    ax.set_ylabel("Y (nm)", fontsize=20, labelpad=40)
    ax.set_zlabel("Z (nm)", fontsize=20, labelpad=40)
    maximum = np.max(xyz_vals)
    ax.set_xlim([-maximum, maximum])
    ax.set_ylim([-maximum, maximum])
    ax.set_zlim([-maximum, maximum])
    ticks = [
        getattr(ax, f"{xyz}axis").get_major_ticks() for xyz in ["x","y","z"]
            ]
    ticks = [i for l in ticks for i in l]
    for tick in ticks:
        tick.label.set_fontsize(16)
    ax.dist = 11
    plt.title(f"Anisotropy ({c_type.capitalize()})", y=1.1)
    filename = f"anisotropy_{c_type}.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.clf()
    print(f"Figure saved as {filepath}")
    return anisotropy


def get_temp_val(string):
    return string.split("-")[-2][2:]


def get_frame_val(string):
    return string.split("-")[0][1:]


def plot_temp_progression(temp, mobility, mob_err, anisotropy, c_type, path):
    plt.gcf()
    plt.clf()
    xvals = temp
    yvals = mobility
    yerrs = mob_err
    plt.xlabel("Temperature (Arb. U)")
    plt.ylabel(r"Mobility (cm$^{2}$ / Vs)")
    plt.errorbar(xvals, yvals, xerr=0, yerr=yerrs)
    plt.yscale("log")
    filename = f"mobility_{c_type}.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300)
    plt.clf()
    print(f"Figure saved as {filepath}")

    plt.plot(temp, anisotropy, c="r")
    plt.xlabel("Temperature (Arb. U)")
    plt.ylabel(r"$\kappa$ (Arb. U)")
    filename = f"anisotropy_{c_type}.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300)
    plt.clf()
    print(f"Figure saved as {filepath}")


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


def plot_neighbor_hist(
    chromo_list, chromo_mol_id, box, d_sepcut, a_sepcut, path,
):
    seps_donor = []
    seps_acceptor = []
    for i, ichromo in enumerate(chromo_list):
        for j, img in ichromo.neighbors:
            jchromo = chromo_list[j]
            # Skip any chromophores that are part of the same molecule
            if chromo_mol_id[i] == chromo_mol_id[j]:
                continue
            sep = np.linalg.norm(jchromo.center + img * box - ichromo.center)
            if ichromo.species == "donor":
                seps_donor.append(sep)
            else:
                seps_acceptor.append(sep)
    species = ["donor", "acceptor"]
    seps = [seps_donor, seps_acceptor]
    for sp, sep in zip(species, seps):
        if len(sep) == 0:
            continue

        if sp == "donor":
            sep_cut = d_sepcut
        else:
            sep_cut = a_sepcut

        plt.figure()
        n, bin_edges, _ = plt.hist(sep, bins=40, color="b")
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        bin_centres = np.insert(bin_centres, 0, 0)
        n = np.insert(n, 0, 0)
        smoothed_n = gaussian_filter(n, 1.0)
        plt.plot(bin_centres, smoothed_n, color="r")
        if sep_cut is None:
            sep_cut = calculate_cutoff_from_dist(
                bin_centres,
                smoothed_n,
                minimum_i=0,
                value_at_least=100,
                logarithmic=False,
            )
        print(
            f"Cluster cut-off based on {sp} chromophore separation set ",
            f"to {sep_cut}"
            )
        plt.axvline(sep_cut, c="k")
        plt.xlabel(rf"{sp.capitalize()} r$_{{i,j}}$ (\AA)")
        plt.ylabel("Frequency (Arb. U.)")
        filename = f"neighbor_hist_{sp}.png"
        filepath = os.path.join(path, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Neighbour histogram figure saved as {filepath}")
        plt.close()

        if sp == "donor":
            d_sepcut = sep_cut
        else:
            a_sepcut = sep_cut
    return d_sepcut, a_sepcut


def plot_orientation_hist(
    chromo_list, chromo_mol_id, orientations, d_ocut, a_ocut, path,
):
    orientations_donor = []
    orientations_acceptor = []
    for i, ichromo in enumerate(chromo_list):
        ivec = orientations[i]
        for j, img in ichromo.neighbors:
            jchromo = chromo_list[j]
            # Skip any chromophores that are part of the same molecule
            if chromo_mol_id[i] == chromo_mol_id[j]:
                continue
            jvec = orientations[j]
            dot_product = np.dot(ivec, jvec)
            # in radians
            angle = np.arccos(np.abs(dot_product))
            if ichromo.species == "donor":
                orientations_donor.append(angle)
            elif ichromo.species == "acceptor":
                orientations_acceptor.append(angle)
    species = ["donor", "acceptor"]
    orients = [orientations_donor, orientations_acceptor]
    for sp, orient in zip(species, orients):
        if len(orient) == 0:
            continue
        if sp == "donor":
            ocut = d_ocut
        else:
            ocut = a_ocut,
        plt.figure()
        n, bin_edges, _ = plt.hist(orient, bins=40, color="b")
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        bin_centres = np.insert(bin_centres, 0, 0)
        n = np.insert(n, 0, 0)
        smoothed_n = gaussian_filter(n, 1.0)
        plt.plot(bin_centres, smoothed_n, color="r")
        if ocut is None:
            ocut = calculate_cutoff_from_dist(
                bin_centres,
                smoothed_n,
                maximum_i=0,
                value_at_least=100,
                logarithmic=False,
            )
        print(
            "Cluster cut-off based on {sp} relative chromophore orientations",
            " set to {ocut}"
            )
        plt.axvline(ocuts, c="k")
        plt.xlabel(f"{sp.capitalize()} Orientations (rad)")
        plt.xlim([0, np.deg2rad(90)])
        plt.xticks(np.arange(0, np.deg2rad(91), np.deg2rad(15)))
        plt.ylabel("Frequency (Arb. U.)")
        filename = f"orientation_hist_{sp}.png"
        filepath = os.path.join(path, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Orientation histogram figure saved as {filepath}")
        if sp == "donor":
            d_ocut = ocut
        else:
            a_ocut = ocut
    return d_ocut, a_ocut


def create_cutoff_dict(
    d_sepcut, a_sepcut, d_ocut, a_ocut, d_ticut, a_ticut, d_freqcut, a_freqcut,
):
    cutoff_dict = {
        "separation": [d_sepcut, a_sepcut],
        "orientation": [d_ocut, a_ocut],
        "ti": [d_ticut, a_ticut],
        "freq": [d_freqcut, a_freqcut],
    }
    return cutoff_dict


def get_clusters(chromo_list, snap, rmax=None):
    box = snap.configuration.box
    species = ["donor", "acceptor"]
    for sp_i, sp in enumerate(species):
        chromo_ids = [c.id for c in chromo_list if c.species == sp]
        positions = np.array([chromo_list[i].center for i in chromo_ids])
        print(f"Examining the {sp} material...")
        if len(positions) == 0:
            print("No material found. Continuing...")
            clusters.append(None)
            continue
        print("Calculating clusters...")
        if rmax is None:
            rmax = max(box)/4
            print(f"No cutoff provided: cluster cutoff set to {rmax}")

        cl = freud.cluster.Cluster()

        cl.compute((box,positions), neighbors={'r_max': rmax})

        large = sum([1 for c in cl.cluster_keys if len(c) > 6])
        biggest = max([len(c) for c in cl.cluster_keys])
        psi = large / cl.num_clusters
        print("----------------------------------------")
        print(f"{sp}: Detected {cl.num_clusters} total")
        print(f"and {large} large clusters (size > 6).")
        print(f"Largest cluster size: {biggest} chromophores.")
        print(f'Ratio in "large" clusters: {psi:.3f}')
        print("----------------------------------------")
    return clusters


def get_orientations(chromo_list, snap):
    orientations = []
    for chromo in chromo_list:
        positions = snap.particles.position[chromo.atom_ids]

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


def get_separations(chromo_list, snap):
    box = snap.configuration.box[:3]
    n = len(chromo_list)
    separations = lil_matrix((n, n), dtype=float)
    for i, ichromo in enumerate(chromo_list):
        for j, img in chromo.neighbors:
            if i > j:
                continue
            jcenter = chromo_list[j].center
            jpos = jcenter + img * box
            separations[i, j] = np.linalg.norm(jpos, ichromo.center)
    return separations


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


def cluster_tcl_script(clusters, large_cluster, path):
    """
    Create a tcl script for each identified cluster.
    """
    # Obtain the IDs of the cluster sizes, sorted by largest first
    print("Sorting the clusters by size...")
    species = ["donor", "acceptor"]
    for i,cl in enumerate(clusters):
        sp = species[i]

        if cl is None:
            print(f"No clusters found for {sp}, continuing...")

        sorted_cl = sorted(cl.cluster_keys, key=lambda i: len(i), reverse=True)
        colors = list(range(int(1e6)))
        count = 0

        print("Creating tcl header...")
        tcl_text = ["mol delrep 0 0;"]
        tcl_text += ["pbc wrap -center origin;"]
        tcl_text += ["pbc box -color black -center origin -width 4;"]
        tcl_text += ["display resetview;"]
        tcl_text += ["color change rgb 9 1.0 0.29 0.5;"]
        tcl_text += ["color change rgb 16 0.25 0.25 0.25;"]

        if sp == "donor":
            for i, chromo_ids in enumerate(sorted_cl):
                print(f"Creating tcl for cluster {i+1}/{len(sorted_cl)}")
                if len(chromo_ids) > large_cluster:
                    # Only make clusters that are ``large''
                    inclust = ""
                    for chromo in chromo_ids:
                        inclust += f"{inclust}{chromo:d} "
                tcl_text += ["mol material AOEdgy;"]
                # Use AOEdgy if donor
                # The +1 makes the largest cluster red rather than
                # blue (looks better with AO, DoF, shadows)
                tcl_text += [f"mol color ColorID {colors[count + 1 % 32]:d};"]
                # VMD has 32 unique colors
                tcl_text += ["mol representation VDW 4.0 8.0;"]
                tcl_text += [f"mol selection resid {inclust:s};"]
                tcl_text += ["mol addrep 0;"]
                count += 1
        else:
            for i, chromo_ids in enumerate(sorted_cl):
                print(f"Creating tcl for cluster {i+1}/{len(sorted_cl)}")
                if len(chromo_ids) > large_cluster:
                    inclust = ""
                    for chromo in chromo_ids:
                        inclust += f"{inclust}{chromo:d} "
                tcl_text += ["mol material Glass2;"]
                # Use Glass2 if acceptor
                # The +1 makes the largest cluster red rather than blue (looks
                # better with AO, DoF, shadows)
                tcl_text += [f"mol color ColorID {colors[count + 1 % 32]:d};"]
                tcl_text += ["mol representation VDW 4.0 8.0;"]
                tcl_text += [f"mol selection resid {inclust};"]
                tcl_text += ["mol addrep 0;"]
                count += 1
    tcl_file_path = os.path.join(path, "{sp}_cluster_colors.tcl")
    with open(tcl_file_path, "w+") as tcl_file:
        tcl_file.writelines("".join(tcl_text))
    print(f"\nClusters coloring written to {tcl_file_path}")


def get_lists_for_3d_clusters(clusters, chromo_list, colors, large_cluster):
    data = []
    species = ["donor", "acceptor"]
    for i,cl in enumerate(clusters):
        sp = species[i]

        if cl is None:
            print(f"No clusters found for {sp}, continuing...")

        if sp == "donor":
            for clust_id, chromo_ids in enumerate(cl.cluster_keys):
                if len(chromo_ids) > large_cluster:
                    for c_id in chromo_ids:
                        chromo = chromo_list[c_id]
                        data.append(
                                [*chromo.center, "w", colors[clust_id % 7]]
                                )
        else:
            for clust_id, chromo_ids in enumerate(cl.cluster_keys):
                if len(chromo_ids) > large_cluster:
                    for c_id in chromo_ids:
                        chromo = chromo_list[c_id]
                        data.append(
                                [*chromo.center, colors[clust_id % 7], "none"]
                                )

    # Needs a sort so that final image is layered properly
    data = list(sorted(data, key=lambda x: x[0]))
    # Split up list into sublists
    xyzs = np.array([row[:3] for row in data])

    face_colors = [row[3] for row in data]
    edge_colors = [row[4] for row in data]
    return xyzs, face_colors, edge_colors


def plot_clusters_3D(chromo_list, clusters, box, generate_tcl, path):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    large_cluster = 6
    if generate_tcl:
        cluster_tcl_script(clusters, large_cluster, path)

    xyzs, face_colors, edge_colors = get_lists_for_3d_clusters(
        clusters, chromo_list, colors, large_cluster
        )
    ax.scatter(
            xyzs[:,0],
            xyzs[:,1],
            xyzs[:,2],
            facecolors=face_colors,
            edgecolors=edge_colors,
            alpha=0.6,
            s=40,
            )

    # Draw boxlines
    box_pts = hf.box_points(box)
    ax.plot(box_pts[:,0], box_pts[:,1], box_pts[:,2], c="k", linewidth=1)

    ax.set_xlim([-box[0]/2, box[0]/2])
    ax.set_ylim([-box[1]/2, box[1]/2])
    ax.set_zlim([-box[2]/2, box[2]/2])

    filepath = os.path.join(path, "figures", "clusters.png"),
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    print(f"3D cluster figure saved as {filepath}")
    plt.close()


def snap_molecule_indices(snap):
    """Find molecule index for each particle.

    Given a snapshot from a trajectory, compute clusters of bonded molecules
    and return an array of the molecule index of each particle.

    Parameters
    ----------
    snap : gsd.hoomd.Snapshot
        Trajectory snapshot.

    Returns
    -------
    numpy array (N_particles,)
    """
    system = freud.AABBQuery.from_system(snap)
    num_query_points = num_points = snap.bonds.N
    query_point_indices = snap.bonds.group[:, 0]
    point_indices = snap.bonds.group[:, 1]
    distances = system.box.compute_distances(
        system.points[query_point_indices], system.points[point_indices]
    )
    nlist = freud.NeighborList.from_arrays(
        num_query_points, num_points, query_point_indices, point_indices, distances
    )
    cluster = freud.cluster.Cluster()
    cluster.compute(system=system, neighbors=nlist)
    return cluster.cluster_idx


def get_molecule_ids(snap, chromo_list):
    print("Determining molecule IDs...")
    # Determine molecules based on bonds
    molecules = snap_molecule_indices(snap)
    # Convert to chromo_mol_id
    chromo_mol_id = {}
    for i,chromo in enumerate(chromo_list):
        chromo_mol_id[i] = molecules[chromo.atom_ids[0]]
    return chromo_mol_id


def plot_energy_levels(chromo_list, data_dict, path,):
    homo_levels = []
    lumo_levels = []
    donor_delta_eij = []
    acceptor_delta_eij = []
    donor_lambda_ij = None
    acceptor_lambda_ij = None
    for chromo in chromo_list:
        if chromo.species == "donor":
            homo_levels.append(chromo.homo)
            for i, delta_eij in enumerate(chromo.neighbors_delta_e):
                ti = chromo.neighbors_ti[i]
                if delta_eij is not None and ti is not None:
                    donor_delta_eij.append(delta_eij)
                donor_lambda_ij = chromo.reorganization_energy
        else:
            lumo_levels.append(chromo.lumo)
            for i, delta_eij in enumerate(chromo.neighbors_delta_e):
                ti = chromo.neighbors_ti[i]
                if delta_eij is not None and ti is not None:
                    acceptor_delta_eij.append(delta_eij)
                acceptor_lambda_ij = chromo.reorganization_energy
    if donor_delta_eij:
        (donor_bin_edges,
         donor_fit_args,
         donor_mean,
         donor_std) = gauss_fit(donor_delta_eij)
        donor_err = donor_std / np.sqrt(len(donor_delta_eij))

        data_dict["donor_delta_eij_mean"] = donor_mean
        data_dict["donor_delta_eij_std"] = donor_std
        data_dict["donor_delta_eij_err"] = donor_err

        homo_av = np.average(homo_levels)
        homo_std = np.std(homo_levels)
        homo_err = homo_std / np.sqrt(len(homo_levels))
        data_dict["donor_homo_mean"] = homo_av
        data_dict["donor_homo_std"] = homo_std
        data_dict["donor_homo_err"] = homo_err
        print(
            f"Donor HOMO Level = {homo_av:.3f} +/- {homo_err:.3f}\n",
            f"Donor Delta E_ij mean = {donor_mean:.3f} +/- {donor_err:.3f}"
            )
        plot_delta_eij(
            donor_delta_eij,
            donor_bin_edges,
            donor_fit_args,
            "donor",
            donor_lambda_ij,
            path,
        )
    if acceptor_delta_eij:
        (acceptor_bin_edges,
         acceptor_fit_args,
         acceptor_mean,
         acceptor_std) = gauss_fit(acceptor_delta_eij)
        acceptor_err = acceptor_std / np.sqrt(len(acceptor_delta_eij))

        data_dict["acceptor_delta_eij_mean"] = acceptor_mean
        data_dict["acceptor_delta_eij_std"] = acceptor_std
        data_dict["acceptor_delta_eij_err"] = acceptor_err
        lumo_av = np.average(lumo_levels)
        lumo_std = np.std(lumo_levels)
        lumo_err = lumo_std / np.sqrt(len(lumo_levels))
        data_dict["acceptor_lumo_mean"] = LUMO_av
        data_dict["acceptor_lumo_std"] = LUMO_std
        data_dict["acceptor_lumo_err"] = LUMO_err
        print(
            f"Acceptor LUMO Level = {lumo_av} +/- {lumo_err}\n",
            f"Acceptor Delta E_ij mean = {acceptor_mean} +/-{acceptor_err}"
            )
        plot_delta_eij(
            acceptor_delta_eij,
            acceptor_bin_edges,
            acceptor_fit_args,
            "acceptor",
            acceptor_lambda_ij,
            path,
        )


def plot_delta_eij(delta_eij, gauss_bins, fit_args, species, lambda_ij, path):
    plt.figure()
    n, bins, patches = plt.hist(
        delta_eij,
        np.arange(np.min(delta_eij), np.max(delta_eij), 0.05),
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
    plt.xlabel(rf"{species.capitalize()} $\Delta E_{{i,j}}$ (eV)")

    filename = f"{species}_delta_E_ij.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Figure saved as {filepath}")


def plot_mixed_hopping_rates(
    chromo_list,
    clusters,
    chromo_mol_id,
    data_dict,
    cutoff_dict,
    temp,
    path,
    use_vrh=False,
    koopmans=None,
    boltz_pen=False,
):
    # Create all the empty lists we need
    hop_types = ["intra", "inter"]
    hop_targets = ["c", "m"]
    # hop_properties = ["r", "T"]
    species = ["d", "a"]
    prop_lists = defaultdict(list)
    for i, ichromo in enumerate(chromo_list):
        imol = chromo_mol_id[i]
        for ineighbor, ti in enumerate(chromo.neighbors_ti):
            if ti is None or ti == 0:
                continue
            j = ichromo.neighbors[ineighbor][0]
            jchromo = chromo_list[j]
            jmol = chromo_mol_id[j]
            delta_e = ichromo.neighbors_delta_E[ineighbor]
            if ichromo.species == jchromo.species:
                lambda_ij = ichromo.reorganization_energy
            else:
                lambda_ij = (
                    ichromo.reorganization_energy +
                    jchromo.reorganization_energy
                ) / 2
            # Now take into account the various behaviours we can have from
            # the parameter file
            prefactor = 1.0
            # Apply the koopmans prefactor
            if koopmans is not None:
                prefactor *= koopmans
            # Apply the distance penalty due to VRH
            if use_vrh:
                vrh_delocalization = 1.0 / ichromo.vrh_delocalization
                rel_image = ichromo.neighbors[ineighbor][1]
                jchromo_center = jchromo.center + rel_image * box
                rij = np.linalg.norm(ichromo.center - jchromo_center) * 1e-10
                rate = hf.calculate_carrier_hop_rate(
                    lambda_ij,
                    ti,
                    delta_e,
                    prefactor,
                    temp,
                    use_VRH=use_vrh,
                    rij=rij,
                    VRH_delocalisation=vrh_delocalisation,
                    boltz_pen=boltz_pen,
                )
            else:
                rate = hf.calculate_carrier_hop_rate(
                    lambda_ij,
                    ti,
                    delta_e,
                    prefactor,
                    temp,
                    boltz_pen=boltz_pen,
                )
            if j < i:
                continue
            # Do intra- / inter- clusters
            if ichromo.species == "acceptor":
                if clusters[1] is not None:
                    for cluster in clusters[1].cluster_keys:
                        if i in cluster and j in cluster:
                            prop_lists["intra_cra"].append(rate)
                            prop_lists["intra_cTa"].append(T_ij)
                            break
                        else:
                            prop_lists["inter_cra"].append(rate)
                            prop_lists["inter_cTa"].append(T_ij)
            else:
                if clusters[0] is not None:
                    for cluster in clusters[0].cluster_keys:
                        if i in cluster and j in cluster:
                            prop_lists["intra_crd"].append(rate)
                            prop_lists["intra_cTd"].append(T_ij)
                            break
                        else:
                            prop_lists["inter_crd"].append(rate)
                            prop_lists["inter_cTd"].append(T_ij)
            # Now do intra- / inter- molecules
            if imol == jmol:
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
            path,
            )
        plot_stacked_hist_tis(
            prop_lists["intra_cTd"],
            prop_lists["inter_cTd"],
            ["Intra-cluster", "Inter-cluster"],
            "donor",
            cutoff_dict["ti"][0],
            path,
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
            path,
        )
        plot_stacked_hist_tis(
            prop_lists["intra_cTa"],
            prop_lists["inter_cTa"],
            ["Intra-cluster", "Inter-cluster"],
            "acceptor",
            cutoff_dict["ti"][1],
            path
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
            path
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
            path,
        )
    # Update the dataDict
    for hop_type,target,sp in itertools.product(hop_types,hop_targets,species):
        hop_name = f"{hop_type}_{target}r{sp}"
        n_hops = len(prop_lists[hop_name])
        if n_hops == 0:
            continue

        other_hop = hop_types[hop_types.index(hop_type) * -1 + 1]
        other_hop_name = f"{other_hop}_{target}r{sp}"

        total_hops = n_hops + len(prop_lists[other_hop_name])
        proportion = n_hops / total_hops

        mean_rate = np.mean(prop_lists[hop_name])
        stdev_rate = np.std(prop_lists[hop_name])
        data_dict[f"{hop_name}_hops"]= n_hops
        data_dict[f"{hop_name}_proportion"] = proportion
        data_dict[f"{hop_name}_rate_mean"] = mean_rate
        data_dict[f"{hop_name}_rate_std"] = stdev_rate


def plot_stacked_hist_rates(data1, data2, labels, species, path):
    plt.figure()
    n, bins, patches = plt.hist(
        [data1, data2],
        bins=np.logspace(1, 18, 40),
        stacked=True,
        color=["r", "b"],
        label=labels,
    )
    plt.ylabel("Frequency (Arb. U.)")
    plt.xlabel(rf"{species.capitalize()} k$_{{i,j}}$ (s$^{-1}$)")
    plt.xlim([1, 1e18])
    plt.xticks([1e0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18])
    plt.ylim([0, np.max(n) * 1.02])
    plt.legend(loc=2, prop={"size": 18})
    plt.gca().set_xscale("log")
    filename = f"{species}_hopping_rate_clusters.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Figure saved as {filepath}")


def plot_stacked_hist_tis(data1, data2, labels, species, cutoff, path):
    plt.figure()
    n, bins, patches = plt.hist(
        [data1, data2],
        bins=np.linspace(0, 1.2, 20),
        stacked=True,
        color=["r", "b"],
        label=labels,
    )
    plt.ylabel("Frequency (Arb. U.)")
    plt.xlabel(rf"{species.capitalize()} J$_{{i,j}}$ (eV)")
    plt.ylim([0, np.max(n) * 1.02])
    if cutoff is not None:
        plt.axvline(float(cutoff), c="k")
    plt.legend(loc=0, prop={"size": 18})

    filename = f"{species}_transfer_integral_clusters.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Figure saved as {filepath}")


def write_csv(data_dict, path):
    filepath = os.path.join(path, "results.csv")
    with open(filepath, "w") as f:
        w = csv.writer(f)
        for key, val in data_dict.items():
            w.writerow([key, val])
    print(f"CSV file written to {filepath}")


#TODO
def calculate_cutoff_from_dist(
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
                cutoff = 0.0
                while True:
                    selected_minimum = minima[minimum_i]
                    cutoff = bin_centres[selected_minimum]
                    if frequencies[selected_minimum] > value_at_least:
                        break
                    minimum_i -= 1
            else:
                # Sometimes a tiny maximum at super LHS breaks this
                cutoff = 0.0
                while True:
                    selected_minimum = minima[minimum_i]
                    cutoff = bin_centres[selected_minimum]
                    if frequencies[selected_minimum] > value_at_least:
                        break
                    minimum_i += 1
        elif maximum_i is not None:
            # Looking for maxima
            maxima = argrelextrema(frequencies, np.greater)[0]
            if maximum_i < 0:
                # Sometimes a tiny maximum at super RHS breaks this
                cutoff = 0.0
                while True:
                    selected_maximum = maxima[maximum_i]
                    cutoff = bin_centres[selected_maximum]
                    if frequencies[selected_maximum] > value_at_least:
                        break
                    maximum_i -= 1
            else:
                # Sometimes a tiny maximum at super LHS breaks this
                cutoff = 0.0
                while True:
                    selected_maximum = maxima[maximum_i]
                    cutoff = bin_centres[selected_maximum]
                    if frequencies[selected_maximum] > value_at_least:
                        break
                    maximum_i += 1
        if logarithmic is True:
            cutoff = 10 ** cutoff
        # Return as string, as it will be converted to a float later
        return str(cutoff)
    except IndexError:
        print(
            "EXCEPTION: No minima found in frequency distribution.",
            " Setting cutoff to None."
        )
        return None


def plot_ti_hist(
    chromo_list, chromo_mol_id, d_ticut, a_ticut, path,
):
    # ti_dist [[DONOR], [ACCEPTOR]]
    ti_intra = [[], []]
    ti_inter = [[], []]
    species = ["donor", "acceptor"]
    labels = ["Intra-mol", "Inter-mol"]
    ti_cuts = [d_ticut, a_ticut]
    for sp_i, sp in enumerate(species):
        for i, ichromo in enumerate(chromo_list):
            for i_neighbor, (j, img) in enumerate(ichromo.neighbors):
                jchromo = chromo_list[j]
                if i >= j:
                    continue
                if chromo_mol_id[i] == chromo_mol_id[j]:
                    ti_intra[sp_i].append(ichromo.neighbors_ti[i_neighbor])
                else:
                    ti_inter[sp_i].append(ichromo.neighbors_ti[i_neighbor])
        if not ti_intra[sp_i] and ti_inter[sp_i]:
            continue
        plt.figure()
        maxti = np.max(ti_intra[sp_i] + ti_inter[sp_i])
        n, bin_edges, _ = plt.hist(
            [ti_intra[sp_i], ti_inter[sp_i]],
            bins=np.linspace(0, maxti, 20),
            color=["r", "b"],
            stacked=True,
            label=labels,
        )
        bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2
        smoothed_n = gaussian_filter(n[0] + n[1], 1.0)
        plt.plot(bin_centres, smoothed_n, color="r")
        if ti_cuts[sp_i] is None:
            ti_cuts[sp_i] = calculate_cutoff_from_dist(
                bin_centres, smoothed_n, minimum_i=-1, value_at_least=100
            )
        print(
            f"Cluster cut-off based on {sp} ",
            f"transfer integrals set to {ti_cuts[sp_i]}"
            )
        plt.axvline(float(ti_cuts[species_i]), c="k")
        plt.xlim([0, np.max(ti_intra[sp_i] + ti_inter[sp_i])])
        plt.ylim([0, np.max(n) * 1.02])
        plt.ylabel("Frequency (Arb. U.)")
        plt.xlabel(rf"{sp.capitalize()} J$_{{i,j}}$ (eV)")
        plt.legend(loc=1, prop={"size": 18})
        filename = f"{sp}_transfer_integral_mols.png"
        filepath = os.path.join(path, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Figure saved as {filepath}")
    return ti_cuts


def plot_frequency_dist(c_type, carrier_history, cutoff, path):
    nonzero = list(zip(*carrier_history.nonzero()))
    frequencies = []
    for i,j in nonzero:
        if i < j:
            # Only consider hops in one direction
            continue
        frequency = carrier_history[i,j]
        frequencies.append(np.log10(frequency))
    plt.figure()
    n, bin_edges, _ = plt.hist(frequencies, bins=60, color="b")
    bin_centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    smoothed_n = gaussian_filter(n, 1.0)
    plt.plot(bin_centres, smoothed_n, color="r")
    if cutoff is None:
        print("DYNAMIC CUT")
        cutoff = calculate_cutoff_from_dist(
            bin_centres,
            smoothed_n,
            minimum_i=-1,
            value_at_least=100,
            logarithmic=True,
        )
    print(f"Cluster cut-off based on hop frequency set to {cutoff}")
    plt.axvline(np.log10(cutoff), c="k")
    plt.xlabel(f"Total {c_type} hops (Arb. U.)")
    ax = plt.gca()
    tick_labels = np.arange(0, np.ceil(np.max(frequencies)) + 1, 1)
    plt.xlim([0, np.ceil(np.max(frequencies))])
    plt.xticks(tick_labels, [rf"10$^{{{int(x)}}}$" for x in tick_labels])
    plt.ylabel("Frequency (Arb. U.)")
    filename = f"total_hop_freq_{c_type}.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Figure saved as {filepath}")
    return cutoff


def plot_net_frequency_dist(c_type, carrier_history, path):
    nonzero = list(zip(*carrier_history.nonzero()))
    frequencies = []
    for i,j in nonzero:
        if i < j:
            # Only consider hops in one direction
            continue
        frequency = np.abs(carrier_history[i,j] - carrier_history[j,i])
        if frequency > 0:
            frequencies.append(np.log10(frequency))
    plt.figure()
    plt.hist(frequencies, bins=60, color="b")
    plt.xlabel(f"Net {c_type} hops (Arb. U.)")
    ax = plt.gca()
    tick_labels = np.arange(0, np.ceil(np.max(frequencies)) + 1, 1)
    plt.xlim([0, np.ceil(np.max(frequencies))])
    plt.xticks(tick_labels, [rf"10$^{{{x}}}$" for x in tick_labels])
    plt.ylabel("Frequency (Arb. U.)")
    filename = f"net_hop_freq_{c_type}.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Figure saved as {filepath}")


def plot_discrepancy_frequency_dist(c_type, carrier_history, path):
    nonzero = list(zip(*carrier_history.nonzero()))
    frequencies = []
    net_equals_total = 0
    net_near_total = 0
    for i,j in nonzero:
        if i < j:
            # Only consider hops in one direction
            continue
        total_hops = carrier_history[i,j] + carrier_history[j,i]
        net_hops = np.abs(carrier_history[i,j] - carrier_history[j,i])
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
    filename = f"hop_discrepancy_{c_type}.png"
    filepath = os.path.join(path, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(
        f"There are {net_equals_total} paths in this morphology with ",
        f"one-way transport. \nThere are {net_near_total} paths in this ",
        "morphology with total - net < 10."
        )
    print(f"Figure saved as {filepath}")


def plot_mobility_msd(
    c_type,
    times,
    msds,
    time_stderr,
    msd_stderr,
    path,
):
    # Create the first figure that will be replotted each time
    plt.figure()
    times, msds = hf.parallel_sort(times, msds)
    mobility, mob_error, r_squared = plot_msd(
        times,
        msds,
        time_stderr,
        msd_stderr,
        c_type,
        path,
    )
    print(
            "----------------------------------------\n",
            f"{c_type.capitalize()} mobility = {mobility:.2E} ",
            "+/- {mob_error:.2E} cm^{2} V^{-1} s^{-1}\n",
            "----------------------------------------"
            )
    plt.close()
    return mobility, mob_error, r_squared


def carrier_plots(
        c_type, carrier_data, chromo_list, snap, freq_cut, threeD, path
        ):
    print(f"Considering the transport of {c_type}...")
    if c_type == "hole":
        carrier_history = carrier_data["hole_history"]
    else:
        carrier_history = carrier_data["electron_history"]

    print("Obtaining mean squared displacements...")
    times, msds, time_stderr, msd_stderr = get_times_msds(carrier_data)

    print(f"Plotting distribution of {c_type} displacements")
    plot_displacement_dist(carrier_data, c_type, path)

    print("Calculating mobility...")
    mobility, mob_error, r_squared = plot_mobility_msd(
            c_type, times, msds, time_stderr, msd_stderr, path
            )

    if three_D:
        print("Plotting hop vector distribution")
        plot_hop_vectors(carrier_data, chromo_list, snap, c_type, path)

        print(f"Calculating {c_type} trajectory anisotropy...")
        anisotrophy = plot_anisotropy(carrier_data, c_type, path)

        if carrier_history is not None:
            print("Determining carrier hopping connections...")
            plot_connections(
                chromo_list,
                carrier_history,
                c_type,
                path,
            )

    print(f"Plotting {c_type} hop frequency distribution...")
    freq_cut = plot_frequency_dist(
            c_type, carrier_history, freq_cut, path
            )

    print(f"Plotting {c_type} net hop frequency distribution...")
    plot_net_frequency_dist(c_type, carrier_history, path)

    print("Plotting (total - net hops) discrepancy distribution...")
    plot_discrepancy_frequency_dist(c_type, carrier_history, path)

    return anisotropy, mobility, mob_error, r_squared, freq_cut


def main(
        combined_data,
        temp,
        path,
        three_D=False,
        d_freqcut=None,
        a_freqcut=None,
        d_sepcut=None,
        a_sepcut=None,
        d_ocut=None,
        a_ocut=None,
        d_ticut=None,
        a_ticut=None,
        generate_tcl=False,
        sequence_donor=None,
        sequence_acceptor=None,
        backend=None,
        use_vrh=False,
        koopmans=None,
        boltz_pen=False,
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

    # Create the figures path if it doesn't already exist
    fig_dir = os.makedirs(os.path.join(path, "figures"), exist_ok=True)
    # Load in all the required data
    data_dict = {}
    print("\n")
    print("---------- KMC_ANALYZE ----------")
    print(path)
    print("---------------------------------")

    hole_data, elec_data = split_carriers(combined_data)
    # Calculate the mobilities
    if hole_data["id"]:
        c_type = "hole"
        carrier_data = hole_data

        anisotropy, mobility, mob_error, r_squared, d_freqcut = carrier_plots(
                c_type,
                carrier_data,
                chromo_list,
                snap,
                d_freqcut,
                threeD,
                fig_dir
                )

        data_dict[f"{c_type}_anisotropy"] = anisotropy
        data_dict[f"{c_type}_mobility"] = mobility
        data_dict[f"{c_type}_mobility_err"] = mob_error
        data_dict[f"{c_type}_mobility_r_squared"] = r_squared

    if elec_data["id"]:
        c_type = "electron"
        carrier_data = elec_data

        anisotropy, mobility, mob_error, r_squared, a_freqcut = carrier_plots(
                c_type,
                carrier_data,
                chromo_list,
                snap,
                a_freqcut,
                threeD,
                fig_dir
                )

        data_dict[f"{c_type}_anisotropy"] = anisotropy
        data_dict[f"{c_type}_mobility"] = mobility
        data_dict[f"{c_type}_mobility_err"] = mob_error
        data_dict[f"{c_type}_mobility_r_squared"] = r_squared

    # Now plot the distributions!
    chromo_mol_id = get_molecule_ids(snap, chromo_list)

    plot_energy_levels(chromo_list, data_dict, fig_dir)

    orientations = get_orientations(chromo_list, snap)

    d_sepcut, a_sepcut = plot_neighbor_hist(
        chromo_list, chromo_mol_id, box, d_sepcut, a_sepcut, fig_dir,
    )
    d_ocut, a_ocut = plot_orientation_hist(
        chromo_list, chromo_mol_id, orientations, d_ocut, a_ocut, fig_dir,
    )
    d_ticut, a_ticut = plot_ti_hist(
        chromo_list, chromo_mol_id, d_ticut, a_ticut, fig_dir,
    )
    cutoff_dict = create_cutoff_dict(
        d_sepcut,
        a_sepcut,
        d_ocut,
        a_ocut,
        d_ticut,
        a_ticut,
        d_freqcut,
        a_freqcut,
    )
    print("Cut-offs specified (value format: [donor, acceptor])")
    print(*[f"\t{i}" for i in cutoff_dict.items()], sep="\n")

    clusters = get_clusters(chromo_list, snap, rmax=None)

    if three_D:
        print("Plotting 3D cluster location plot...")
        plot_clusters_3D(chromo_list, clusters, box, generate_tcl, path)

    plot_mixed_hopping_rates(
        chromo_list,
        clusters,
        chromo_mol_id,
        data_dict,
        cutoff_dict,
        temp,
        path,
        use_vrh=use_vrh,
        koopmans=koopmans,
        boltz_pen=boltz_pen,
    )

    print("Plotting cluster size distribution...")
    plot_cluster_size_dist(clusters, fig_dir)

    print("Writing CSV Output File...")
    write_csv(data_dict, path)

    print("Plotting Mobility and Anisotropy progressions...")
    if sequence_donor is not None:
        if data_dict["hole_anisotropy"]:
            plot_temp_progression(
                sequence_donor,
                data_dict["hole_mobility"],
                data_dict["hole_mobility_err"],
                data_dict["hole_anisotropy"],
                "hole",
                path,
            )
    if sequence_acceptor is not None:
        data_dict["electron_anisotropy"]
        data_dict["electron_mobility"]
        if data_dict["electron_anisotropy"]:
            plot_temp_progression(
                sequence_acceptor,
                data_dict["electron_mobility"],
                data_dict["electron_mobility_err"],
                data_dict["electron_anisotropy"],
                "electron",
                path,
            )
    else:
        print("Skipping plotting mobility evolution.")
