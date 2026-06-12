# -*- coding: utf-8 -*-
# random_displacement_analysis.py
"""
Author:   Raghuveer Parthasarathy
Date: June 2, 2026

Last modified June 9, 2026 -- Raghu Parthasarathy

Description
-----------

Code to extract from single-fish behavior analysis properties of inter-bout
intervals (IBIs), defined as sequences of frames for which isActive == False.

For each IBI (excluding the first and last per trajectory), computes:
  - r_mm_mean, gamma_mean: mean radial position and polar angle (circular mean),
    excluding bad tracking frames
  - r_mm_std, gamma_std: corresponding standard deviations
  - Delta_r_mm, Delta_gamma: change in mean radial coordinate r and 
    polar angle gamma to the NEXT IBI
  - Delta_s_mm, Delta_theta: change in displacement (|Delta \vec{r}|) and 
    heading angle theta to the NEXT IBI
  - IB_duration_s: IBI duration in seconds (including bad tracking frames)
  - Delta_t_s: time from end of this IBI to start of the next IBI (i.e. the
    duration of the intervening bout), in seconds

Results are pooled across datasets into pooled_IB_properties.

See Simulating Zebrafish Trajectories.docx

"""

import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Note: calls from IO_toolkit import load_and_assign_from_pickle only if needed
from behavior_plots import make_2D_histogram
from IO_toolkit import plot_probability_distr

def get_InterBout_properties(datasets):
    """
    For each dataset and fish, identify inter-bout intervals (frames where
    isActive == False) and compute properties of each IBI.

    Outputs rows for IBIs indexed 1 through N-2 (second to second-to-last),
    skipping the first and last IBIs to avoid edge effects. For each such IBI,
    Delta_r_mm and Delta_gamma describe the change from this IBI to the next,
    and Delta_t_s is the duration of the intervening bout.

    Bad tracking frames are excluded from means and standard deviations, but
    are counted in IB_duration_s.

    Inputs
    ------
    datasets : list of dataset dictionaries, each containing:
        - "isActive_Fish{k}": frames dictionary with "combine_frames" (2 x N_bouts)
        - "radial_position_mm", "polar_angle_rad": Nframes x Nfish arrays
        - "bad_bodyTrack_frames": frames dictionary with "raw_frames"
        - "frameArray", "fps", "Nfish"

    Returns
    -------
    all_results : list of dicts, one per (dataset, fish, IBI) row
    pooled_IB_properties : dict mapping property name -> 1D numpy array,
        pooled across all datasets and fish
    """
    all_results = []

    for j, dataset in enumerate(datasets):
        fps = dataset["fps"]
        Nfish = dataset["Nfish"]
        frameArray = np.array(dataset["frameArray"]).astype(int)
        frame_start = int(frameArray.min())
        frame_end = int(frameArray.max())
        idx_offset = frame_start
        bad_frames = set(np.array(dataset["bad_bodyTrack_frames"]["raw_frames"]).astype(int))

        r_mm = dataset["radial_position_mm"]   # Nframes x Nfish
        gamma = dataset["polar_angle_rad"]      # Nframes x Nfish, in [-pi, pi]

        for k in range(Nfish):
            active_info = dataset[f"isActive_Fish{k}"]["combine_frames"]
            if active_info.shape[1] == 0:
                # No bouts detected: entire recording is one IBI, skip
                continue

            bout_starts = active_info[0, :].astype(int)
            bout_durations = active_info[1, :].astype(int)
            bout_ends = bout_starts + bout_durations - 1  # last frame of each bout

            # Build list of IBIs as (ibi_start_frame, ibi_end_frame)
            ibis = []
            if bout_starts[0] > frame_start:
                ibis.append((frame_start, bout_starts[0] - 1))
            for i in range(len(bout_starts) - 1):
                ibi_start = bout_ends[i] + 1
                ibi_end = bout_starts[i + 1] - 1
                if ibi_start <= ibi_end:
                    ibis.append((ibi_start, ibi_end))
            if bout_ends[-1] < frame_end:
                ibis.append((bout_ends[-1] + 1, frame_end))

            N_ibis = len(ibis)
            if N_ibis < 3:
                # Need at least 3 IBIs to have a valid second-to-second-to-last range
                continue

            # --- Compute means and stds for all IBIs (0 through N-1) ---
            r_means = np.full(N_ibis, np.nan)
            gamma_means = np.full(N_ibis, np.nan)
            r_stds = np.full(N_ibis, np.nan)
            gamma_stds = np.full(N_ibis, np.nan)
            ibi_durations_s = np.zeros(N_ibis)
            x_means = np.full(N_ibis, np.nan)
            y_means = np.full(N_ibis, np.nan)
            theta = np.full(N_ibis, np.nan)

            for idx_i, (ibi_start, ibi_end) in enumerate(ibis):
                ibi_frames = np.arange(ibi_start, ibi_end + 1, dtype=int)
                ibi_durations_s[idx_i] = len(ibi_frames) / fps

                # Exclude bad tracking frames for means/stds
                good_frames = ibi_frames[~np.isin(ibi_frames, list(bad_frames))]
                if len(good_frames) == 0:
                    continue

                good_idx = good_frames - idx_offset
                r_vals = r_mm[good_idx, k]
                gamma_vals = gamma[good_idx, k]

                r_means[idx_i] = np.mean(r_vals)
                r_stds[idx_i] = np.std(r_vals)

                # Circular mean and circular std for gamma
                sin_m = np.mean(np.sin(gamma_vals))
                cos_m = np.mean(np.cos(gamma_vals))
                gamma_means[idx_i] = np.arctan2(sin_m, cos_m)
                R = np.sqrt(sin_m**2 + cos_m**2)
                # Circular std is undefined for R==0; clip to avoid log(0)
                R_clipped = np.clip(R, 1e-12, 1.0)
                gamma_stds[idx_i] = np.sqrt(-2.0 * np.log(R_clipped))

                # Cartesian coordinates, for magnitude of displacement calc.
                x_means[idx_i] = r_means[idx_i]*np.cos(gamma_means[idx_i])
                y_means[idx_i] = r_means[idx_i]*np.sin(gamma_means[idx_i])

                # Heading angle from previous displacement
                if idx_i==0:
                    # set theta[0] to a random number. This is unnecessary
                    # since theta[0] is not used, but it may be that theta[0]
                    # is used for some future application.
                    theta[idx_i] = np.random.uniform(low=0.0, high=2.0*np.pi)
                else:
                    theta[idx_i] = np.arctan2(y_means[idx_i] - y_means[idx_i-1], 
                                              x_means[idx_i] - x_means[idx_i-1])

            # --- Output rows for IBIs 1 through N-2 ---
            for idx_i in range(1, N_ibis - 1):
                # Skip rows where this or the next IBI has no valid position data
                if np.isnan(r_means[idx_i]) or np.isnan(r_means[idx_i + 1]):
                    continue

                _, ibi_end_i = ibis[idx_i]
                next_ibi_start, _ = ibis[idx_i + 1]

                # Duration of the bout between IBI[idx_i] and IBI[idx_i+1]
                bout_between_nframes = next_ibi_start - (ibi_end_i + 1)
                Delta_t_s = bout_between_nframes / fps

                Delta_r_mm = r_means[idx_i + 1] - r_means[idx_i]

                # For magnitude of displacement
                Delta_x_mm = x_means[idx_i + 1] - x_means[idx_i]
                Delta_y_mm = y_means[idx_i + 1] - y_means[idx_i]
                Delta_s_mm = np.sqrt(Delta_x_mm**2 + Delta_y_mm**2)
                
                # For change in heading angle of displacement
                raw_Delta_theta = theta[idx_i + 1] - theta[idx_i]
                Delta_theta = (raw_Delta_theta + np.pi) % (2.0 * np.pi) - np.pi

                # Channge in polar angle
                raw_Delta_gamma = gamma_means[idx_i + 1] - gamma_means[idx_i]
                Delta_gamma = (raw_Delta_gamma + np.pi) % (2.0 * np.pi) - np.pi

                all_results.append({
                    "dataset_number": j,
                    "r_mm_mean":     r_means[idx_i],
                    "gamma_mean":    gamma_means[idx_i],
                    "r_mm_std":      r_stds[idx_i],
                    "gamma_std":     gamma_stds[idx_i],
                    "Delta_r_mm":    Delta_r_mm,
                    "Delta_gamma":   Delta_gamma,
                    "Delta_s_mm":    Delta_s_mm,
                    "theta":         theta[idx_i],
                    "Delta_theta":         Delta_theta,
                    "Delta_t_s":     Delta_t_s,
                    "IB_duration_s": ibi_durations_s[idx_i],
                })

    # Build pooled_IB_properties
    property_keys = ["r_mm_mean", "gamma_mean", "r_mm_std", "gamma_std",
                     "Delta_r_mm", "Delta_gamma", "Delta_s_mm",
                     "theta", "Delta_theta", "Delta_t_s", "IB_duration_s"]
    pooled_IB_properties = {
        key: np.array([row[key] for row in all_results])
        for key in property_keys
    }

    print(f'\nFound {len(all_results)} inter-bout intervals '
          f'(second to second-to-last, across all datasets and fish).')

    return all_results, pooled_IB_properties


def export_interbout_CSV(all_results, default_filename="interbout_properties.csv"):
    """
    Export IBI properties to a CSV file, pooling rows across all datasets.

    Columns: dataset_number, r_mm_mean, gamma_mean, r_mm_std, gamma_std,
             Delta_r_mm, Delta_gamma, Delta_s_mm,
             theta, Delta_theta, Delta_t_s, IB_duration_s

    Inputs
    ------
    all_results : list of dicts returned by get_InterBout_properties()
    default_filename : str, default CSV filename (used if user presses Enter)
    """
    user_input = input(f'\nFilename for CSV (default: "{default_filename}"): ').strip()
    filename = user_input if user_input else default_filename

    columns = [
        "dataset_number", "r_mm_mean", "gamma_mean", "r_mm_std", "gamma_std",
        "Delta_r_mm", "Delta_gamma", "Delta_s_mm",
        "theta", "Delta_theta","Delta_t_s", "IB_duration_s"
    ]

    formatted_results = []
    for row in all_results:
        formatted_row = {
            "dataset_number": int(row["dataset_number"]),

            "r_mm_mean": f'{float(row["r_mm_mean"]):.3f}',
            "r_mm_std": f'{float(row["r_mm_std"]):.3f}',
            "Delta_r_mm": f'{float(row["Delta_r_mm"]):.3f}',
            "Delta_t_s": f'{float(row["Delta_t_s"]):.3f}',
            "IB_duration_s": f'{float(row["IB_duration_s"]):.3f}',

            "gamma_mean": f'{float(row["gamma_mean"]):.4f}',
            "gamma_std": f'{float(row["gamma_std"]):.4f}',
            "Delta_gamma": f'{float(row["Delta_gamma"]):.4f}',
            "Delta_s_mm": f'{float(row["Delta_s_mm"]):.3f}',
            "theta":  f'{float(row["theta"]):.4f}',
            "Delta_theta": f'{float(row["Delta_theta"]):.4f}'}
        formatted_results.append(formatted_row)

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(formatted_results)

    print(f'Saved {len(all_results)} rows to: {filename}')


def plot_interbout_histograms(pooled_IB_properties):
    """
    Plot histograms of Delta_r_mm, Delta_gamma, Delta_t_s, and IB_duration_s
    from the pooled IBI properties. Print mean and std for each.

    Input
    -----
    pooled_IB_properties : dict returned by get_InterBout_properties()
    """
    keys = ["Delta_r_mm", "Delta_gamma", "Delta_t_s", "IB_duration_s"]
    labels = {
        "Delta_r_mm":    "Δr (mm)",
        "Delta_gamma":   "Δγ (rad)",
        "Delta_t_s":     "Δt between IBIs (s)",
        "IB_duration_s": "IBI duration (s)",
    }
    bin_specs = {
        "Delta_r_mm":    50,
        "Delta_gamma":   np.linspace(-np.pi, np.pi, 37),
        "Delta_t_s":     50,
        "IB_duration_s": 50,
    }
    xtick_specs = {
        "Delta_gamma": ([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                        ['-π', '-π/2', '0', 'π/2', 'π']),
    }

    print('\nInter-bout interval summary statistics:')
    print(f'  {"Property":<20}  {"Mean":>10}  {"Std":>10}  {"N":>6}')
    print('  ' + '-' * 52)

    _, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, key in enumerate(keys):
        vals = pooled_IB_properties[key]
        good = vals[np.isfinite(vals)]
        mean_v = np.mean(good)
        std_v = np.std(good)
        print(f'  {key:<20}  {mean_v:>10.4f}  {std_v:>10.4f}  {len(good):>6d}')

        axes[i].hist(good, bins=bin_specs[key], color='steelblue',
                     edgecolor='white', linewidth=0.5)
        axes[i].set_yscale('log')
        axes[i].set_xlabel(labels[key], fontsize=12)
        axes[i].set_ylabel('Count', fontsize=12)
        axes[i].set_title(f'{labels[key]}\nmean={mean_v:.3f},  std={std_v:.3f}',
                          fontsize=11)
        if key in xtick_specs:
            ticks, ticklabels = xtick_specs[key]
            axes[i].set_xticks(ticks)
            axes[i].set_xticklabels(ticklabels)

    plt.tight_layout()
    plt.show(block=False)


def build_radial_bin_distributions(pooled_IB_properties, arena_radius_mm,
                                    bin_size_mm=1.0):
    """
    Bin the pooled IBI properties by radial position (r_mm_mean) and store
    the joint empirical distribution of the four step properties in each bin.

    For simulation, given the current radial position, look up which bin it
    falls in and draw a random 6-tuple (with replacement) from that bin's
    observations.  This preserves the joint distribution and all correlations
    without requiring a high-dimensional histogram.
    Originally a 4-tuple (Delta_r_mm, Delta_gamma, Delta_t_s, IB_duration_s),
    and then add (Delta_s_mm, Delta_theta) to facilitate pair simulations later;
    see June 9, 2026 notes.


    Inputs
    ------
    pooled_IB_properties : dict returned by get_InterBout_properties()
    arena_radius_mm : float, outer radius of the arena (mm)
    bin_size_mm : float, width of each radial bin (mm); default 1.0

    Returns
    -------
    radial_bins : list of dicts, length = number of radial bins.
        radial_bins[i] contains:
            "r_edges"       : (r_low, r_high) for this bin (mm)
            "Delta_r_mm"    : 1D array of observed values in this bin
            "Delta_gamma"   : 1D array
            "Delta_t_s"     : 1D array
            "IB_duration_s" : 1D array
            "Delta_s_mm" : 1D array
            "Delta_theta" : 1D array
            "N"             : number of observations in this bin
    bin_edges : 1D array of radial bin edges (mm), length = n_bins + 1
    """
    step_keys = ["Delta_r_mm", "Delta_gamma", "Delta_t_s", "IB_duration_s",
                 "Delta_s_mm", "Delta_theta"]

    bin_edges = np.arange(0.0, arena_radius_mm + bin_size_mm, bin_size_mm)
    n_bins = len(bin_edges) - 1

    r = pooled_IB_properties["r_mm_mean"]
    # np.digitize returns 1-indexed bin numbers; clamp to valid range
    bin_idx = np.clip(np.digitize(r, bin_edges) - 1, 0, n_bins - 1)

    radial_bins = []
    for i in range(n_bins):
        mask = bin_idx == i
        entry = {
            "r_edges": (bin_edges[i], bin_edges[i + 1]),
            "N": int(mask.sum()),
        }
        for key in step_keys:
            entry[key] = pooled_IB_properties[key][mask]
        radial_bins.append(entry)

    # Print a summary of bin occupancy
    print(f'\nRadial bin occupancy (bin size = {bin_size_mm} mm, '
          f'arena radius = {arena_radius_mm} mm):')
    print(f'  {"Bin (mm)":<16}  {"N obs":>6}')
    for i, b in enumerate(radial_bins):
        lo, hi = b["r_edges"]
        print(f'  {lo:.1f} – {hi:.1f}{"":>8}  {b["N"]:>6}')

    return radial_bins, bin_edges


def sample_from_radial_bin(radial_bins, r_current, rng=None):
    """
    Given the current radial position, find the corresponding radial bin and
    draw one random tuple
    (Delta_r_mm, Delta_gamma, Delta_t_s, IB_duration_s, Delta_s_mm, Delta_theta)
    from the empirical observations in that bin.
    If the bin is empty (no observations), the nearest non-empty bin is used.

    Inputs
    ------
    radial_bins : list of dicts returned by build_radial_bin_distributions()
    r_current : float, current radial position (mm)
    rng : numpy.random.Generator or None

    Returns
    -------
    sample : dict with keys Delta_r_mm, Delta_gamma, Delta_t_s, IB_duration_s
             Delta_s_mm, Delta_theta
    """
    if rng is None:
        rng = np.random.default_rng()

    # Find the bin whose r_edges bracket r_current
    bin_i = None
    for i, b in enumerate(radial_bins):
        lo, hi = b["r_edges"]
        if lo <= r_current < hi:
            bin_i = i
            break
    if bin_i is None:
        # r_current is beyond the last edge — use the outermost bin
        bin_i = len(radial_bins) - 1

    # If chosen bin is empty, walk outward then inward to find the nearest
    # non-empty bin
    if radial_bins[bin_i]["N"] == 0:
        n_bins = len(radial_bins)
        found = False
        for offset in range(1, n_bins):
            for candidate in [bin_i - offset, bin_i + offset]:
                if 0 <= candidate < n_bins and radial_bins[candidate]["N"] > 0:
                    bin_i = candidate
                    found = True
                    break
            if found:
                break

    b = radial_bins[bin_i]
    idx = rng.integers(0, b["N"])
    return {
        "Delta_r_mm":    float(b["Delta_r_mm"][idx]),
        "Delta_gamma":   float(b["Delta_gamma"][idx]),
        "Delta_t_s":     float(b["Delta_t_s"][idx]),
        "IB_duration_s": float(b["IB_duration_s"][idx]),
        "Delta_s_mm": float(b["Delta_s_mm"][idx]),
        "Delta_theta": float(b["Delta_theta"][idx]),
    }


def sim_sampled_random_walk(radial_bins, arena_radius_mm, r_init=None,
                             gamma_init=None, T_total_s=600.0,
                             plot_positions=False, rng=None):
    """
    Simulate a random walk of a zebrafish using the empirical IBI distributions
    binned by radial position.

    At each step the fish pauses for a drawn IB_duration_s (the IBI), then
    undergoes a bout of drawn duration Delta_t_s during which its position
    changes by (Delta_r_mm, Delta_gamma).  This repeats until elapsed time
    exceeds T_total_s.

    Radial boundary conditions: r < 0 is reflected through the origin (r → -r,
    gamma → gamma + pi); r > arena_radius_mm is reflected at the wall
    (r → 2*arena_radius - r). In impose_radial_boundary()

    Inputs
    ------
    radial_bins : list of dicts from build_radial_bin_distributions()
    arena_radius_mm : float, radius of the arena (mm)
    r_init : float or None; initial radial position (mm).  If None, drawn
             from a uniform distribution over the arena disk:
             r = arena_radius_mm * sqrt(U(0,1))
    gamma_init : float or None; initial polar angle (rad).  If None, drawn
                 uniformly from [0, 2*pi).
    T_total_s : float, minimum total simulation time in seconds (default 600)
    plot_positions : bool, if True scatter-plot all simulated positions with
                     semi-transparent circles (default False)
    rng : numpy.random.Generator or None

    Returns
    -------
    r_sim : 1D numpy array of radial positions at the start of each IBI (mm)
    gamma_sim : 1D numpy array of polar angles at the start of each IBI (rad)
    t_sim : 1D numpy array of elapsed times at the start of each IBI (s)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initial position
    if r_init is None:
        r = arena_radius_mm * np.sqrt(rng.uniform())
    else:
        r = float(r_init)
    if gamma_init is None:
        gamma = rng.uniform(0.0, 2.0 * np.pi)
    else:
        gamma = float(gamma_init)

    r_list = [r]
    gamma_list = [gamma]
    t_list = [0.0]
    t = 0.0

    while t < T_total_s:
        sample = sample_from_radial_bin(radial_bins, r, rng=rng)

        t += sample["IB_duration_s"] + sample["Delta_t_s"]

        r_new = r + sample["Delta_r_mm"]
        gamma_new = gamma + sample["Delta_gamma"]

        # Impose r >= 0 and r <= arena_radius_mm
        r_new, gamma_new = impose_radial_boundary(r_new, arena_radius_mm, gamma_new)

        # Wrap gamma to [-pi, pi]
        gamma_new = (gamma_new + np.pi) % (2.0 * np.pi) - np.pi

        r = r_new
        gamma = gamma_new
        r_list.append(r)
        gamma_list.append(gamma)
        t_list.append(t)

    r_sim = np.array(r_list)
    gamma_sim = np.array(gamma_list)
    t_sim = np.array(t_list)

    if plot_positions:
        x_sim = r_sim * np.cos(gamma_sim)
        y_sim = r_sim * np.sin(gamma_sim)

        fig = plt.figure(figsize=(10, 6))

        # All points
        ax1 = fig.add_subplot(121)
        ax1.scatter(x_sim, y_sim, s=20, alpha=0.3, color='steelblue',
                   edgecolors='none')
        # Arena boundary
        phi = np.linspace(0.0, 2.0 * np.pi, 300)
        ax1.plot(arena_radius_mm * np.cos(phi), arena_radius_mm * np.sin(phi),
                'k-', linewidth=1.5)
        ax1.set_xlabel('x (mm)', fontsize=12)
        ax1.set_ylabel('y (mm)', fontsize=12)
        ax1.set_title(
            f'Simulated positions  (N steps = {len(r_sim)},  '
            f'T = {t_sim[-1]:.0f} s)', fontsize=13)
        ax1.set_aspect('equal')

        # Polar histogram plot of angles
        num_bins = 90
        bins = np.linspace(-np.pi, np.pi, num_bins + 1)
        counts, bin_edges = np.histogram(gamma_sim, bins=bins)
        widths = 2*np.pi/num_bins
        bin_centers = bin_edges[:-1] + widths / 2
        ax2 = fig.add_subplot(122, projection='polar')
        _ = ax2.bar(
            bin_centers, 
            counts, 
            width=widths, 
            bottom=0.0, 
            edgecolor='lightblue', 
            color='skyblue', 
            alpha=0.75
        )
        ax2.set_title("Polar Angle Histogram", va='bottom', fontsize=14)

        plt.tight_layout()
        plt.show(block=False)

    return r_sim, gamma_sim, t_sim

def impose_radial_boundary(r, arena_radius_mm, gamma = None):
    """
    Impose radial boundary conditions (0 <= r <= arena_radius)

    Inputs
    ------
    r : float, radial position (mm)
    arena_radius_mm : float, radius of the arena (mm)
    gamma : polar angle, in case of reflection

    Returns
    -------
    r : float, radial position (mm)
    gamma : float, polar angle (radians)
    """

    # Reflect through origin if r goes negative
    if r < 0.0:
        r = -r
        if gamma is not None:
            gamma = gamma + np.pi
    # Reflect at arena wall if r exceeds arena radius
    if r > arena_radius_mm:
        r = 2.0 * arena_radius_mm - r
    # Clamp to [0, arena_radius_mm] in case of extreme overshooting
    r = float(np.clip(r, 0.0, arena_radius_mm))

    return r, gamma

def load_interbout_CSV(filename=None):
    """
    Load IBI properties from a previously exported CSV file and reconstruct
    the all_results list and pooled_IB_properties dict produced by
    get_InterBout_properties().

    Inputs
    ------
    filename : str or None; if None, the user is prompted for a path

    Returns
    -------
    all_results : list of dicts (one row per IBI)
    pooled_IB_properties : dict mapping property name -> 1D numpy array
    """
    if filename is None:
        user_input = input(
            'CSV filename to load (default: "interbout_properties.csv"): ').strip()
        filename = user_input if user_input else "interbout_properties.csv"

    float_keys = ["r_mm_mean", "gamma_mean", "r_mm_std", "gamma_std",
                  "Delta_r_mm", "Delta_gamma", "Delta_s_mm", 
                  "theta", "Delta_theta", "Delta_t_s", "IB_duration_s"]

    all_results = []
    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {"dataset_number": int(row["dataset_number"])}
            for key in float_keys:
                entry[key] = float(row[key])
            all_results.append(entry)

    property_keys = ["r_mm_mean", "gamma_mean", "r_mm_std", "gamma_std",
                     "Delta_r_mm", "Delta_gamma", "Delta_s_mm",
                     "theta", "Delta_theta", "Delta_t_s", "IB_duration_s"]
    pooled_IB_properties = {
        key: np.array([row[key] for row in all_results])
        for key in property_keys
    }

    print(f'Loaded {len(all_results)} IBI rows from: {filename}')
    return all_results, pooled_IB_properties


def plot_radial_bin_occupancy(radial_bins, bin_edges):
    """
    Bar chart showing the number of IBI observations in each radial bin.

    Inputs
    ------
    radial_bins : list of dicts returned by build_radial_bin_distributions()
    bin_edges   : 1D array of radial bin edges (mm)
    """
    counts = np.array([b["N"] for b in radial_bins])
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)

    _, ax = plt.subplots(figsize=(8, 4))
    ax.bar(centers, counts, width=widths * 0.85, color='steelblue',
           edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Radial position (mm)', fontsize=13)
    ax.set_ylabel('Number of IBIs', fontsize=13)
    ax.set_title('IBI observations per radial bin', fontsize=14)
    plt.tight_layout()
    plt.show(block=False)


def calc_dh_vec(r, gamma):
    """
    Vector from Fish 0 to Fish 1
    Two-element numpy array of (dx, dy)

    Inputs
    ------
    r : 2-tuple of float, radial coordinate
    gamma : 2-tuple of float, polar angle

    Returns
    -------
    dh_vec : Two-item numpy array of (dx, dy)
    """

    # calculate initial head-head vector (two-item list for x, y)
    dx = r[1]*np.cos(gamma[1]) - r[0]*np.cos(gamma[0])
    dy = r[1]*np.sin(gamma[1]) - r[0]*np.sin(gamma[0])
    dh_vec = np.array([dx, dy])

    return dh_vec


def calc_relative_orientation(theta, dh_vec):
    """ 
    Calculate the relative orientation of each fish with respect to the
    head-to-head vector to the other fish.

    Inputs:
        theta : heading angles for fish 0, 1 (two element list)
        dh_vec : Vector from Fish 0 to Fish 1, two-element numpy array of (dx, dy)

    Outputs: 
        relative_orientation : two element numpy array,
            relative orientation (phi), radians, for fish 0 and fish 1
            Signed angles in range [-π, π] 
    
    Note: Valid only for Nfish==2. (Does not check this.) .
    """
    
    # All heading angles
    theta = np.array(theta)
    
    # Unit vectors for heading directions
    v0 = np.array([np.cos(theta[0]), np.sin(theta[0])])
    v1 = np.array([np.cos(theta[1]), np.sin(theta[1])])

    # Normalize dh_vec
    dh_unit = dh_vec / np.linalg.norm(dh_vec)

    # Calculate dot products for magnitude
    dot_product_0 = np.sum(v0 * dh_unit)
    dot_product_1 = np.sum(v1 * -dh_unit)

    # Calculate unsigned angles
    phi0_unsigned = np.arccos(np.clip(dot_product_0, -1.0, 1.0))
    phi1_unsigned = np.arccos(np.clip(dot_product_1, -1.0, 1.0))
    
    # Calculate cross products to determine sign (z-component for 2D)
    # cross_z = v_x * dh_y - v_y * dh_x
    cross_z_0 = v0[0] * dh_unit[1] - v0[1] * dh_unit[0]
    cross_z_1 = v1[0] * (-dh_unit[1]) - v1[1] * (-dh_unit[0])
    
    # Apply sign: positive if cross product in -z, negative if in +z
    phi0 = np.where(cross_z_0 >= 0, -phi0_unsigned, phi0_unsigned)
    phi1 = np.where(cross_z_1 >= 0, -phi1_unsigned, phi1_unsigned)
    
    relative_orientation = np.array([phi0, phi1])

    return relative_orientation

def get_random_turning_angle(rel_orient, dHH, turn_2Dhist_mean, turn_2Dhist_std,
                              rel_orient_bins, dHH_bins, rng=None):

    """
    Get random turning angle based on the mean and standard deviation of
    empirical turning angles at a given relative orientation and inter-fish
    distance. (Draw from Gaussian distribution)

    Inputs
    ------
    rel_orient : float, relative orientation angle to the other fish.
    dHH : float, head-head distance (mm)
    turn_2Dhist_mean : numpy array of mean turning angle binned (2D) by
                       rel. orientation and head-head distance
    turn_2Dhist_std : numpy array of std. dev. turning angle in the 2D bins
    rel_orient_bins : bin centers used for relative orientation angles
    dHH_bins : bin centers used for head-head distance (mm)
    rng : numpy.random.Generator or None

    Returns
    -------
    theta_T : float, turning angle (radians), in [-pi, pi]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Find closest rel_orient, dHH bin
    rel_orient_idx = np.argmin(np.abs(rel_orient - rel_orient_bins))
    dHH_idx = np.argmin(np.abs(dHH - dHH_bins))
    theta_T = rng.normal(turn_2Dhist_mean[rel_orient_idx, dHH_idx],
                         turn_2Dhist_std[rel_orient_idx, dHH_idx])
    # Make sure in [-pi, pi]
    theta_T = (theta_T + np.pi) % (2.0 * np.pi) - np.pi
    theta_T = np.clip(theta_T, a_min = -1.0*np.pi, a_max = np.pi)

    return theta_T



def sim_pair_interacting_walk(radial_bins, arena_radius_mm, 
                              turn_2Dhist_mean, turn_2Dhist_std, 
                              rel_orient_bins, dHH_bins,
                              r_init=None,
                              gamma_init=None, theta_init=None, T_total_s=600.0,
                              plot_positions=False, rng=None):
    """
    Simulate a random walk of a pair of zebrafish using the 
    empirical IBI step distributions from single fish data, binned by 
    radial position, and the empirical turn-angle preferences from 
    pair data.

    At each step each fish pauses for a drawn IB_duration_s (the IBI), then
    undergoes a bout of drawn duration Delta_t_s during which its position
    changes by displacement Delta_s_mm. The angle of the step is random,
    based on the empirical turn-angle preferences from 
    pair data.
    This repeats until elapsed time for each fish exceeds T_total_s.

    Radial boundary conditions: r < 0 is reflected through the origin (r → -r,
    gamma → gamma + pi); r > arena_radius_mm is reflected at the wall
    (r → 2*arena_radius - r). In impose_radial_boundary()

    Inputs
    ------
    radial_bins : list of dicts from build_radial_bin_distributions()
    arena_radius_mm : float, radius of the arena (mm)
    turn_2Dhist_mean : numpy array of mean turning angle binned (2D) by 
                       rel. orientation and head-head distance
    turn_2Dhist_std : numpy array of std. dev. turning angle in the 2D bins
    rel_orient_bins : bin centers used for relative orientation angles
    dHH_bins : bin centers used for head-head distance (mm)
    r_init : 2-tuple of float, or None; initial radial position (mm).  
             If None, drawn both r_init from a uniform distribution 
             over the arena disk:
             r = arena_radius_mm * sqrt(U(0,1))
    gamma_init : float or None; initial polar angle (rad).  If None, drawn
                 uniformly from [0, 2*pi).
    theta_init : float or None; initial heading angle (rad).  If None, drawn
                 uniformly from [0, 2*pi).
    T_total_s : float, minimum total simulation time in seconds (default 600)
    plot_positions : bool, if True scatter-plot all simulated positions with
                     semi-transparent circles (default False)
    rng : numpy.random.Generator or None

    Returns
    -------
    r_sim : list of 2 1D numpy arrays of radial positions at the start of 
            each IBI (mm), for each fish
    gamma_sim : list of 2 1D numpy arrays of polar angles at the start of 
            each IBI (rad), for each fish
    t_sim : list of 2 1D numpy arrays of elapsed times at the start of each IBI (s),
            for each fish
    """

    if rng is None:
        rng = np.random.default_rng()

    # Initial radial positions
    if r_init is None:
        r = arena_radius_mm * np.sqrt(rng.uniform(low=0.0, high=1.0, size=(2,)))
    else:
        r = r_init
    if gamma_init is None:
        gamma = rng.uniform(0.0, 2.0 * np.pi, size=(2,))
    else:
        gamma = gamma_init

    # Initial heading angles
    if theta_init is None:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=(2,))
    else:
        theta = theta_init

    # calculate initial head-head vector (two-item list for x, y)
    # Defined as from fish 0 to fish 1
    dh_vec = calc_dh_vec(r, gamma)

    # Calculate initial relative orientation for each fish
    relative_orientation = calc_relative_orientation(theta, dh_vec)

    # Initialize each fish; list of lists
    r_list = [[r[0]], [r[1]]]
    gamma_list = [[gamma[0]], [gamma[1]]]
    t_list = [[0.0], [0.0]]
    t = 0.0

    # Counters: total steps and how often the boundary-reflection fallback fires
    n_steps = 0
    n_fallback = 0

    while min(t_list[0][-1], t_list[1][-1]) < T_total_s:

        # which fish has the lowest recent time      
        fish_idx = np.argmin(np.array((t_list[0][-1], t_list[1][-1])))

        r_this = r[fish_idx]
        gamma_this = gamma[fish_idx]
        dHH = np.sqrt(np.sum(dh_vec**2))

        sample = sample_from_radial_bin(radial_bins, r_this, rng=rng)

        t_this = t_list[fish_idx][-1]
        t_this += sample["IB_duration_s"] + sample["Delta_t_s"]

        Delta_s = sample["Delta_s_mm"]

        # Get turning angle from experimental histogram.
        # Resample the turning angle until the new position is within the
        # arena, up to MAX_TRIES. If no valid direction is found (e.g. Delta_s
        # too large for the current radius, so no step of that length can land
        # inside), fall back to reflecting the proposed step off the boundary
        # so the loop always terminates.
        MAX_TRIES = 100
        # Current position is fixed across tries (only the direction varies)
        x = r_this*np.cos(gamma_this)
        y = r_this*np.sin(gamma_this)
        gamma_new = gamma[fish_idx]
        r_new = 2.0*arena_radius_mm
        for _try in range(MAX_TRIES):
            theta_T = get_random_turning_angle(relative_orientation[fish_idx],
                                            dHH, turn_2Dhist_mean,
                                            turn_2Dhist_std,
                                            rel_orient_bins, dHH_bins,
                                            rng=rng)
            # Empty (NaN) histogram bins give a NaN turning angle; fall back
            # to a uniform random direction so NaNs don't propagate.
            if np.isnan(theta_T):
                theta_T = rng.uniform(-np.pi, np.pi)

            # Update position. Note change in heading angle is -theta_T
            dx = Delta_s*np.cos(theta[fish_idx] - theta_T)
            dy = Delta_s*np.sin(theta[fish_idx] - theta_T)
            x_new = x + dx
            y_new = y + dy
            gamma_new = np.arctan2(y_new, x_new)
            r_new = np.sqrt(x_new**2 + y_new**2)
            if r_new <= arena_radius_mm:
                break
        else:
            n_fallback += 1
            # No valid direction found: reflect the last proposed step off
            # the wall (reuses the same boundary handling as the single-fish
            # walk). Guarantees r_new <= arena_radius_mm.
            r_new, gamma_new = impose_radial_boundary(r_new, arena_radius_mm,
                                                      gamma_new)

        n_steps += 1
        r[fish_idx] = r_new
        gamma[fish_idx] = gamma_new

        # Wrap gamma to [-pi, pi]
        gamma[fish_idx] = (gamma[fish_idx] + np.pi) % (2.0 * np.pi) - np.pi

        # Update heading to the direction of the ACTUAL displacement this step.
        # In the normal case this equals theta - theta_T exactly; in the
        # fallback (reflection) case it follows the reflected step, so the fish
        # does not "stick" to the wall by repeatedly heading back into it.
        x_final = r_new * np.cos(gamma[fish_idx])
        y_final = r_new * np.sin(gamma[fish_idx])
        dx_actual = x_final - x
        dy_actual = y_final - y
        if dx_actual != 0.0 or dy_actual != 0.0:
            theta[fish_idx] = np.arctan2(dy_actual, dx_actual)
        # else: zero-length step (Delta_s == 0), leave heading unchanged

        # Update head-head vector and relative orientation
        dh_vec = calc_dh_vec(r, gamma)  # fish 0 to 1
        relative_orientation = calc_relative_orientation(theta, dh_vec)

        r_list[fish_idx].append(r[fish_idx])
        gamma_list[fish_idx].append(gamma[fish_idx])
        t_list[fish_idx].append(t_this)

    r_sim = [np.array(r_list[0]), np.array(r_list[1])]
    gamma_sim = [np.array(gamma_list[0]), np.array(gamma_list[1])]
    t_sim = [np.array(t_list[0]), np.array(t_list[1])]

    pct = 100.0 * n_fallback / n_steps if n_steps > 0 else 0.0
    print(f'  sim_pair_interacting_walk: boundary-reflection fallback fired '
          f'{n_fallback} / {n_steps} steps ({pct:.2f}%).')
    if pct > 5.0:
        print('    NOTE: fallback rate > 5%; the empirical Delta_s distribution '
              'may have a large-step tail relative to the arena size.')

    if plot_positions:
        colors = ['steelblue', 'firebrick']
        fig = plt.figure(figsize=(10, 6))

        # Positions of both fish
        ax1 = fig.add_subplot(121)
        for k in range(2):
            x_k = r_sim[k] * np.cos(gamma_sim[k])
            y_k = r_sim[k] * np.sin(gamma_sim[k])
            ax1.scatter(x_k, y_k, s=20, alpha=0.3, color=colors[k],
                        edgecolors='none', label=f'Fish {k}')
        # Arena boundary
        phi = np.linspace(0.0, 2.0 * np.pi, 300)
        ax1.plot(arena_radius_mm * np.cos(phi), arena_radius_mm * np.sin(phi),
                 'k-', linewidth=1.5)
        ax1.set_xlabel('x (mm)', fontsize=12)
        ax1.set_ylabel('y (mm)', fontsize=12)
        ax1.set_title(
            f'Simulated pair positions  (N steps = {len(r_sim[0])}, '
            f'{len(r_sim[1])};  T = {min(t_sim[0][-1], t_sim[1][-1]):.0f} s)',
            fontsize=12)
        ax1.set_aspect('equal')
        ax1.legend(fontsize=10)

        # Polar histogram of polar angles, one distribution per fish
        num_bins = 90
        bins = np.linspace(-np.pi, np.pi, num_bins + 1)
        widths = 2 * np.pi / num_bins
        bin_centers = bins[:-1] + widths / 2
        ax2 = fig.add_subplot(122, projection='polar')
        for k in range(2):
            counts, _ = np.histogram(gamma_sim[k], bins=bins)
            ax2.bar(bin_centers, counts, width=widths, bottom=0.0,
                    edgecolor=colors[k], color=colors[k], alpha=0.4,
                    label=f'Fish {k}')
        ax2.set_title("Polar Angle Histogram", va='bottom', fontsize=14)
        ax2.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1.1))

        plt.tight_layout()
        plt.show(block=False)

    return r_sim, gamma_sim, t_sim


def interpolate_pair_rsim(r_sim, gamma_sim, t_sim, dt_s = 0.04, T_total_s=600.0):
    """
    From simulated trajectories of two fish, where the two fish positions 
    are calculated independently and therefore at different times, interpolate
    values at a regular grid of time points

    Inputs
    ------
    dt_s : float, time step for interpolation (s) (defaul 1/25 s)
    T_total_s : float, minimum total simulation time in seconds (default 600)
    r_sim : list of 2 1D numpy arrays of radial positions at the start of 
            each IBI (mm), for each fish
    gamma_sim : list of 2 1D numpy arrays of polar angles at the start of 
            each IBI (rad), for each fish
    t_sim : list of 2 1D numpy arrays of elapsed times at the start of each IBI (s),
            for each fish

    Returns
    t_array_s : numpy array, interpolated time points (s)
    r_sim_interp : numpy array of shape (2, len(t_array)) of radial positions 
                   the interpolated times.  
    gamma_sim_interp : numpy array of shape (2, len(t_array)) of polar angles 
                   the interpolated times.  
    dHH_mm : inter-fish distance at the interpolated points (mm)

    """
    if T_total_s > min(t_sim[0][-1], t_sim[1][-1]):
        raise ValueError('Interpolation time is greater than simulation time.')

    t_array_s = np.arange(0.0, T_total_s + dt_s, dt_s, )
    r_sim_interp = np.zeros((2, len(t_array_s)))
    gamma_sim_interp = np.zeros((2, len(t_array_s)))
    for j, t in enumerate(t_array_s):
        idx0 = np.argmin(abs(t - t_sim[0]))
        idx1 = np.argmin(abs(t - t_sim[1]))
        r_sim_interp[0, j] = r_sim[0][idx0]
        r_sim_interp[1, j] = r_sim[1][idx1]
        gamma_sim_interp[0, j] = gamma_sim[0][idx0]
        gamma_sim_interp[1, j] = gamma_sim[1][idx1]

    # inter-fish distance
    x0 = r_sim_interp[0,:]*np.cos(gamma_sim_interp[0,:])
    y0 = r_sim_interp[0,:]*np.sin(gamma_sim_interp[0,:])
    x1 = r_sim_interp[1,:]*np.cos(gamma_sim_interp[1,:])
    y1 = r_sim_interp[1,:]*np.sin(gamma_sim_interp[1,:])
    dHH_mm = np.sqrt((x0-x1)**2 + (y0-y1)**2)

    return t_array_s, r_sim_interp, gamma_sim_interp, dHH_mm


def export_turning_histogram_CSV(turn_2Dhist_mean, turn_2Dhist_std,
                                 rel_orient_bins, dHH_bins,
                                 default_filename="turning_histogram.csv"):
    """
    Export the binned mean turning angle and its standard deviation to CSV.

    Writes two files:
      - <filename>           : mean turning angle (rad), with relative-orientation
                               bin centers as the row index and head-head-distance
                               bin centers as the column headers.
      - <base>_std<ext>      : standard deviation (rad), same layout.

    Inputs
    ------
    turn_2Dhist_mean : 2D array (n_relorient_bins x n_dHH_bins)
    turn_2Dhist_std  : 2D array, same shape
    rel_orient_bins  : 1D array of relative-orientation bin centers (rad)
    dHH_bins         : 1D array of head-head-distance bin centers (mm)
    default_filename : str, default CSV filename (used if user presses Enter)
    """
    user_input = input(f'\nFilename for turning-histogram CSV '
                       f'(default: "{default_filename}"): ').strip()
    filename = user_input if user_input else default_filename
    base, ext = os.path.splitext(filename)
    std_filename = base + '_std' + ext

    pd.DataFrame(turn_2Dhist_mean, index=rel_orient_bins,
                 columns=dHH_bins).to_csv(filename)
    pd.DataFrame(turn_2Dhist_std, index=rel_orient_bins,
                 columns=dHH_bins).to_csv(std_filename)
    print(f'Saved turning-angle mean to: {filename}')
    print(f'Saved turning-angle std to:  {std_filename}')


def load_turning_histogram_CSV(filename=None):
    """
    Load the binned mean turning angle and its standard deviation from the two
    CSV files written by export_turning_histogram_CSV(). The relative-orientation
    and head-head-distance bin centers are read back from the row index and
    column headers, respectively.

    Inputs
    ------
    filename : str or None; the mean-file name. If None, the user is prompted.
               The std file is assumed to be the same name with '_std' inserted
               before the extension.

    Returns
    -------
    turn_2Dhist_mean : 2D array (n_relorient_bins x n_dHH_bins)
    turn_2Dhist_std  : 2D array, same shape
    rel_orient_bins  : 1D array of relative-orientation bin centers (rad)
    dHH_bins         : 1D array of head-head-distance bin centers (mm)
    """
    if filename is None:
        user_input = input(
            'Turning-histogram CSV to load '
            '(default: "turning_histogram.csv"): ').strip()
        filename = user_input if user_input else "turning_histogram.csv"
    base, ext = os.path.splitext(filename)
    std_filename = base + '_std' + ext

    df_mean = pd.read_csv(filename, index_col=0)
    df_std = pd.read_csv(std_filename, index_col=0)

    turn_2Dhist_mean = df_mean.to_numpy()
    turn_2Dhist_std = df_std.to_numpy()
    rel_orient_bins = df_mean.index.to_numpy(dtype=float)
    dHH_bins = df_mean.columns.to_numpy(dtype=float)

    print(f'Loaded turning-angle histogram from: {filename} and {std_filename}')
    return turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins


def get_turning_histogram(datasets=None, Nbins=(11, 13)):
    """
    Obtain the 2D histogram of mean turning angle binned by relative orientation
    and head-head distance, either by computing it from pair-tracking data (via
    make_2D_histogram) or by loading a previously exported CSV.

    Prompts the user to:
      (s) compute from the datasets already loaded (passed in as `datasets`),
      (p) compute from a new pair-data pickle file, or
      (c) load from a previously exported CSV.
    For (s)/(p), optionally exports the result to CSV afterward.

    Inputs
    ------
    datasets : list of dataset dictionaries already loaded, used for option (s);
               may be None if the user will choose (p) or (c).
    Nbins : (n_relorient_bins, n_dHH_bins) for the 2D histogram.

    Returns
    -------
    turn_2Dhist_mean : 2D array (n_relorient_bins x n_dHH_bins), mean turning
                       angle (rad) in each bin
    turn_2Dhist_std  : 2D array, std dev of turning angle (rad) in each bin
    rel_orient_bins  : 1D array of relative-orientation bin centers (rad)
    dHH_bins         : 1D array of head-head-distance bin centers (mm)
    """
    choice = input(
        '\nLoad turning probabilities (vs d_HH and rel. orientation) from the '
        'same pickle files (s), new pickle files (p), or load from CSV (c)? '
        ).strip().lower()

    if choice == 'c':
        return load_turning_histogram_CSV()

    if choice == 'p':
        from IO_toolkit import load_and_assign_from_pickle
        _, variable_tuple = load_and_assign_from_pickle()
        datasets = variable_tuple[0]
    elif choice == 's':
        if datasets is None:
            raise ValueError("Option (s) requires datasets to be loaded already "
                             "(load the inter-bout data from pickle, not CSV).")
    else:
        raise ValueError(f"Unrecognized choice: {choice!r}")

    turn_2Dhist_mean, rel_orient_bins, dHH_bins, _, turn_2Dhist_std \
        = make_2D_histogram(
            datasets,
            keyNames=('relative_orientation', 'head_head_distance_mm'),
            keyIdx=(None, None),
            keyNameC='turning_angle_rad', keyIdxC=None,
            colorRange=(-2.5*np.pi/180.0, 2.5*np.pi/180.0),
            dilate_minus1=False,
            bin_ranges=((-np.pi, np.pi), (0.0, 50.0)), Nbins=Nbins,
            titleStr='p(Turning Angle)',
            clabelStr='Mean Turning Angle (degrees)',
            xlabelStr='Relative Orientation (degrees)',
            ylabelStr='Head-Head Distance (mm)',
            mask_by_sem_limit=5.0*np.pi/180.0,
            unit_scaling_for_plot=[180.0/np.pi, 1.0, 180.0/np.pi],
            cmap='RdYlBu_r',
            plot_type='heatmap',
            outputFileName=None,
            closeFigure=True,
            outputCSVFileName=None)
    # turn_2Dhist_mean = 10.0*turn_2Dhist_mean
    # turn_2Dhist_std = 10.0*turn_2Dhist_std
    # print('\n\n10X turning pref!')
    # _ = input('Press Enter... ')

    # Reduce the meshgrid X, Y to 1D bin-center arrays
    rel_orient_bins = rel_orient_bins[:, 0]   # shape (Nbins[0],)
    dHH_bins = dHH_bins[0, :]                  # shape (Nbins[1],)

    export_choice = input(
        'Export this turning histogram to CSV? (y/n): ').strip().lower()
    if export_choice == 'y':
        export_turning_histogram_CSV(turn_2Dhist_mean, turn_2Dhist_std,
                                     rel_orient_bins, dHH_bins)

    return turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins


def main():
    """
    Main function for loading data and calling analysis functions.

    The user is prompted to either:
      (p) compute IBI properties from pickle files, or
      (c) load them from a previously exported CSV.
    """

    plt.ion()              # interactive mode → all plt.show() calls are non-blocking

    choice = input(
        '\nCompute IBI properties from pickle files (p) or load from CSV (c)? '
    ).strip().lower()

    datasets = None   # set only if IBI data is loaded from pickle (option p)
    if choice == 'p':
        from IO_toolkit import load_and_assign_from_pickle
        _, variable_tuple = load_and_assign_from_pickle()
        # Follow the prompts. Then:
        (datasets, CSVcolumns, expt_config, params, N_datasets, Nfish,
         basePath, dataPath, subGroupName) = variable_tuple

        arena_radius_mm = expt_config['arena_radius_mm']

        all_results, pooled_IB_properties = get_InterBout_properties(datasets)
        export_interbout_CSV(all_results)
    else:
        all_results, pooled_IB_properties = load_interbout_CSV()
        arena_radius_mm = float(input('Arena radius (mm): ').strip())

    # Show histograms of pooled properties
    plot_interbout_histograms(pooled_IB_properties)

    # Build radial-binned empirical distributions
    print('\nBuilding radial bin distributions...')
    radial_bins, bin_edges = build_radial_bin_distributions(
        pooled_IB_properties, arena_radius_mm, bin_size_mm=1.0)
    plot_radial_bin_occupancy(radial_bins, bin_edges)

    T_total_s = 600.0
    # Single fish random walk simulation
    r_sim, gamma_sim, t_sim = \
        sim_sampled_random_walk(radial_bins, arena_radius_mm, T_total_s=T_total_s,
                                plot_positions=True, rng=None)
    print(f'Simulated time: {t_sim[-1]:.1f} s')

    # Obtain the turning-angle 2D histogram (compute from pair data or load CSV)
    turn_2Dhist_mean, turn_2Dhist_std, rel_orient_bins, dHH_bins = \
        get_turning_histogram(datasets=datasets, Nbins=(11, 13))

    # Simulate pairs, with turning biased by the other fish
    T_total_s = 7200.0
    print(f'\nSimulating pairs for {T_total_s} seconds.')
    r_sim, gamma_sim, t_sim = \
        sim_pair_interacting_walk(radial_bins, arena_radius_mm,
                              turn_2Dhist_mean, turn_2Dhist_std,
                              rel_orient_bins, dHH_bins,
                              r_init=None,
                              gamma_init=None, theta_init=None, T_total_s=T_total_s,
                              plot_positions=True, rng=None)


    # Interpolate positions over a regular time grid
    t_array_s, r_sim_interp, gamma_sim_interp, dHH_mm = \
        interpolate_pair_rsim(r_sim, gamma_sim, t_sim, dt_s = 0.04, T_total_s=T_total_s)

    # Plot a histogram of inter-fish distances
    outputFileName = f'pair_sim_dHH_{T_total_s:.1f}_s.png'
    plot_probability_distr([dHH_mm], bin_width=1.0, bin_range=[0.0, 50.0], 
                            xlabelStr='Inter-fish distance', 
                            titleStr='Probability density',
                            yScaleType = 'linear', 
                            plot_each_dataset = True, plot_sem_band = False,
                            xlim = None, ylim = None, color = 'black', 
                            outputFileName = outputFileName, closeFigure = False,
                            outputCSVFileName = None)
                            
    print('\nClose figures to end.')
    plt.ioff()             # turn blocking back on for the final hold
    plt.show()             


    return all_results, pooled_IB_properties, radial_bins, bin_edges


if __name__ == '__main__':
    main()
