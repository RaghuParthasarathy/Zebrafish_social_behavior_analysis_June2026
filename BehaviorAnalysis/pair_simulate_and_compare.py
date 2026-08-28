# -*- coding: utf-8 -*-
"""
pair_simulate_and_compare.py

A self-contained, agent-based simulation of the behaviour of a **pair** of larval
zebrafish, with a direct comparison to experimental data. It is the minimal,
publishable subset of a larger pipeline that contains abandoned and alternative 
variants -- contact Raghuveer Parthasarathy for details. 

This minimal pipeline was extracted by AI (Claude Opus 4.8) from the full 
simulation and analysis pipeline. 

The model, in brief: Each fish's trajectory is a memoryless random walk
of linear bouts separated by inter-bout intervals. 
Bout kinematics (step size Delta_s, bout duration Delta_t,
inter-bout duration) are sampled from single-fish data binned by 
radial position r and wall-alignment angle psi, optionally (default True)
overridden by pair (r, dHH) kinematic bins. The turning angle is

    Delta_theta = mu + f * (Delta_theta_0 - <Delta_theta>),

where Delta_theta_0 is an empirical (r, psi) single-fish turn, 
<Delta_theta> its bin
mean, mu the circular blend of that mean with the neighbour-directed target 
(weight w_excess(dHH)), and f = sigma(dHH,|phi|)/sigma_far the 
distance/orientation-dependent focus factor. w_excess and f are both derived 
from data; there are no free parameters.

Data. Two input datasets are loaded, each stored as TWO pickle files -- a position-
data pickle (..._positionData.pickle) and an analysis "datasets" pickle
(..._datasets.pickle), as in default_pickle_filename_sets() of
random_displacement_analysis.py. The SINGLE-FISH dataset is the backbone ((r, psi)
turn/kinematic distributions AND the "two real single fish paired" null for w_excess);
the PAIR dataset supplies w_excess's real-pair term, the focus-ratio sigma, and the
(r, dHH) kinematic bins. Set each as a (positionData, datasets) path pair below; leave
an entry None to be prompted for it.

Dependencies (assumed available): IO_toolkit, IBI_properties_utils, IBI_diagnostics,
pair_social_estimation. random_displacement_analysis.py is NOT required.

Raghuveer Parthasarathy
August 27, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from IO_toolkit import load_and_assign_from_pickle
from IBI_properties_utils import (get_InterBout_properties,
                                  build_radial_bin_distributions,
                                  build_radial_psi_bin_distributions,
                                  build_radial_dHH_bin_distributions,
                                  _good_frame_mask, _density_and_sem)
from IBI_diagnostics import phi_resolved_turn_std_vs_distance
from pair_social_estimation import (estimate_social_blend_weight_vs_distance,
                                    compute_within_condition_turn_std,
                                    build_real_paired_null_bouts)


# =====================================================================
# RUN CONFIG  (edit these)
# =====================================================================

# Each dataset is TWO pickle files: the position-data pickle (..._positionData.pickle)
# and the analysis "datasets" pickle (..._datasets.pickle). Give each as a
# (positionData, datasets) pair. Leave an entry None to be prompted for it (text input).
SINGLE_FISH_PICKLE = (None, None)   # single-fish (Nfish==1): (positionData, datasets)
PAIR_PICKLE        = (None, None)   # pair (Nfish==2):        (positionData, datasets)

# arena_radius_mm is read from the single-fish datasets pickle's expt_config (below).
Ntrials         = 40          # independent pair simulations
T_total_s       = 600.0       # duration of each simulation (s)
social_min_delta_s = 1.0      # min bout size (mm) for the pair social measurements

# Bout quality control.
# NOTE: fps is hard-coded here; change it if your data use a different frame rate.
fps = 25.0
max_bout_speed_mm_s = 100.0   # reject tracking-error jumps faster than this (mm/s)

# Outer-wall handling: 'reflection' (default), 'sliding', 'retraction', or 'reject'.
edgeMethod = 'reflection'

# [dHH-KIN] dHH-conditioned kinematics: sample (Delta_s, IB_duration, Delta_t) jointly
# from the pair (r, dHH) bins instead of the single-fish r-bins, per these flags.
condition_by_dHH_delta_s   = True
condition_by_dHH_delta_t   = True
condition_by_dHH_IB_duration = True
kinematic_resolution      = 'dHH'   # 'average' (r) | 'dHH' (r, dHH) | 'dHH_phi'
kinematic_r_bin_size_mm   = 2.0
kinematic_dHH_bin_size_mm = 2.0

# Social-turn mean target: 'full' (face the neighbour) or 'tangential' (along-wall).
social_track_target = 'full'

# Output toggles.
plot_exp_vs_sim_dHH        = True   # overlay simulated vs experimental p(dHH)
plot_exp_vs_sim_r          = True   # overlay simulated vs experimental p(r)
plot_phi_resolved_turn_std = True   # (dHH, |phi|) turn-std diagnostic figure

sim_color   = 'darkorange'          # simulation curve colour
run_label   = 'pair_sim'            # appended to output filenames
closeFigures = False                # close figures after saving (True for batch runs)

# =====================================================================
# END RUN CONFIG
# =====================================================================


def sim_pair_interacting_walk(radial_bins, arena_radius_mm, radial_psi_bins,
                              social_track_w_excess, social_focus_f_ratio,
                              kinematic_cond=None, social_track_target='full',
                              edgeMethod='reflection',
                              r_init=None, gamma_init=None, theta_init=None,
                              T_total_s=600.0, plot_positions=False, rng=None):
    """
    Simulate a random walk of a pair of zebrafish -- the final
    'turn_sampling_social_track' model only.

    Each fish pauses for a drawn inter-bout interval, then takes a bout of drawn
    duration and displacement Delta_s. The step direction is (heading - turn), where
    the turn is:
        - drawn as an empirical (r, psi_in) single-fish turn Delta_theta_0 (psi_in =
          wrap(theta - gamma) is the incoming heading relative to the wall);
        - re-centred on mu, the circular blend of the asocial (r, psi) bin mean with
          the neighbour-directed target (facing the neighbour for 'full', along-wall
          for 'tangential'), with asocial weight a_int = 1 - w_excess(dHH);
        - and its deviation from the bin mean scaled by the focus factor
          f = sigma(dHH,|phi|)/sigma_far  (ti = mu + f*(Delta_theta_0 - bin_mean)).
    Kinematics (Delta_s, IB_duration, Delta_t) come from the (r, psi) single-fish bins,
    optionally overridden per bout by the pair (r, dHH) kinematic bins (kinematic_cond).

    social_track_w_excess and social_focus_f_ratio are REQUIRED (no fallback).

    Inputs
    ------
    radial_bins : list of dicts from build_radial_bin_distributions() (single-fish).
    arena_radius_mm : arena radius (mm).
    radial_psi_bins : (r, psi) structure from build_radial_psi_bin_distributions()
        (single-fish; carries the per-bin turn mean ti_mean).
    social_track_w_excess : (dHH_centers, w_excess) arrays; the mean-steering weight.
    social_focus_f_ratio : (dHH_centers, absphi_edges, f_map, f_marginal, f_floor);
        the focus factor lookup (see _f_ratio_lookup).
    kinematic_cond : dict {"bins", "delta_s", "IB_duration", "delta_t", "resolution"}
        for the (r, dHH) kinematic override, or None.
    social_track_target : 'full' (face the neighbour) or 'tangential' (along-wall).
    edgeMethod : 'reflection' (default), 'sliding', 'retraction', or 'reject'.
    r_init, gamma_init, theta_init : optional initial state (default random).
    T_total_s : simulated duration (s). plot_positions : sanity-check plot. rng.

    Returns
    -------
    r_sim, gamma_sim, t_sim : each a list of 2 1D arrays (per fish) of the radial
        position, polar angle, and elapsed time at the start of each IBI.
    """
    if rng is None:
        rng = np.random.default_rng()
    if radial_psi_bins is None:
        raise ValueError("sim_pair_interacting_walk requires radial_psi_bins "
                         "(the (r, psi) intrinsic-turn source).")
    if social_track_w_excess is None:
        raise ValueError("sim_pair_interacting_walk requires social_track_w_excess "
                         "(the data-derived w_excess(dHH)); the minimal model has no "
                         "k_focus fallback.")
    if social_focus_f_ratio is None:
        raise ValueError("sim_pair_interacting_walk requires social_focus_f_ratio "
                         "(the data-derived focus factor f).")
    if edgeMethod.lower() not in ('sliding', 'retraction', 'reflection', 'reject'):
        raise ValueError(f"Unrecognized edgeMethod: {edgeMethod!r}. Use "
                         "'sliding', 'retraction', 'reflection', or 'reject'.")
    edgeMethod = edgeMethod.lower()

    # Initial radial positions, polar angles, headings.
    r = (arena_radius_mm * np.sqrt(rng.uniform(0.0, 1.0, size=(2,)))
         if r_init is None else r_init)
    gamma = rng.uniform(0.0, 2.0*np.pi, size=(2,)) if gamma_init is None else gamma_init
    theta = rng.uniform(0.0, 2.0*np.pi, size=(2,)) if theta_init is None else theta_init

    dh_vec = calc_dh_vec(r, gamma)                          # fish 0 -> fish 1
    relative_orientation = calc_relative_orientation(theta, dh_vec)

    r_list = [[r[0]], [r[1]]]
    gamma_list = [[gamma[0]], [gamma[1]]]
    t_list = [[0.0], [0.0]]

    n_steps = 0
    n_fallback = 0

    while min(t_list[0][-1], t_list[1][-1]) < T_total_s:

        # The fish with the lowest recent time takes the next step.
        fish_idx = int(np.argmin(np.array((t_list[0][-1], t_list[1][-1]))))
        r_this = r[fish_idx]
        gamma_this = gamma[fish_idx]
        dHH = np.sqrt(np.sum(dh_vec**2))
        x = r_this*np.cos(gamma_this)
        y = r_this*np.sin(gamma_this)

        use_kin = (kinematic_cond is not None
                   and (kinematic_cond["delta_s"]
                        or kinematic_cond["IB_duration"]
                        or kinematic_cond["delta_t"]))
        psi_in = (theta[fish_idx] - gamma_this + np.pi) % (2.0*np.pi) - np.pi

        def _draw_step():
            # (r, psi_in) wall-conditioned single-fish step (fall back to r-only bins).
            if radial_psi_bins is not None:
                s = sample_from_radial_psi_bin(radial_psi_bins, r_this, psi_in, rng=rng)
                if s is None:
                    s = sample_from_radial_bin(radial_bins, r_this, rng=rng)
            else:
                s = sample_from_radial_bin(radial_bins, r_this, rng=rng)
            # [dHH-KIN] optional joint (r, dHH) kinematic override (one bout, so the
            # conditioned quantities keep their correlations).
            k = None
            if use_kin:
                k = sample_kinematics_from_radial_dHH_bin(
                    kinematic_cond["bins"], r_this, dHH, rng=rng,
                    phi_current=relative_orientation[fish_idx],
                    resolution=kinematic_cond.get("resolution", 'dHH'))
            ds = s["Delta_s_mm"]
            if k is not None and kinematic_cond["delta_s"]:
                ds = k["Delta_s_mm"]
            # Intrinsic empirical (r, psi) turn: the displacement-direction change
            # -Delta_theta (the sim heading IS the displacement direction).
            ti = -s["Delta_theta"]
            if not np.isfinite(ti):
                ti = 0.0
            # [SOCIAL TRACK] shift the turn CENTRE toward the social target with
            # asocial weight a_int = 1 - w_excess(dHH), and scale the empirical
            # deviation from the bin mean by the focus factor f (shape-preserving).
            i_rp, j_rp = _find_radial_psi_bin_index(radial_psi_bins, r_this, psi_in)
            b_rp = radial_psi_bins["bins"][i_rp][j_rp]
            tim = b_rp.get("ti_mean", np.nan)
            phi_f = relative_orientation[fish_idx]
            if social_track_target == 'full':
                turn_soc = phi_f                      # face the neighbour
            else:
                psi_tgt = (np.pi/2.0 if np.sin(psi_in - phi_f) >= 0.0 else -np.pi/2.0)
                turn_soc = psi_in - psi_tgt           # tangential (along-wall)
            base = tim if np.isfinite(tim) else ti    # asocial turn centre
            _wd, _wv = social_track_w_excess
            w_trk = min(max(float(np.interp(dHH, _wd, _wv)), 0.0), 1.0)
            a_int = 1.0 - w_trk
            mu = np.arctan2(a_int*np.sin(base) + (1.0 - a_int)*np.sin(turn_soc),
                            a_int*np.cos(base) + (1.0 - a_int)*np.cos(turn_soc))
            f_foc = _f_ratio_lookup(social_focus_f_ratio, phi_f, dHH)
            dev = (ti - tim + np.pi) % (2.0*np.pi) - np.pi
            ti = (mu + f_foc*dev) if np.isfinite(tim) else ti
            d = theta[fish_idx] - ti
            return s, k, ds, x + ds*np.cos(d), y + ds*np.sin(d)

        if edgeMethod == 'reject':
            # Redraw the whole step until the proposed point is inside; after 50
            # failures fall back to 'sliding'.
            for _try in range(50):
                sample, kin, Delta_s, x_new, y_new = _draw_step()
                if np.hypot(x_new, y_new) <= arena_radius_mm:
                    break
            step_edge = 'sliding'
        else:
            sample, kin, Delta_s, x_new, y_new = _draw_step()
            step_edge = edgeMethod

        # [dHH-KIN] durations: single-fish unless the (r, dHH) draw overrides them.
        IB_dur = sample["IB_duration_s"]
        Delta_t = sample["Delta_t_s"]
        if kin is not None:
            if kinematic_cond["IB_duration"]:
                IB_dur = kin["IB_duration_s"]
            if kinematic_cond["delta_t"]:
                Delta_t = kin["Delta_t_s"]
        t_this = t_list[fish_idx][-1] + IB_dur + Delta_t

        r_prop = np.hypot(x_new, y_new)
        gamma_prop = np.arctan2(y_new, x_new)
        if r_prop > arena_radius_mm:
            n_fallback += 1
        wall_slide = (r_prop > arena_radius_mm) and (step_edge == 'sliding')
        r_new, gamma_new = impose_radial_boundary(
            r_prop, arena_radius_mm, gamma_prop, gamma_prev=gamma_this,
            r_prev=r_this, edgeMethod=step_edge)

        # ---- commit position, heading, neighbour state ----
        n_steps += 1
        r[fish_idx] = r_new
        gamma[fish_idx] = (gamma_new + np.pi) % (2.0*np.pi) - np.pi
        # Heading: wall tangent after a slide, else the actual displacement direction.
        tang_sign = 0.0
        if wall_slide:
            tang_sign = np.sign((gamma[fish_idx] - gamma_this + np.pi)
                                % (2.0*np.pi) - np.pi)
        if tang_sign != 0.0:
            theta[fish_idx] = ((gamma[fish_idx] + tang_sign*0.5*np.pi + np.pi)
                               % (2.0*np.pi) - np.pi)
        else:
            x_final = r_new*np.cos(gamma[fish_idx])
            y_final = r_new*np.sin(gamma[fish_idx])
            dx_actual = x_final - x
            dy_actual = y_final - y
            if dx_actual != 0.0 or dy_actual != 0.0:
                theta[fish_idx] = np.arctan2(dy_actual, dx_actual)
            # else: zero-length step, leave heading unchanged.

        dh_vec = calc_dh_vec(r, gamma)
        relative_orientation = calc_relative_orientation(theta, dh_vec)
        r_list[fish_idx].append(r[fish_idx])
        gamma_list[fish_idx].append(gamma[fish_idx])
        t_list[fish_idx].append(t_this)

    r_sim = [np.array(r_list[0]), np.array(r_list[1])]
    gamma_sim = [np.array(gamma_list[0]), np.array(gamma_list[1])]
    t_sim = [np.array(t_list[0]), np.array(t_list[1])]

    pct = 100.0*n_fallback/n_steps if n_steps > 0 else 0.0
    print(f'  sim_pair_interacting_walk: wall overshoot (edge handling) on '
          f'{n_fallback} / {n_steps} steps ({pct:.2f}%).')
    if pct > 5.0:
        print('    NOTE: wall-contact rate > 5%; the empirical Delta_s distribution '
              'may have a large-step tail relative to the arena size.')

    if plot_positions:
        colors = ['steelblue', 'firebrick']
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(121)
        for k in range(2):
            x_k = r_sim[k] * np.cos(gamma_sim[k])
            y_k = r_sim[k] * np.sin(gamma_sim[k])
            ax1.scatter(x_k, y_k, s=20, alpha=0.3, color=colors[k],
                        edgecolors='none', label=f'Fish {k}')
        ph = np.linspace(0.0, 2.0*np.pi, 300)
        ax1.plot(arena_radius_mm*np.cos(ph), arena_radius_mm*np.sin(ph),
                 'k-', linewidth=1.5)
        ax1.set_xlabel('x (mm)', fontsize=12)
        ax1.set_ylabel('y (mm)', fontsize=12)
        ax1.set_title(f'Simulated pair positions (N steps = {len(r_sim[0])}, '
                      f'{len(r_sim[1])})', fontsize=12)
        ax1.set_aspect('equal')
        ax1.legend(fontsize=10)
        num_bins = 90
        bins = np.linspace(-np.pi, np.pi, num_bins + 1)
        widths = 2*np.pi/num_bins
        bin_centers = bins[:-1] + widths/2
        ax2 = fig.add_subplot(122, projection='polar')
        for k in range(2):
            counts, _ = np.histogram(gamma_sim[k], bins=bins)
            ax2.bar(bin_centers, counts, width=widths, bottom=0.0,
                    edgecolor=colors[k], color=colors[k], alpha=0.4, label=f'Fish {k}')
        ax2.set_title("Polar Angle Histogram", va='bottom', fontsize=14)
        ax2.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.15, 1.1))
        plt.tight_layout()
        plt.show(block=False)

    return r_sim, gamma_sim, t_sim


def simulate_pair_dHH_trials(radial_bins, arena_radius_mm, radial_psi_bins,
                             social_track_w_excess, social_focus_f_ratio,
                             kinematic_cond=None, social_track_target='full',
                             edgeMethod='reflection',
                             Ntrials=20, T_total_s=600.0, dt_s=0.04,
                             interp_method='nearest',
                             plot_first_positions=True, rng=None):
    """
    Run Ntrials independent pair simulations and return the inter-fish-distance (dHH)
    and pooled radial-position (r) time series, one per trial (each dHH series
    interpolated onto a common regular time grid). Suitable for
    plot_experimental_vs_sim_dHH / plot_experimental_vs_sim_r.

    Returns
    -------
    dHH_list : list of Ntrials 1D arrays of inter-fish distance (mm).
    r_list   : list of Ntrials 1D arrays of pooled per-IBI radial position (mm).
    """
    if rng is None:
        rng = np.random.default_rng()
    dHH_list = []
    r_list = []
    for trial in range(Ntrials):
        print(f'  Pair simulation trial {trial + 1} / {Ntrials} ...')
        plot_positions = plot_first_positions and (trial == 0)
        r_sim, gamma_sim, t_sim = sim_pair_interacting_walk(
            radial_bins, arena_radius_mm, radial_psi_bins,
            social_track_w_excess, social_focus_f_ratio,
            kinematic_cond=kinematic_cond, social_track_target=social_track_target,
            edgeMethod=edgeMethod, r_init=None, gamma_init=None, theta_init=None,
            T_total_s=T_total_s, plot_positions=plot_positions, rng=rng)
        _, r_interp, gamma_interp, dHH_mm = interpolate_pair_rsim(
            r_sim, gamma_sim, t_sim, dt_s=dt_s, T_total_s=T_total_s,
            interp_method=interp_method)
        dHH_list.append(dHH_mm)
        r_list.append(np.concatenate([r_sim[0], r_sim[1]]))
    return dHH_list, r_list


# =====================================================================
# VERBATIM HELPERS copied from random_displacement_analysis.py
# (the simulation-support samplers, boundary handling, small geometry
#  helpers, the experimental pooling, and the two comparison plots).
# =====================================================================
def _pool_experimental_dHH(datasets):
    """
    Pool the frame-level "head_head_distance_mm" (inter-fish distance) across the
    given pair datasets into a single 1D array (finite values only, bad-tracking
    frames excluded). Returns an empty array if no dataset carries it (e.g. single-
    fish data).
    """
    vals = _pool_experimental_dHH_by_dataset(datasets)
    return np.concatenate(vals) if vals else np.array([])


def _pool_experimental_dHH_by_dataset(datasets):
    """
    Like _pool_experimental_dHH, but return a LIST of per-dataset 1D arrays
    (one finite, good-frame array of "head_head_distance_mm" per pair dataset)
    rather than a single concatenated array. Used for the across-dataset s.e.m.
    band in plot_experimental_vs_sim_dHH. Datasets lacking the field (e.g. single-
    fish data) are skipped, so the list may be shorter than len(datasets).
    """
    vals = []
    for ds in datasets:
        d = np.asarray(ds.get("head_head_distance_mm", []), dtype=float)
        if d.size == 0:
            continue
        mask = _good_frame_mask(ds, d.shape[0])
        if mask is not None:
            d = d[mask]
        d = d.ravel()
        d = d[np.isfinite(d)]
        if d.size:
            vals.append(d)
    return vals


def _pool_experimental_r(datasets):
    """
    Pool the frame-level "radial_position_mm" (radial position of every fish)
    across the given datasets into a single 1D array (finite values only, bad-
    tracking frames excluded -- those mis-detections can place a fish well outside
    the arena and otherwise leak into the p(r) overlay). Returns an empty array if
    no dataset carries it.
    """
    vals = _pool_experimental_r_by_dataset(datasets)
    return np.concatenate(vals) if vals else np.array([])


def _pool_experimental_r_by_dataset(datasets):
    """
    Like _pool_experimental_r, but return a LIST of per-dataset 1D radial-position
    arrays (one finite, good-frame array of "radial_position_mm" per dataset) rather
    than a single concatenated array. Used for the across-dataset s.e.m. band in
    plot_experimental_vs_sim_r. Datasets lacking the field are skipped.
    """
    vals = []
    for ds in datasets:
        d = np.asarray(ds.get("radial_position_mm", []), dtype=float)
        if d.size == 0:
            continue
        mask = _good_frame_mask(ds, d.shape[0])   # rows = frames
        if mask is not None:
            d = d[mask]
        d = d.ravel()
        d = d[np.isfinite(d)]
        if d.size:
            vals.append(d)
    return vals


def _areal_density_and_sem(arrays, edges, centers, bin_width):
    """
    Areal (1/r-normalized) analogue of _density_and_sem for radial p(r): each
    replicate's histogram is divided by the bin-center r and scaled to unit area,
    then the mean (over the POOLED samples) and the across-replicate s.e.m.
    (std(per-replicate areal densities) / sqrt(Nrep)) are returned. Returns
    (pooled_density, None) if fewer than 2 non-empty replicates are available.
    """
    def _areal(x):
        h, _ = np.histogram(x, bins=edges)
        d = h / centers
        area = np.sum(d) * bin_width
        return d / area if area > 0 else d

    per_rep = []
    for a in arrays:
        a = np.asarray(a, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        per_rep.append(_areal(a))
    pooled = np.concatenate([np.asarray(a, dtype=float).ravel()
                             for a in arrays]) if len(arrays) else np.array([])
    pooled = pooled[np.isfinite(pooled)]
    pooled_density = _areal(pooled) if pooled.size else np.zeros_like(centers)
    if len(per_rep) >= 2:
        stack = np.vstack(per_rep)
        sem = np.std(stack, axis=0) / np.sqrt(stack.shape[0])
    else:
        sem = None
    return pooled_density, sem


def _find_radial_bin_index(radial_bins, r_current):
    """
    Return the index of the radial bin whose edges bracket r_current. If
    r_current is beyond the last edge, the outermost bin is used; if the bracketing
    bin is empty (N == 0), the nearest non-empty bin (searched outward then inward)
    is returned.
    """
    bin_i = None
    for i, b in enumerate(radial_bins):
        lo, hi = b["r_edges"]
        if lo <= r_current < hi:
            bin_i = i
            break
    if bin_i is None:
        # r_current is beyond the last edge — use the outermost bin
        bin_i = len(radial_bins) - 1

    # If the chosen bin is empty, walk outward then inward to the nearest non-empty
    if radial_bins[bin_i]["N"] == 0:
        n_bins = len(radial_bins)
        for offset in range(1, n_bins):
            for candidate in [bin_i - offset, bin_i + offset]:
                if 0 <= candidate < n_bins and radial_bins[candidate]["N"] > 0:
                    return candidate
    return bin_i


def _find_radial_dHH_bin_index(radial_dHH_bins, r_current, dHH_current):
    """
    Return (i_r, j_dHH) of the (r, dHH) bin bracketing (r_current, dHH_current)
    in a radial_dHH_bins structure (from build_radial_dHH_bin_distributions).

    If that bin is empty, fall back: first to the nearest non-empty dHH bin at
    the SAME radius (preserving the radial / edge structure, i.e. marginalising
    over the social axis), then to the nearest radial row (inward then outward)
    and the nearest non-empty dHH bin within it.
    """
    bins = radial_dHH_bins["bins"]
    r_edges = radial_dHH_bins["r_edges"]
    dHH_edges = radial_dHH_bins["dHH_edges"]
    n_r = len(bins)
    n_d = len(bins[0])

    i_r = int(np.clip(np.digitize(r_current, r_edges) - 1, 0, n_r - 1))
    j_d = int(np.clip(np.digitize(dHH_current, dHH_edges) - 1, 0, n_d - 1))
    if bins[i_r][j_d]["N"] > 0:
        return i_r, j_d

    # 1) nearest non-empty dHH bin at the same radius
    for off in range(1, n_d):
        for cand in (j_d - off, j_d + off):
            if 0 <= cand < n_d and bins[i_r][cand]["N"] > 0:
                return i_r, cand

    # 2) nearest radial row (inward first, then outward); nearest dHH within it
    for off in range(1, n_r):
        for ci in (i_r - off, i_r + off):
            if 0 <= ci < n_r:
                if bins[ci][j_d]["N"] > 0:
                    return ci, j_d
                for off2 in range(1, n_d):
                    for cand in (j_d - off2, j_d + off2):
                        if 0 <= cand < n_d and bins[ci][cand]["N"] > 0:
                            return ci, cand

    return i_r, j_d   # entire grid empty (no data); caller handles N == 0


def sample_from_radial_bin(radial_bins, r_current, rng=None):
    """
    Given the current radial position, find the corresponding radial bin and
    draw one random tuple (Delta_r_mm, Delta_gamma, Delta_t_s, IB_duration_s,
    Delta_s_mm, Delta_theta, turning_angle_IBI) from the empirical observations in
    that bin. All values come from the SAME observed IBI (one index), so their
    joint distribution / correlations are preserved (used e.g. by
    'turn_sampling_additive', which needs (Delta_s_mm, turning_angle_IBI) jointly).
    If the bin is empty (no observations), the nearest non-empty bin is used.

    Inputs
    ------
    radial_bins : list of dicts returned by build_radial_bin_distributions()
    r_current : float, current radial position (mm)
    rng : numpy.random.Generator or None

    Returns
    -------
    sample : dict with keys Delta_r_mm, Delta_gamma, Delta_t_s, IB_duration_s,
             Delta_s_mm, Delta_theta, turning_angle_IBI
    """
    if rng is None:
        rng = np.random.default_rng()

    bin_i = _find_radial_bin_index(radial_bins, r_current)
    b = radial_bins[bin_i]
    idx = rng.integers(0, b["N"])
    return {
        "Delta_r_mm":    float(b["Delta_r_mm"][idx]),
        "Delta_gamma":   float(b["Delta_gamma"][idx]),
        "Delta_t_s":     float(b["Delta_t_s"][idx]),
        "IB_duration_s": float(b["IB_duration_s"][idx]),
        "Delta_s_mm": float(b["Delta_s_mm"][idx]),
        "Delta_theta": float(b["Delta_theta"][idx]),
        "turning_angle_IBI": float(b["turning_angle_IBI"][idx]),
    }


def sample_kinematics_from_radial_dHH_bin(radial_dHH_bins, r_current,
                                          dHH_current, rng=None,
                                          phi_current=None, resolution='dHH',
                                          min_phi_N=25):
    """
    [dHH-KIN] Jointly draw (Delta_s_mm, IB_duration_s, Delta_t_s) from the pair
    kinematic bins (build_radial_dHH_bin_distributions). All three come from the
    SAME observed pair IBI (one index), so their within-bout correlations are kept
    (critical: Delta_s-Delta_t corr ~0.65 drives the joint conditioning effect).
    This is the neighbour-conditioned kinematic source for the additive-family and
    social_focus/track methods: real fish modulate step size, pause and bout
    duration by the neighbour, which the single-fish r-bins cannot carry.

    resolution (level of detail):
      'average'  -> the (r)-only marginal (bins_r): pair kinematics with dHH AND
                    |phi| averaged out.
      'dHH'      -> the (r, dHH) bin (default; step statistics vs inter-fish
                    distance). Empty (r, dHH) bins fall back to the nearest non-empty
                    bin at the same radius via _find_radial_dHH_bin_index.
      'dHH_phi'  -> the (r, dHH, |phi|) cell (also resolved by neighbour BEARING;
                    |phi| in folded 45-deg bins), with a (r, dHH) phi-marginal
                    fallback where that cell has < min_phi_N steps (e.g. the sparse
                    forward-far corner). Needs phi_current (rad, relative orientation).

    Returns dict with Delta_s_mm, IB_duration_s, Delta_t_s, or None if no data.
    """
    if rng is None:
        rng = np.random.default_rng()

    def _draw(cell):
        if cell is None or cell["N"] == 0:
            return None
        idx = rng.integers(0, cell["N"])
        return {"Delta_s_mm":    float(cell["Delta_s_mm"][idx]),
                "IB_duration_s": float(cell["IB_duration_s"][idx]),
                "Delta_t_s":     float(cell["Delta_t_s"][idx])}

    if resolution == 'average' and "bins_r" in radial_dHH_bins:
        i_r = _find_radial_bin_index(radial_dHH_bins["bins_r"], r_current)
        return _draw(radial_dHH_bins["bins_r"][i_r])

    i_r, j_d = _find_radial_dHH_bin_index(radial_dHH_bins, r_current, dHH_current)
    if (resolution == 'dHH_phi' and phi_current is not None
            and "bins_phi" in radial_dHH_bins):
        pe = radial_dHH_bins["phi_edges"]
        absphi = abs((phi_current + np.pi) % (2.0*np.pi) - np.pi)
        k_p = int(np.clip(np.digitize(absphi, pe) - 1, 0,
                          radial_dHH_bins["n_phi_bins"] - 1))
        cell = radial_dHH_bins["bins_phi"][i_r][j_d][k_p]
        if cell["N"] >= min_phi_N:
            return _draw(cell)
        # else fall through to the (r, dHH) phi-marginal
    return _draw(radial_dHH_bins["bins"][i_r][j_d])


def _find_radial_psi_bin_index(radial_psi_bins, r_current, psi_current):
    """
    Return (i_r, j_psi) of the (r, psi) bin bracketing (r_current, psi_current),
    with psi periodic on [-pi, pi]. If that bin is empty, fall back to the nearest
    non-empty psi bin at the SAME radius (PERIODIC distance, marginalising over
    wall orientation), then to the nearest radial row (inward first, then outward)
    and the nearest psi within it.
    """
    bins = radial_psi_bins["bins"]
    r_edges = radial_psi_bins["r_edges"]
    psi_edges = radial_psi_bins["psi_edges"]
    n_r = len(bins)
    n_psi = len(bins[0])

    psi_w = (psi_current + np.pi) % (2.0*np.pi) - np.pi
    i_r = int(np.clip(np.digitize(r_current, r_edges) - 1, 0, n_r - 1))
    j_p = int(np.clip(np.digitize(psi_w, psi_edges) - 1, 0, n_psi - 1))
    if bins[i_r][j_p]["N"] > 0:
        return i_r, j_p

    # 1) nearest non-empty psi bin at the same radius (periodic in psi)
    for off in range(1, n_psi // 2 + 1):
        for cand in ((j_p - off) % n_psi, (j_p + off) % n_psi):
            if bins[i_r][cand]["N"] > 0:
                return i_r, cand

    # 2) nearest radial row (inward first, then outward); nearest psi within it
    for off in range(1, n_r):
        for ci in (i_r - off, i_r + off):
            if 0 <= ci < n_r:
                if bins[ci][j_p]["N"] > 0:
                    return ci, j_p
                for off2 in range(1, n_psi // 2 + 1):
                    for cand in ((j_p - off2) % n_psi, (j_p + off2) % n_psi):
                        if bins[ci][cand]["N"] > 0:
                            return ci, cand

    return i_r, j_p   # entire grid empty (no data); caller handles N == 0


def sample_from_radial_psi_bin(radial_psi_bins, r_current, psi_current, rng=None):
    """
    Draw one joint step tuple (Delta_r_mm, Delta_gamma, Delta_t_s, IB_duration_s,
    Delta_s_mm, Delta_theta, turning_angle_IBI) from the (r, psi) bin bracketing
    (r_current, psi_current) -- the wall-orientation-conditioned analog of
    sample_from_radial_bin. All values come from the SAME observed IBI (one index),
    preserving the joint distribution. Empty bins fall back via
    _find_radial_psi_bin_index; returns None only if the entire grid is empty.

    Inputs
    ------
    radial_psi_bins : structure from build_radial_psi_bin_distributions()
    r_current : float, current radial position (mm)
    psi_current : float, current incoming wall orientation = wrap(theta - gamma)
    rng : numpy.random.Generator or None

    Returns
    -------
    dict with the seven step keys, or None if no data anywhere.
    """
    if rng is None:
        rng = np.random.default_rng()
    i_r, j_p = _find_radial_psi_bin_index(radial_psi_bins, r_current, psi_current)
    b = radial_psi_bins["bins"][i_r][j_p]
    if b["N"] == 0:
        return None
    idx = rng.integers(0, b["N"])
    return {
        "Delta_r_mm":    float(b["Delta_r_mm"][idx]),
        "Delta_gamma":   float(b["Delta_gamma"][idx]),
        "Delta_t_s":     float(b["Delta_t_s"][idx]),
        "IB_duration_s": float(b["IB_duration_s"][idx]),
        "Delta_s_mm":    float(b["Delta_s_mm"][idx]),
        "Delta_theta":   float(b["Delta_theta"][idx]),
        "turning_angle_IBI": float(b["turning_angle_IBI"][idx]),
    }


def impose_radial_boundary(r, arena_radius_mm, gamma=None, gamma_prev=None,
                           r_prev=None, edgeMethod='sliding'):
    """
    Impose radial boundary conditions (0 <= r <= arena_radius).

    Two boundary behaviours:
      - r < 0 : reflect through the origin (r -> -r, gamma -> gamma + pi). This
                is the polar-coordinate singularity, NOT a wall, and is handled
                the same way for every edgeMethod.
      - r > R : use "edgeMethod" to handle reaching the outer wall (radius R).

    edgeMethod options (all impose only the boundary; none adds wall attraction):
       'sliding' : "slide along the wall". The radial overshoot (r - R) is
                converted to an arc length along the wall, advancing gamma by
                (r - R)/R (= r/R - 1) in the direction the fish is travelling
                TANGENTIALLY, i.e. sign(wrap(gamma - gamma_prev)) -- the sign of
                the change in polar angle this step. r is set to R. This changes
                the step's heading to be parallel to the wall.
       'retraction' : reflect the radial coordinate about the wall, r -> 2R - r,
                keeping gamma unchanged. (An earlier method. Note this is NOT a
                geometric reflection off the circular wall -- a true reflection
                would also change gamma -- so "retraction" is the apter name.)
                Needs neither gamma_prev nor r_prev.
       'reflection' : true specular reflection of the straight-line step off the
                circular wall, changing both r and gamma (and hence the step's
                heading). The chord from the previous point (r_prev, gamma_prev)
                to the proposed point (r, gamma) is intersected with the circle;
                the remnant of the step beyond the wall has its wall-normal
                (radial) component reversed and its tangential component kept.
                Requires r_prev, gamma_prev and gamma; if any is None it falls
                back to clamping r at R.

    The 'sliding' tangential direction needs the previous polar angle
    (gamma_prev). Using sign(gamma) instead (the fish's absolute angular
    position) would make every wall hit drift gamma toward +-pi regardless of
    the motion, breaking the arena's rotational symmetry and piling fish up at
    gamma = +-pi. If gamma_prev is None, no tangential slide is applied (r is
    clamped to R, gamma unchanged); a head-on radial hit
    (wrap(gamma - gamma_prev) == 0) likewise does not slide.

    Inputs
    ------
    r : float, proposed radial position (mm)
    arena_radius_mm : float, radius of the arena (mm)
    gamma : float or None, proposed polar angle (rad)
    gamma_prev : float or None, polar angle at the PREVIOUS point; sets the
                 wall-slide direction ('sliding') or the chord start ('reflection')
    r_prev : float or None, radial position at the PREVIOUS point; needed only by
             'reflection' to reconstruct the straight-line step
    edgeMethod : 'sliding' | 'retraction' | 'reflection'; method used to handle
                 reaching the outer wall

    Returns
    -------
    r : float, radial position (mm), in [0, arena_radius_mm]
    gamma : float, polar angle (rad), wrapped to [-pi, pi] (None if gamma is None)
    """

    valid_methods = ('sliding', 'retraction', 'reflection')
    if edgeMethod.lower() not in valid_methods:
        raise ValueError(f"Unrecognized edgeMethod: {edgeMethod!r}. Use one of "
                         f"{valid_methods}.")
    method = edgeMethod.lower()

    # Reflect through origin if r goes negative (polar singularity; all methods).
    if r < 0.0:
        r = -r
        if gamma is not None:
            gamma = gamma + np.pi

    if r > arena_radius_mm:
        if method == 'sliding':
            # Slide along arena wall if r exceeds arena radius
            if gamma is not None and gamma_prev is not None:
                # Tangential direction of travel (CCW > 0 / CW < 0); compute the slide
                # before r is updated. sign == 0 (head-on radial hit) => no slide.
                tang = (gamma - gamma_prev + np.pi) % (2.0*np.pi) - np.pi
                gamma = gamma + np.sign(tang)*(r/arena_radius_mm - 1.0)
            r = arena_radius_mm
        elif method == 'retraction':
            # Reflect the radial coordinate about the wall; keep gamma. A large
            # overshoot (r > 2R) would send r < 0; the clamp below catches that.
            r = 2.0*arena_radius_mm - r
        elif method == 'reflection':
            r, gamma = _specular_reflect_circle(r, gamma, r_prev, gamma_prev,
                                                arena_radius_mm)

    # Clamp to [0, arena_radius_mm] in case of extreme overshooting
    r = float(np.clip(r, 0.0, arena_radius_mm))
    # Wrap gamma to [-pi, pi]
    if gamma is not None:
        gamma = (gamma + np.pi) % (2.0*np.pi) - np.pi

    return r, gamma


def _specular_reflect_circle(r_new, gamma_new, r_prev, gamma_prev,
                             arena_radius_mm, max_reflections=5):
    """
    Specular reflection of a straight-line step off the inside of a circular wall.

    The step is the chord from the previous (inside) point A = (r_prev, gamma_prev),
    |A| <= R, to the proposed (outside) point B = (r_new, gamma_new), |B| > R.
    The chord is intersected with the circle of radius R at Q; the remnant of the
    step beyond Q (w = B - Q) is reflected across the wall by reversing its
    wall-normal (radial) component while preserving its tangential component:
        w_reflected = w - 2 (w . n) n,   n = Q / R  (outward unit normal).
    The reflected endpoint is Q + w_reflected. If that endpoint is still outside
    (a long step grazing the wall), the reflection is repeated, up to
    max_reflections, after which any residual overshoot is clamped to the wall.

    Inputs
    ------
    r_new, gamma_new : float; proposed (outside) polar position (rad for gamma)
    r_prev, gamma_prev : float or None; previous (inside) polar position
    arena_radius_mm : float, wall radius R (mm)
    max_reflections : int, cap on successive reflections for a single step

    Returns
    -------
    r : float, reflected radial position (mm), <= R
    gamma : float, reflected polar angle (rad), unwrapped (caller wraps)
    """
    R = arena_radius_mm
    # Need the full previous point and a proposed angle to reconstruct the chord.
    if r_prev is None or gamma_prev is None or gamma_new is None:
        return min(r_new, R), gamma_new

    A = np.array([r_prev*np.cos(gamma_prev), r_prev*np.sin(gamma_prev)])
    B = np.array([r_new*np.cos(gamma_new),  r_new*np.sin(gamma_new)])

    for _ in range(max_reflections):
        if B.dot(B) <= R*R:
            break                       # endpoint is inside the arena
        AB = B - A
        a = AB.dot(AB)
        if a == 0.0:                    # degenerate (no displacement)
            B = A.copy()
            break
        b = 2.0*A.dot(AB)
        c = A.dot(A) - R*R              # <= 0 since |A| <= R
        disc = b*b - 4.0*a*c
        if disc < 0.0:                  # numerical guard; shouldn't occur
            B = B*(R/np.hypot(*B))
            break
        # Forward intersection of the chord with the circle, t in (0, 1].
        t = (-b + np.sqrt(disc))/(2.0*a)
        t = min(max(t, 0.0), 1.0)
        Q = A + t*AB                    # point on the wall, |Q| = R
        n = Q/R                         # outward unit normal
        w = B - Q                       # remnant of the step beyond the wall
        w_ref = w - 2.0*w.dot(n)*n      # reverse the normal component
        A = Q
        B = Q + w_ref

    rB = np.hypot(*B)
    if rB > R:                          # clamp any residual overshoot
        B = B*(R/rB)
        rB = R
    gamma_out = np.arctan2(B[1], B[0])
    return float(rB), float(gamma_out)


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


def _f_ratio_lookup(sfr, phi_val, dHH):
    """Neighbour-dependent focus factor f(dHH,|phi|) = sigma_A(dHH,|phi|)/sigma_far
    [FOCUS-RATIO], with a phi-marginal fallback and a floor (NO upper clip -- f>1 in
    close-range jockeying is kept). sfr = (dHH_centers, absphi_edges, f(|phi|,dHH),
    f_marginal, f_floor). Interpolate f along dHH WITHIN the selected |phi| row's
    populated span; outside it use the marginal f(dHH) (-> 1 far away). Returns
    max(f, f_floor); f_floor keeps f away from 0 (which would freeze the turn to the
    deterministic bin mean)."""
    dc, ae, fmap, fmarg, ffloor = sfr
    absphi = abs((phi_val + np.pi) % (2.0*np.pi) - np.pi)
    ia = int(min(max(np.digitize(absphi, ae) - 1, 0), fmap.shape[0] - 1))
    row = fmap[ia]; fin = np.isfinite(row)
    if np.any(fin) and dc[fin][0] <= dHH <= dc[fin][-1]:
        f = float(np.interp(dHH, dc[fin], row[fin]))
    else:
        mf = np.isfinite(fmarg)
        f = float(np.interp(dHH, dc[mf], fmarg[mf])) if np.any(mf) else 1.0
    return max(f, ffloor)


def interpolate_pair_rsim(r_sim, gamma_sim, t_sim, dt_s = 0.04, T_total_s=600.0,
                          interp_method='nearest'):
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
    interp_method : 'nearest' (default) or 'linear'.
            'nearest' holds each fish at its current IBI position and jumps at the
            next bout -- a piecewise-CONSTANT (step) trajectory, so dHH is frozen
            between bouts.
            'linear' moves the fish along a straight LAB-FRAME line between
            consecutive IBI positions (continuous motion during the displacement),
            matching the model's own bout geometry and removing the frozen-vs-
            continuous mismatch with the frame-continuous experimental trajectories.
            Interpolation is done in CARTESIAN (x, y) -- interpolating r and gamma
            separately would trace a curved path, not the straight displacement --
            then converted back to (r, gamma). Nearest snaps to the closest IBI-start
            time; linear interpolates the interval, so gamma can swing quickly when a
            path passes near the arena centre (physically correct).

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
    if interp_method not in ('nearest', 'linear'):
        raise ValueError(f"interp_method must be 'nearest' or 'linear', "
                         f"got {interp_method!r}.")

    t_array_s = np.arange(0.0, T_total_s + dt_s, dt_s, )
    r_sim_interp = np.zeros((2, len(t_array_s)))
    gamma_sim_interp = np.zeros((2, len(t_array_s)))
    # Cartesian positions at the interpolated grid (used for dHH and, for 'linear',
    # for the straight-line interpolation itself).
    x_interp = np.zeros((2, len(t_array_s)))
    y_interp = np.zeros((2, len(t_array_s)))
    for k in range(2):
        xk = r_sim[k]*np.cos(gamma_sim[k])
        yk = r_sim[k]*np.sin(gamma_sim[k])
        if interp_method == 'linear':
            # Straight-line (Cartesian) motion between consecutive IBI positions.
            x_interp[k] = np.interp(t_array_s, t_sim[k], xk)
            y_interp[k] = np.interp(t_array_s, t_sim[k], yk)
        else:
            # Nearest IBI-start time -> piecewise-constant (step) trajectory.
            # searchsorted (t_sim is increasing) -> O(N log M), O(N) memory.
            tk = np.asarray(t_sim[k], dtype=float)
            hi = np.clip(np.searchsorted(tk, t_array_s), 1, len(tk) - 1)
            lo = hi - 1
            idx = np.where(t_array_s - tk[lo] <= tk[hi] - t_array_s, lo, hi)
            x_interp[k] = xk[idx]
            y_interp[k] = yk[idx]
        r_sim_interp[k] = np.hypot(x_interp[k], y_interp[k])
        gamma_sim_interp[k] = np.arctan2(y_interp[k], x_interp[k])

    # inter-fish distance
    dHH_mm = np.hypot(x_interp[0] - x_interp[1], y_interp[0] - y_interp[1])

    return t_array_s, r_sim_interp, gamma_sim_interp, dHH_mm


def plot_experimental_vs_sim_dHH(exp_dHH, dHH_list, social_method='',
                                 bin_width_mm=1.0, dHH_max_mm=50.0,
                                 exp_dHH_list=None,
                                 exp_color='black', sim_color='darkorange',
                                 outputFileName='compare_dHH_exp_vs_sim.png',
                                 closeFigure=False):
    """
    Overlay the SIMULATED inter-fish-distance (dHH) distribution (pooled across
    trials) on the EXPERIMENTAL one, as normalized densities, for a direct visual
    comparison of how well the pair simulation reproduces the real separation.
    Each curve carries a semi-transparent +/- s.e.m. band: for the simulation the
    s.e.m. is taken across trials (dHH_list); for the experiment it is taken across
    pair datasets when exp_dHH_list is supplied (else across-dataset spread is
    unavailable and no experimental band is drawn).

    The experimental distribution is the pooled frame-level inter-fish distance
    passed in as exp_dHH (e.g. from _pool_experimental_dHH on the pair minuend
    datasets). The simulated distribution is pooled from dHH_list (e.g. the first
    return of simulate_pair_dHH_trials). Also prints summary statistics (mean,
    median, P(dHH < 10 mm)) for both.

    Inputs
    ------
    exp_dHH : 1D array of experimental frame-level inter-fish distance (mm).
    dHH_list : list of 1D arrays of simulated inter-fish distance (mm), one per
        trial (the across-trial spread gives the simulated s.e.m. band).
    social_method : label for the legend / title (e.g. the social_method used).
    bin_width_mm : histogram bin width (mm).
    dHH_max_mm : upper edge of the histogram (mm).
    exp_dHH_list : optional list of per-dataset 1D experimental dHH arrays (e.g.
        from _pool_experimental_dHH_by_dataset). When given (>=2 datasets), the
        experimental s.e.m. band is the across-dataset standard error. None (or a
        single dataset) -> no experimental band.
    exp_color : color of the experimental curve/band (default 'black').
    sim_color : color of the simulated curve/band (default 'darkorange').
    outputFileName : figure filename; None to skip saving.
    closeFigure : if True, close the figure after creating it.

    Returns
    -------
    centers, exp_density, exp_density_sem, sim_density, sim_density_sem : 
        1D arrays (bin centers, the two normalized histograms, and the standard
        error of the mean for each histogram), 
        or (None, None, None, None, None) if no experimental dHH is
        available.
    """
    exp_dHH = np.asarray(exp_dHH, dtype=float).ravel()
    exp_dHH = exp_dHH[np.isfinite(exp_dHH)]
    if exp_dHH.size == 0:
        print('\nplot_experimental_vs_sim_dHH: empty experimental dHH; '
              'skipping overlay.')
        return None, None, None, None, None

    sim_dHH = np.concatenate([np.asarray(a, dtype=float).ravel()
                              for a in dHH_list]) if len(dHH_list) else np.array([])
    sim_dHH = sim_dHH[np.isfinite(sim_dHH)]

    edges = np.arange(0.0, dHH_max_mm + bin_width_mm, bin_width_mm)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Densities + across-replicate s.e.m.: experiment across datasets (if the
    # per-dataset list is supplied), simulation across trials (dHH_list).
    if exp_dHH_list:
        exp_density, exp_sem = _density_and_sem(exp_dHH_list, edges)
    else:
        exp_density, _ = np.histogram(exp_dHH, bins=edges, density=True)
        exp_sem = None
    sim_density, sim_sem = _density_and_sem(list(dHH_list), edges)

    def _summary(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float('nan'), float('nan'), float('nan')
        return float(np.mean(x)), float(np.median(x)), float(np.mean(x < 10.0))

    em, emed, ep = _summary(exp_dHH)
    sm, smed, sp = _summary(sim_dHH)
    print('\n--- Experimental vs simulated inter-fish distance (mm) ---')
    print(f'  experimental: mean={em:5.1f}  median={emed:5.1f}  P(<10mm)={ep:.3f}')
    print(f'  simulated:    mean={sm:5.1f}  median={smed:5.1f}  P(<10mm)={sp:.3f}')

    fig = plt.figure(figsize=(9, 5))
    alpha_sem = 0.3
    plt.plot(centers, exp_density, '-', color=exp_color, lw=2, label='Experimental')
    if exp_sem is not None:
        plt.fill_between(centers, exp_density - exp_sem, exp_density + exp_sem,
                         color=exp_color, alpha=alpha_sem, linewidth=0)
    lbl = 'Simulated' # + (f' ({social_method})' if social_method else '')
    plt.plot(centers, sim_density, '-', color=sim_color, lw=2, label=lbl)
    if sim_sem is not None:
        plt.fill_between(centers, sim_density - sim_sem, sim_density + sim_sem,
                         color=sim_color, alpha=alpha_sem, linewidth=0)
    plt.ylim(bottom=0)
    plt.xlabel('Inter-fish distance dHH (mm)', fontsize=12)
    plt.ylabel('Probability density', fontsize=12)
    plt.title('Experimental vs simulated inter-fish distance', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    if outputFileName is not None:
        plt.savefig(outputFileName, dpi=130)
        print(f'  Saved overlay figure: {outputFileName}')
    plt.show(block=False)
    if closeFigure:
        plt.close(fig)

    return centers, exp_density, exp_sem, sim_density, sim_sem


def plot_experimental_vs_sim_r(exp_r, r_list, social_method='',
                               bin_width_mm=0.5, r_max_mm=None,
                               exp_r_list=None,
                               exp_color='black', sim_color='darkorange',
                               outputFileName='compare_r_exp_vs_sim.png',
                               closeFigure=False):
    """
    Overlay the SIMULATED radial-position distribution p(r) (pooled across trials)
    on the EXPERIMENTAL one, as 1/r-normalized areal densities, for a direct visual
    comparison of how well the simulation reproduces the radial occupancy (edge-
    dwelling / thigmotaxis). The radial-position analogue of
    plot_experimental_vs_sim_dHH(). Each curve carries a semi-transparent +/- s.e.m.
    band: for the simulation the s.e.m. is taken across trials (r_list); for the
    experiment it is taken across datasets when exp_r_list is supplied (else no
    experimental band is drawn).

    The 1/r (areal) normalization matters: in a uniform disk the raw radial
    histogram grows linearly with r simply because there is more area at larger r,
    so dividing each bin by its center r gives the areal density and makes edge
    preference visible. Both curves are 1/r-normalized identically here, then
    scaled to unit area, so they are directly comparable.

    The experimental distribution is the pooled frame-level radial position passed
    in as exp_r (e.g. from _pool_experimental_r on the pair minuend datasets). The
    simulated distribution is pooled from r_list (e.g. the second return of
    simulate_pair_dHH_trials, one array of pooled per-step r per trial). Also prints
    summary statistics (mean, median, fraction beyond 0.8*r_max) for both.

    Inputs
    ------
    exp_r : 1D array of experimental frame-level radial position (mm).
    r_list : list of 1D arrays of simulated radial position (mm), one per trial (the
        across-trial spread gives the simulated s.e.m. band).
    social_method : label for the legend / title (e.g. the social_method used).
    bin_width_mm : histogram bin width (mm).
    r_max_mm : upper edge of the histogram (mm); None -> derived from the data
        (max observed r, rounded up to a bin edge).
    exp_r_list : optional list of per-dataset 1D experimental r arrays (e.g. from
        _pool_experimental_r_by_dataset). When given (>=2 datasets), the experimental
        s.e.m. band is the across-dataset standard error. None -> no experimental band.
    exp_color : color of the experimental curve/band (default 'black').
    sim_color : color of the simulated curve/band (default 'darkorange').
    outputFileName : figure filename; None to skip saving.
    closeFigure : if True, close the figure after creating it.

    Returns
    -------
    centers, exp_density, exp_density_sem, sim_density, sim_density_sem : 1D arrays
        (bin centers, the two 1/r-normalized histograms, and the across-replicate
        standard error of the mean for each), or (None, None, None, None, None) if
        no experimental r is available.
    """
    exp_r = np.asarray(exp_r, dtype=float).ravel()
    exp_r = exp_r[np.isfinite(exp_r)]
    if exp_r.size == 0:
        print('\nplot_experimental_vs_sim_r: empty experimental r; '
              'skipping overlay.')
        return None, None, None, None, None

    sim_r = np.concatenate([np.asarray(a, dtype=float).ravel()
                            for a in r_list]) if len(r_list) else np.array([])
    sim_r = sim_r[np.isfinite(sim_r)]

    if r_max_mm is None:
        r_max_mm = max(float(exp_r.max()),
                       float(sim_r.max()) if sim_r.size else 0.0)
        r_max_mm = bin_width_mm * np.ceil(r_max_mm / bin_width_mm)

    edges = np.arange(0.0, r_max_mm + bin_width_mm, bin_width_mm)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Areal (1/r-normalized) densities + across-replicate s.e.m.: experiment across
    # datasets (if the per-dataset list is supplied), simulation across trials.
    if exp_r_list:
        exp_density, exp_sem = _areal_density_and_sem(
            exp_r_list, edges, centers, bin_width_mm)
    else:
        exp_density, _ = _areal_density_and_sem(
            [exp_r], edges, centers, bin_width_mm)
        exp_sem = None
    sim_density, sim_sem = _areal_density_and_sem(
        list(r_list), edges, centers, bin_width_mm)

    def _summary(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return float('nan'), float('nan'), float('nan')
        return (float(np.mean(x)), float(np.median(x)),
                float(np.mean(x > 0.8 * r_max_mm)))

    em, emed, ep = _summary(exp_r)
    sm, smed, sp = _summary(sim_r)
    print('\n--- Experimental vs simulated radial position (mm) ---')
    print(f'  experimental: mean={em:5.1f}  median={emed:5.1f}  '
          f'P(r>{0.8*r_max_mm:.0f}mm)={ep:.3f}')
    print(f'  simulated:    mean={sm:5.1f}  median={smed:5.1f}  '
          f'P(r>{0.8*r_max_mm:.0f}mm)={sp:.3f}')

    fig = plt.figure(figsize=(9, 5))
    alpha_sem = 0.3
    plt.plot(centers, exp_density, '-', color=exp_color, lw=2, label='Experimental')
    if exp_sem is not None:
        plt.fill_between(centers, exp_density - exp_sem, exp_density + exp_sem,
                         color=exp_color, alpha=alpha_sem, linewidth=0)
    lbl = 'Simulated'  # + (f' ({social_method})' if social_method else '')
    plt.plot(centers, sim_density, '-', color=sim_color, lw=2, label=lbl)
    if sim_sem is not None:
        plt.fill_between(centers, sim_density - sim_sem, sim_density + sim_sem,
                         color=sim_color, alpha=alpha_sem, linewidth=0)
    plt.ylim(bottom=0)
    plt.xlabel('radial position r (mm)', fontsize=12)
    plt.ylabel('areal probability density (1/r-normalized)', fontsize=12)
    plt.title('Experimental vs simulated radial position', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()
    if outputFileName is not None:
        plt.savefig(outputFileName, dpi=130)
        print(f'  Saved overlay figure: {outputFileName}')
    plt.show(block=False)
    if closeFigure:
        plt.close(fig)

    return centers, exp_density, exp_sem, sim_density, sim_sem



def _resolve_two_paths(paths, label):
    """
    Resolve a (positionData, datasets) pickle-path pair. paths may be a 2-tuple, a
    single value, or None; any element that is missing/None is prompted for (full
    path, text input). Both resolved paths must be existing files.
    """
    if paths is None:
        pos, data = None, None
    elif isinstance(paths, (tuple, list)):
        pos, data = (list(paths) + [None, None])[:2]
    else:
        pos, data = paths, None
    if not pos:
        pos = input(f'Enter the FULL path to the {label} POSITION-DATA pickle '
                    '(..._positionData.pickle): ').strip()
    if not data:
        data = input(f'Enter the FULL path to the {label} DATASETS pickle '
                     '(..._datasets.pickle): ').strip()
    for kind, p in (('positionData', pos), ('datasets', data)):
        if not p or not Path(p).is_file():
            raise FileNotFoundError(
                f'The {label} {kind} pickle path is not a file: {p!r}')
    return pos, data


def main():
    """
    Build the final pair model from two datasets (single-fish + pair, each a
    positionData + datasets pickle pair) and compare its p(dHH) and p(r) to the
    experimental distributions. No free parameters.
    """
    # --- resolve the two (positionData, datasets) pickle pairs ---
    single_pos, single_data = _resolve_two_paths(SINGLE_FISH_PICKLE, 'SINGLE-FISH')
    pair_pos, pair_data = _resolve_two_paths(PAIR_PICKLE, 'PAIR')

    build_kinematic_bins = (condition_by_dHH_delta_s or condition_by_dHH_delta_t
                            or condition_by_dHH_IB_duration)
    suffix = run_label

    # --- single-fish backbone: (r) and (r, psi) bins, plus the w_excess null source ---
    # load_and_assign_from_pickle takes (positionData, datasets) and returns the
    # datasets list (with everything downstream needs) in its variable tuple.
    _, single_vt = load_and_assign_from_pickle(single_pos, single_data)
    single_datasets = single_vt[0]
    if not all(ds.get("Nfish", 2) == 1 for ds in single_datasets):
        raise ValueError('SINGLE_FISH_PICKLE must contain single-fish (Nfish==1) '
                         'datasets (the backbone that supplies the (r, psi) bins and '
                         'the real-paired w_excess null).')
    # Arena radius from the loaded experiment config (variable tuple index 2).
    expt_config = single_vt[2]
    arena_radius_mm = expt_config.get('arena_radius_mm')
    if arena_radius_mm is None:
        raise ValueError("'arena_radius_mm' not found in the single-fish datasets "
                         "pickle's expt_config.")
    print(f'  arena_radius_mm = {arena_radius_mm} mm (from the single-fish pickle)')
    _all_results, pooled_IB = get_InterBout_properties(single_datasets)
    radial_bins, _bin_edges = build_radial_bin_distributions(
        pooled_IB, arena_radius_mm, bin_size_mm=1.0,
        max_bout_speed_mm_s=max_bout_speed_mm_s,
        max_bout_turn_angle_rad_s=None, fps=fps)
    radial_psi_bins = build_radial_psi_bin_distributions(
        pooled_IB, arena_radius_mm, bin_size_mm=1.0, n_psi_bins=12,
        max_bout_speed_mm_s=max_bout_speed_mm_s,
        max_bout_turn_angle_rad_s=None, fps=fps)

    # --- pair data: kinematic bins, pooled experimental dHH / r ---
    _, pair_vt = load_and_assign_from_pickle(pair_pos, pair_data)
    pair_datasets = pair_vt[0]
    if not all(ds.get("Nfish", 0) == 2 for ds in pair_datasets):
        raise ValueError('PAIR_PICKLE must contain pair (Nfish==2) datasets.')

    kinematic_cond = None
    if build_kinematic_bins:
        kinematic_dHH_bins = build_radial_dHH_bin_distributions(
            pair_datasets, arena_radius_mm,
            bin_size_mm=kinematic_r_bin_size_mm,
            dHH_bin_size_mm=kinematic_dHH_bin_size_mm,
            max_bout_speed_mm_s=max_bout_speed_mm_s,
            max_bout_turn_angle_rad_s=None, fps=fps)
        if kinematic_dHH_bins is not None:
            kinematic_cond = {"bins": kinematic_dHH_bins,
                              "delta_s": condition_by_dHH_delta_s,
                              "IB_duration": condition_by_dHH_IB_duration,
                              "delta_t": condition_by_dHH_delta_t,
                              "resolution": kinematic_resolution}

    exp_dHH_values = _pool_experimental_dHH(pair_datasets)
    exp_r_values = _pool_experimental_r(pair_datasets)
    exp_dHH_by_dataset = _pool_experimental_dHH_by_dataset(pair_datasets)
    exp_r_by_dataset = _pool_experimental_r_by_dataset(pair_datasets)

    # --- focus factor f = sigma(dHH, |phi|) / sigma_far, from the pair data ---
    print('\n[FOCUS-RATIO] within-condition turn std sigma(dHH, |phi|) and its '
          'far-field asymptote...')
    _wcf = compute_within_condition_turn_std(
        pair_datasets, min_delta_s=social_min_delta_s,
        max_bout_speed_mm_s=max_bout_speed_mm_s,
        max_bout_turn_angle_rad_s=None, fps=fps)
    _sfar = _wcf["sigma_far"]
    if not (np.isfinite(_sfar) and _sfar > 0.0):
        raise ValueError('Focus ratio unavailable: sigma_far from the pair data is '
                         'not finite/positive. Check the pair pickle.')
    social_focus_f_ratio = (_wcf["dHH_centers"], _wcf["absphi_edges"],
                            _wcf["sigma_within_absphidHH"]/_sfar,
                            _wcf["sigma_within"]/_sfar, 0.0)   # f_floor = 0.0

    # --- excess social weight w_excess(dHH) = w_A - w_null, null = two real single
    #     fish paired (pseudopair; genuinely asocial) ---
    print('\n[W-EXCESS] w_excess(dHH) = w_A - w_null, null = two REAL single fish '
          'paired...')
    null_bouts = build_real_paired_null_bouts(single_datasets, np.random.default_rng(0))
    _bw = estimate_social_blend_weight_vs_distance(
        pair_datasets, radial_psi_bins, null_bouts=null_bouts, n_boot=200,
        min_delta_s=social_min_delta_s, target=social_track_target,
        max_bout_speed_mm_s=max_bout_speed_mm_s,
        max_bout_turn_angle_rad_s=None, fps=fps,
        labelB='two single fish (paired)',
        outputFileName1=f'social_blend_weight_vs_dHH_{suffix}.png',
        outputFileName2=f'social_blend_wexcess_vs_dHH_{suffix}.png',
        csvFileName=f'social_blend_weight_vs_dHH_{suffix}.csv',
        closeFigure=closeFigures)
    _wd, _we = _bw["dHH_centers"], _bw["w_excess"]
    _okw = np.isfinite(_wd) & np.isfinite(_we)
    if not np.any(_okw):
        raise ValueError('w_excess could not be estimated (no finite bins). Check the '
                         'pair and single-fish pickles.')
    social_track_w_excess = (_wd[_okw], _we[_okw])

    # --- optional diagnostic: (dHH, |phi|) turn-std figure ---
    if plot_phi_resolved_turn_std:
        print('\n[phi-RESOLVED spread] turn-std vs (dHH, |phi|) diagnostic...')
        phi_resolved_turn_std_vs_distance(
            pair_datasets, min_delta_s=social_min_delta_s,
            max_bout_speed_mm_s=max_bout_speed_mm_s,
            max_bout_turn_angle_rad_s=None, fps=fps,
            outputFileName=f'turn_std_phi_dHH_{suffix}.png',
            closeFigure=closeFigures)

    # --- simulate ---
    print(f'\nRunning {Ntrials} pair simulations '
          f'(social_track, target={social_track_target}, focus_ratio spread)...')
    dHH_list, r_list = simulate_pair_dHH_trials(
        radial_bins, arena_radius_mm, radial_psi_bins,
        social_track_w_excess, social_focus_f_ratio,
        kinematic_cond=kinematic_cond, social_track_target=social_track_target,
        edgeMethod=edgeMethod, Ntrials=Ntrials, T_total_s=T_total_s, dt_s=0.04,
        interp_method='nearest', plot_first_positions=True)

    # --- compare to experiment ---
    if plot_exp_vs_sim_dHH and exp_dHH_values is not None \
            and np.asarray(exp_dHH_values).size > 0:
        exp_dHH_by_ds = exp_dHH_by_dataset if exp_dHH_by_dataset else None
        plot_experimental_vs_sim_dHH(
            exp_dHH_values, dHH_list, social_method='turn_sampling_social_track',
            exp_dHH_list=exp_dHH_by_ds, sim_color=sim_color,
            outputFileName=f'compare_dHH_exp_vs_sim_{suffix}.png',
            closeFigure=closeFigures)
    elif plot_exp_vs_sim_dHH:
        print('\nSkipping p(dHH) overlay: no experimental head_head_distance_mm found.')

    if plot_exp_vs_sim_r and exp_r_values is not None \
            and np.asarray(exp_r_values).size > 0:
        exp_r_by_ds = exp_r_by_dataset if exp_r_by_dataset else None
        plot_experimental_vs_sim_r(
            exp_r_values, r_list, social_method='turn_sampling_social_track',
            r_max_mm=None, exp_r_list=exp_r_by_ds, sim_color=sim_color,
            outputFileName=f'compare_r_exp_vs_sim_{suffix}.png',
            closeFigure=closeFigures)
    elif plot_exp_vs_sim_r:
        print('\nSkipping p(r) overlay: no experimental radial_position_mm found.')

    print('\nDone.')
    if not closeFigures:
        plt.show()


if __name__ == '__main__':
    main()
