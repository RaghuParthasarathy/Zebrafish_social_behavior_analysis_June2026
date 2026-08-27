"""
IBI_properties_utils.py -- shared inter-bout-interval (IBI) / bout property
layer for the zebrafish behaviour analysis. Extracted from
random_displacement_analysis.py (2026-07) so that BOTH the pair simulation
(random_displacement_analysis) AND the single-fish simulation / assessment
(assess_single_fish_walks) AND the data-only IBI diagnostics can sit on ONE
sim-independent foundation -- and so these can be run on any dataset (e.g.
mutant fish) without importing the simulation code.

Contains: IBI-property extraction/pooling (get_InterBout_properties), the
radial / (r,psi) / (r,dHH) bin builders, and the physical-plausibility bout
filters (max bout speed / turn) plus the good-frame mask. Pure data + numpy;
depends on NOTHING in random_displacement_analysis (no import cycle).
"""
import numpy as np


def _good_frame_mask(ds, n_frames):
    """
    Boolean mask (length n_frames) of frames that are NOT in this dataset's
    "bad_bodyTrack_frames" (tracking failures). Mirrors the bad-frame exclusion the
    IBI pipeline uses, so frame-level pooling does not pick up mis-detections (e.g.
    a fish placed far outside the arena). Returns None if the frame numbering can't
    be matched (no/!=length frameArray), so the caller keeps all frames.
    """
    fa = np.asarray(ds.get("frameArray", []), dtype=int).ravel()
    if fa.size != n_frames:
        return None
    bad = ds.get("bad_bodyTrack_frames", {})
    bad_raw = (np.asarray(bad.get("raw_frames", []), dtype=int).ravel()
               if isinstance(bad, dict) else np.array([], dtype=int))
    if bad_raw.size == 0:
        return np.ones(n_frames, dtype=bool)
    return ~np.isin(fa, bad_raw)


def _bout_speed_ok(delta_s_mm, delta_t_s, max_bout_speed_mm_s):
    """Boolean keep-mask that rejects implausibly FAST bouts (bout speed =
    Delta_s_mm / Delta_t_s > max_bout_speed_mm_s) -- the ID-swap tracking jumps that
    appear at close range (a fish's track leaps to its partner), giving a large
    apparent displacement + turn. max_bout_speed_mm_s=None -> no filtering (all True).
    Bouts with non-finite or <= 0 Delta_t (speed undefined) are KEPT (not flagged)."""
    ds = np.asarray(delta_s_mm, dtype=float)
    if max_bout_speed_mm_s is None:
        return np.ones(ds.shape, dtype=bool)
    dt = np.asarray(delta_t_s, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        speed = ds / dt
    too_fast = (dt > 0) & np.isfinite(speed) & (speed > float(max_bout_speed_mm_s))
    return ~too_fast


def _bout_turn_ok(delta_theta_rad, max_bout_turn_angle_rad_s, fps=25.0):
    """Boolean keep-mask that rejects implausibly LARGE bout turns (|Delta_theta| >
    max_bout_turn_angle_rad_s / fps) -- tracking errors (e.g. ID swaps) that flip the
    apparent heading by ~pi within a single frame, which is physically impossible for
    a real fish. The cap is expressed as an angular RATE (rad/s) so it scales with
    frame rate: at fps=25 the default 22.5*pi rad/s rejects |Delta_theta| > 0.9*pi.
    max_bout_turn_angle_rad_s=None -> no filtering (all True). Non-finite turns are
    KEPT (not flagged), matching _bout_speed_ok's treatment of undefined quantities."""
    dth = np.asarray(delta_theta_rad, dtype=float)
    if max_bout_turn_angle_rad_s is None:
        return np.ones(dth.shape, dtype=bool)
    thresh = float(max_bout_turn_angle_rad_s) / float(fps)
    too_big = np.isfinite(dth) & (np.abs(dth) > thresh)
    return ~too_big


def get_InterBout_properties(datasets):
    """
    Build the pooled inter-bout step properties used by the random-walk
    simulations.

    Nothing is recomputed here: all per-IBI quantities (means AND the inter-bout
    step differences) are read from datasets[j]["IBI_properties"], which is
    produced by get_IBI_properties() in the main analysis pipeline
    (behavior_identification.py). This function pools them across datasets / fish
    and drops rows with non-finite values.

    Read from IBI_properties (per fish, per IBI):
        r_mm_mean, gamma_mean, r_mm_std, gamma_std, IB_duration_s, Delta_t_s,
        Delta_r_mm, Delta_gamma, Delta_s_mm, theta, Delta_theta, turning_angle_IBI
    (see get_IBI_properties for the step-difference / turning definitions).

    All stored IBIs are used: get_IBI_properties defines the step differences for
    every stored IBI (using the true neighbouring IBIs), so no further trimming is
    needed; rows whose mean position or step quantities are non-finite are skipped.

    Requires datasets[j]["IBI_properties"] to exist AND to contain the step-
    difference sub-keys (run the main pipeline, or regenerate existing pickle files
    with revise_datasets(keys_to_modify=["IBI_properties"]) from IO_toolkit).

    Returns
    -------
    all_results : list of dicts, one per (dataset, fish, IBI) row
    pooled_IB_properties : dict mapping property name -> 1D numpy array,
        pooled across all datasets and fish
    """
    missing = [j for j, ds in enumerate(datasets) if "IBI_properties" not in ds]
    if missing:
        raise KeyError(
            f'"IBI_properties" is missing from dataset(s) {missing}. Calculate '
            'IBI properties first (run the main analysis pipeline, or use '
            'revise_datasets(keys_to_modify=["IBI_properties"]) from IO_toolkit '
            'on the existing pickle files).')

    # Step-difference sub-keys are now produced by get_IBI_properties; older
    # pickle files predating that need regenerating.
    step_subkeys = ["Delta_r_mm", "Delta_gamma", "Delta_s_mm", "theta",
                    "Delta_theta", "v_r_mm_s", "wall_alignment"]
    for j, ds in enumerate(datasets):
        miss = [key for key in step_subkeys if key not in ds["IBI_properties"]]
        if miss:
            raise KeyError(
                f'IBI_properties in dataset {j} is missing step-difference '
                f'sub-key(s) {miss}. Regenerate with '
                'revise_datasets(keys_to_modify=["IBI_properties"]) from IO_toolkit '
                '(get_IBI_properties now stores these).')

    all_results = []

    for j, dataset in enumerate(datasets):
        ibi = dataset["IBI_properties"]
        Nfish = dataset["Nfish"]

        for k in range(Nfish):
            # Per-IBI means and step differences, all read from IBI_properties
            # (one value per stored IBI; aligned by index).
            r = np.asarray(ibi["r_mm_mean"][k], dtype=float)
            gamma = np.asarray(ibi["gamma_mean"][k], dtype=float)
            r_std = np.asarray(ibi["r_mm_std"][k], dtype=float)
            gamma_std = np.asarray(ibi["gamma_std"][k], dtype=float)
            ib_dur = np.asarray(ibi["IB_duration_s"][k], dtype=float)
            dt = np.asarray(ibi["Delta_t_s"][k], dtype=float)
            Delta_r = np.asarray(ibi["Delta_r_mm"][k], dtype=float)
            Delta_gamma = np.asarray(ibi["Delta_gamma"][k], dtype=float)
            Delta_s = np.asarray(ibi["Delta_s_mm"][k], dtype=float)
            theta = np.asarray(ibi["theta"][k], dtype=float)
            Delta_theta = np.asarray(ibi["Delta_theta"][k], dtype=float)
            turning_IBI = np.asarray(ibi["turning_angle_IBI"][k], dtype=float)
            v_r = np.asarray(ibi["v_r_mm_s"][k], dtype=float)
            wall_align = np.asarray(ibi["wall_alignment"][k], dtype=float)

            for i in range(len(r)):
                # Skip rows whose mean position or step quantities are non-finite
                if not (np.isfinite(r[i]) and np.isfinite(gamma[i])
                        and np.isfinite(Delta_r[i]) and np.isfinite(Delta_gamma[i])
                        and np.isfinite(Delta_s[i]) and np.isfinite(theta[i])
                        and np.isfinite(Delta_theta[i])
                        and np.isfinite(turning_IBI[i])
                        and np.isfinite(v_r[i])
                        and np.isfinite(wall_align[i])):
                    continue

                all_results.append({
                    "dataset_number": j,
                    "r_mm_mean":     r[i],
                    "gamma_mean":    gamma[i],
                    "r_mm_std":      r_std[i],
                    "gamma_std":     gamma_std[i],
                    "Delta_r_mm":    Delta_r[i],
                    "Delta_gamma":   Delta_gamma[i],
                    "Delta_s_mm":    Delta_s[i],
                    "theta":         theta[i],
                    "Delta_theta":   Delta_theta[i],
                    "turning_angle_IBI": turning_IBI[i],
                    "Delta_t_s":     dt[i],
                    "IB_duration_s": ib_dur[i],
                    "v_r_mm_s":      v_r[i],
                    "wall_alignment": wall_align[i],
                })

    # Build pooled_IB_properties
    property_keys = ["r_mm_mean", "gamma_mean", "r_mm_std", "gamma_std",
                     "Delta_r_mm", "Delta_gamma", "Delta_s_mm",
                     "theta", "Delta_theta", "turning_angle_IBI",
                     "Delta_t_s", "IB_duration_s", "v_r_mm_s", "wall_alignment"]
    pooled_IB_properties = {
        key: np.array([row[key] for row in all_results])
        for key in property_keys
    }

    print(f'\nFound {len(all_results)} inter-bout step rows '
          f'(read from IBI_properties; non-finite rows dropped).')

    return all_results, pooled_IB_properties


def build_radial_bin_distributions(pooled_IB_properties, arena_radius_mm,
                                   bin_size_mm=1.0, max_bout_speed_mm_s=None,
                                   max_bout_turn_angle_rad_s=None, fps=25.0):
    """
    Bin pooled IBI properties by radial position (r_mm_mean) and store
    the joint empirical distribution of the properties in each bin.

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
    max_bout_speed_mm_s : float or None (default). Universal bout-speed cap: drop
        physically impossible bouts (Delta_s / Delta_t > cap) -- ID-swap tracking
        jumps (a ~no-op on single-fish data). None -> keep all.
    max_bout_turn_angle_rad_s, fps : universal bout-turn cap (rad/s) and frame rate;
        drop bouts with |Delta_theta| > max_bout_turn_angle_rad_s / fps -- tracking
        errors that flip the heading by ~pi in one frame. None -> keep all.

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
            "turning_angle_IBI" : 1D array
            "wall_alignment"     : 1D array of per-IBI |sin(heading - gamma)|
            "v_r_mm_s"      : mean radial velocity in this bin (mm/s)
            "v_r_sem_mm_s"      : standard error of the mean,
                                  radial velocity in this bin (mm/s)
            "wall_alignment_mean" : mean |sin(heading - gamma)| in this bin
                                  (1 = parallel to wall, 0 = radial)
            "wall_alignment_sem"  : standard error of the mean wall alignment
            "disp_alignment"     : 1D array of per-IBI |sin(theta - gamma)| using
                                  the displacement direction theta (not heading)
            "disp_alignment_mean" : mean |sin(theta - gamma)| in this bin
                                  (incoming displacement-direction alignment)
            "disp_alignment_sem"  : standard error of the mean disp. alignment
            "disp_alignment_out" : 1D array of |sin(theta + Delta_theta - gamma)|,
                                  the OUTGOING displacement direction
            "disp_alignment_out_mean" : mean outgoing-displacement alignment in
                                  this bin (the simulation's modeling target)
            "disp_alignment_out_sem"  : standard error of that mean
            "N"             : number of observations in this bin
    bin_edges : 1D array of radial bin edges (mm), length = n_bins + 1
    """
    step_keys = ["Delta_r_mm", "Delta_gamma", "Delta_t_s", "IB_duration_s",
                 "Delta_s_mm", "Delta_theta", "turning_angle_IBI",
                 "v_r_mm_s", "wall_alignment"]

    bin_edges = np.arange(0.0, arena_radius_mm + bin_size_mm, bin_size_mm)
    n_bins = len(bin_edges) - 1

    r = np.asarray(pooled_IB_properties["r_mm_mean"], dtype=float)
    step_data = {key: np.asarray(pooled_IB_properties[key], dtype=float)
                 for key in step_keys}
    # [PHYSICAL FILTER] universal bout-speed cap: drop physically impossible bouts
    # (Delta_s / Delta_t > max_bout_speed_mm_s) -- ID-swap tracking jumps in pair data;
    # a ~no-op on single-fish data (which has no such fast bouts). None -> keep all.
    # Also drop physically impossible large TURNS (|Delta_theta| > cap/fps).
    keep = (_bout_speed_ok(step_data["Delta_s_mm"], step_data["Delta_t_s"],
                           max_bout_speed_mm_s)
            & _bout_turn_ok(step_data["Delta_theta"],
                            max_bout_turn_angle_rad_s, fps))

    # Displacement-direction alignment to the wall, |sin(theta - gamma_mean)|,
    # where theta is the per-IBI displacement direction (the step INTO each IBI)
    # and gamma_mean the polar position. Computed here from already-stored
    # quantities (no get_IBI_properties change / pickle regeneration needed);
    # 1 = displacement tangential to the wall, 0 = radial. Companion to the
    # body-heading "wall_alignment".
    theta_disp = np.asarray(pooled_IB_properties["theta"], dtype=float)
    gamma_pos = np.asarray(pooled_IB_properties["gamma_mean"], dtype=float)
    disp_align_all = np.abs(np.sin(theta_disp - gamma_pos))

    # OUTGOING-displacement alignment, |sin(theta_next - gamma_mean)|, where the
    # outgoing direction theta_next = theta + Delta_theta (Delta_theta is the
    # stored wrap(theta[next] - theta[this]); sin is periodic so the wrap is moot).
    # This is "from position r, which way does the fish leave" -- exactly what the
    # simulation moves along from a given position -- whereas disp_align_all uses
    # the INCOMING step (which carries the radial approach toward the wall). Also
    # from already-stored quantities; no regeneration needed.
    Delta_theta_pool = np.asarray(pooled_IB_properties["Delta_theta"],
                                  dtype=float)
    disp_align_out_all = np.abs(np.sin(theta_disp + Delta_theta_pool - gamma_pos))

    # np.digitize returns 1-indexed bin numbers; clamp to valid range
    bin_idx = np.clip(np.digitize(r, bin_edges) - 1, 0, n_bins - 1)

    radial_bins = []
    for i in range(n_bins):
        mask = (bin_idx == i) & keep
        entry = {
            "r_edges": (bin_edges[i], bin_edges[i + 1]),
            "N": int(mask.sum()),
        }
        for key in step_keys:
            entry[key] = step_data[key][mask]
        entry["disp_alignment"] = disp_align_all[mask]
        entry["disp_alignment_out"] = disp_align_out_all[mask]
        radial_bins.append(entry)

    # Average radial velocity in each bin, from the per-bout Cartesian radial
    # projection v_r_mm_s (computed in get_IBI_properties: the step's Cartesian
    # displacement projected onto the radial direction at the step's start,
    # divided by Delta_t). This is reliable at small r, where Delta_r/Delta_t is
    # not -- a near-center crossing flips the polar angle and inflates Delta_r.
    # The per-bout array (from the step_keys loop above) is replaced here by the
    # bin's scalar mean; SEM uses the finite-value count, not N.
    for j in range(n_bins):
        v_r_bin = radial_bins[j]["v_r_mm_s"]   # per-bout array (from step_keys)
        n_finite = int(np.sum(np.isfinite(v_r_bin)))
        if n_finite > 0:
            radial_bins[j]["v_r_mm_s"] = float(np.nanmean(v_r_bin))
            radial_bins[j]["v_r_sem_mm_s"] = float(np.nanstd(v_r_bin)
                                                   / np.sqrt(n_finite))
        else:
            radial_bins[j]["v_r_mm_s"] = np.nan
            radial_bins[j]["v_r_sem_mm_s"] = np.nan

    # Mean wall alignment in each bin, from the per-IBI |sin(heading - gamma)|
    # (computed in get_IBI_properties): 1 = heading parallel to the wall tangent,
    # 0 = radial. The per-bout array (from the step_keys loop) is kept under
    # "wall_alignment"; the bin's scalar mean and SEM are added as new keys.
    for j in range(n_bins):
        wa_bin = radial_bins[j]["wall_alignment"]   # per-bout array (step_keys)
        n_finite = int(np.sum(np.isfinite(wa_bin)))
        if n_finite > 0:
            radial_bins[j]["wall_alignment_mean"] = float(np.nanmean(wa_bin))
            radial_bins[j]["wall_alignment_sem"] = float(np.nanstd(wa_bin)
                                                         / np.sqrt(n_finite))
        else:
            radial_bins[j]["wall_alignment_mean"] = np.nan
            radial_bins[j]["wall_alignment_sem"] = np.nan

    # Mean displacement-direction alignment in each bin, from the per-bout
    # |sin(theta - gamma_mean)| (disp_align_all above): 1 = displacement
    # tangential to the wall, 0 = radial. Companion to wall_alignment_mean.
    for j in range(n_bins):
        da_bin = radial_bins[j]["disp_alignment"]
        n_finite = int(np.sum(np.isfinite(da_bin)))
        if n_finite > 0:
            radial_bins[j]["disp_alignment_mean"] = float(np.nanmean(da_bin))
            radial_bins[j]["disp_alignment_sem"] = float(np.nanstd(da_bin)
                                                         / np.sqrt(n_finite))
        else:
            radial_bins[j]["disp_alignment_mean"] = np.nan
            radial_bins[j]["disp_alignment_sem"] = np.nan

    # Mean OUTGOING-displacement alignment in each bin (disp_align_out_all above):
    # |sin(theta_next - gamma)| with theta_next = theta + Delta_theta. This is the
    # direction the fish leaves a given position, the simulation's modeling target.
    for j in range(n_bins):
        dao_bin = radial_bins[j]["disp_alignment_out"]
        n_finite = int(np.sum(np.isfinite(dao_bin)))
        if n_finite > 0:
            radial_bins[j]["disp_alignment_out_mean"] = float(np.nanmean(dao_bin))
            radial_bins[j]["disp_alignment_out_sem"] = float(np.nanstd(dao_bin)
                                                             / np.sqrt(n_finite))
        else:
            radial_bins[j]["disp_alignment_out_mean"] = np.nan
            radial_bins[j]["disp_alignment_out_sem"] = np.nan

    # Print a summary of bin occupancy
    print(f'\nRadial bin occupancy (bin size = {bin_size_mm} mm, '
          f'arena radius = {arena_radius_mm} mm):')
    print(f'  {"Bin (mm)":<16}  {"N obs":>6}')
    for i, b in enumerate(radial_bins):
        lo, hi = b["r_edges"]
        print(f'  {lo:.1f} - {hi:.1f}{"":>8}  {b["N"]:>6}')

    return radial_bins, bin_edges


def build_radial_psi_bin_distributions(pooled_IB_properties, arena_radius_mm,
                                       bin_size_mm=1.0, n_psi_bins=8,
                                       max_bout_speed_mm_s=None,
                                       max_bout_turn_angle_rad_s=None, fps=25.0):
    """
    Bin the pooled single-fish IBI steps jointly by radial position r and the
    INCOMING wall orientation psi_in = wrap(theta - gamma_mean) -- the displacement
    direction INTO each IBI relative to the outward radial at that position
    (psi_in = 0 radially outward, +-pi/2 tangential, +-pi radially inward). Stores
    the joint empirical step distribution in each (r, psi) bin: the wall-
    conditioned analog of build_radial_bin_distributions, and the single-fish
    analog of the (r, dHH) pair bins (the wall is the "feature", psi its bearing).

    For the simulation this is the non-parametric wall-following / retention model.
    Sampling the turn (and step) conditioned on the fish's current orientation to
    the wall reproduces the empirical corrective-turning RESPONSE -- e.g. a fish
    pointing inward (psi ~ +-pi) tends to turn back toward tangential/outward, and
    one near the wall rarely commits an inward step -- which a turn distribution
    conditioned on r alone (or a symmetric g*sin(2 psi) alignment torque -- see
    single_fish_archived_methods.py) cannot capture. Conditioning on the INCOMING
    psi (known before the bout) avoids circularity. This is the current single-fish
    wall model; see single_fish_simulation_summary.md.

    Inputs
    ------
    pooled_IB_properties : dict from get_InterBout_properties() (needs theta,
        gamma_mean, r_mm_mean, and the step keys).
    arena_radius_mm : float; radial bins span [0, arena] in steps of bin_size_mm.
    bin_size_mm : float; radial bin width (mm).
    n_psi_bins : int; number of (periodic) psi bins over [-pi, pi]. Keep small
        (e.g. 4-8) so each (r, psi) cell has adequate counts.
    max_bout_speed_mm_s : float or None (default). Universal bout-speed cap: drop
        physically impossible bouts (Delta_s / Delta_t > cap; ID-swap tracking jumps,
        a ~no-op on single-fish data). None -> keep all.
    max_bout_turn_angle_rad_s, fps : universal bout-turn cap (rad/s) and frame rate;
        drop bouts with |Delta_theta| > max_bout_turn_angle_rad_s / fps (impossible
        heading flips from tracking errors). None -> keep all.

    Returns
    -------
    radial_psi_bins : dict with
        "bins"     : 2-D list (n_r x n_psi) of dicts, each with "r_edges",
                     "psi_edges", the step arrays (Delta_r_mm, Delta_gamma,
                     Delta_t_s, IB_duration_s, Delta_s_mm, Delta_theta,
                     turning_angle_IBI), "N", and the per-bin CIRCULAR summary of
                     the intrinsic turn ti = -Delta_theta: "ti_mean" (circular
                     mean, rad) and "ti_std" (circular std = sqrt(-2 ln R), rad;
                     inf if the resultant length R is 0, NaN if N == 0).
                     [PRECISION FEATURE] ti_mean/ti_std are the intrinsic-turn
                     cue's mean and precision for a precision-weighted blend with
                     the social turn -- computed analytically from the pooled bin
                     array (no Monte Carlo). NOTE: tiny-N bins give an
                     under-dispersed (even zero) ti_std; weight by N when using.
        "r_edges"  : 1D array of radial bin edges (mm), length n_r + 1
        "psi_edges": 1D array of psi bin edges (rad), length n_psi + 1
    """
    step_keys = ["Delta_r_mm", "Delta_gamma", "Delta_t_s", "IB_duration_s",
                 "Delta_s_mm", "Delta_theta", "turning_angle_IBI"]

    r_edges = np.arange(0.0, arena_radius_mm + bin_size_mm, bin_size_mm)
    psi_edges = np.linspace(-np.pi, np.pi, n_psi_bins + 1)
    n_r = len(r_edges) - 1
    n_psi = len(psi_edges) - 1

    r_all = np.asarray(pooled_IB_properties["r_mm_mean"], dtype=float)
    theta_all = np.asarray(pooled_IB_properties["theta"], dtype=float)
    gamma_all = np.asarray(pooled_IB_properties["gamma_mean"], dtype=float)
    psi_all = (theta_all - gamma_all + np.pi) % (2.0*np.pi) - np.pi

    # [PHYSICAL FILTER] universal bout-speed cap (Delta_s / Delta_t) and bout-turn cap
    # (|Delta_theta| > cap/fps): drop physically impossible bouts (ID-swap tracking
    # jumps / heading flips in pair data; ~no-op on single-fish).
    _ds_all = np.asarray(pooled_IB_properties["Delta_s_mm"], dtype=float)
    _dt_all = np.asarray(pooled_IB_properties["Delta_t_s"], dtype=float)
    _dth_all = np.asarray(pooled_IB_properties["Delta_theta"], dtype=float)
    keep = (np.isfinite(psi_all) & np.isfinite(r_all)
            & _bout_speed_ok(_ds_all, _dt_all, max_bout_speed_mm_s)
            & _bout_turn_ok(_dth_all, max_bout_turn_angle_rad_s, fps))
    r_k = r_all[keep]
    psi_k = psi_all[keep]
    step_data = {key: np.asarray(pooled_IB_properties[key], dtype=float)[keep]
                 for key in step_keys}

    i_r_all = np.clip(np.digitize(r_k, r_edges) - 1, 0, n_r - 1)
    j_p_all = np.clip(np.digitize(psi_k, psi_edges) - 1, 0, n_psi - 1)

    bins2D = []
    for i in range(n_r):
        row = []
        for jj in range(n_psi):
            mask = (i_r_all == i) & (j_p_all == jj)
            entry = {
                "r_edges": (r_edges[i], r_edges[i + 1]),
                "psi_edges": (psi_edges[jj], psi_edges[jj + 1]),
                "N": int(mask.sum()),
            }
            for key in step_keys:
                entry[key] = step_data[key][mask]
            # [PRECISION FEATURE] Per-bin circular mean and std of the intrinsic
            # turn ti = -Delta_theta, computed directly from this bin's pooled
            # array (no sampling). R = resultant length; circular std =
            # sqrt(-2 ln R) (-> inf as R -> 0, i.e. a flat/low-precision turn).
            if entry["N"] > 0:
                dth = entry["Delta_theta"]
                C = np.mean(np.cos(dth))
                S = np.mean(np.sin(dth))
                R_len = np.hypot(C, S)
                entry["ti_mean"] = -np.arctan2(S, C)   # circ. mean of -Delta_theta
                entry["ti_std"] = (np.sqrt(-2.0*np.log(R_len)) if R_len > 0.0
                                   else np.inf)
            else:
                entry["ti_mean"] = np.nan
                entry["ti_std"] = np.nan
            row.append(entry)
        bins2D.append(row)

    radial_psi_bins = {"bins": bins2D, "r_edges": r_edges, "psi_edges": psi_edges}

    n_empty = sum(1 for i in range(n_r) for jj in range(n_psi)
                  if bins2D[i][jj]["N"] == 0)
    print(f'\nBuilt (r, psi) step bins: {len(r_k)} steps in a {n_r} x {n_psi} grid '
          f'(radial {bin_size_mm} mm, {n_psi} psi bins); '
          f'{n_empty} of {n_r*n_psi} bins empty.')

    return radial_psi_bins


def build_radial_dHH_bin_distributions(datasets, arena_radius_mm,
                                       bin_size_mm=1.0, dHH_bin_size_mm=5.0,
                                       dHH_max_mm=None, n_phi_bins=4,
                                       max_bout_speed_mm_s=None,
                                       max_bout_turn_angle_rad_s=None, fps=25.0):
    """
    Bin per-IBI steps from PAIR data jointly by the focal fish's radial position
    and the inter-fish head-head distance (dHH), both taken at the START of each
    step, storing the empirical step distribution in each (r, dHH) bin.

    This is the (r, dHH) pool that the [dHH-KIN] kinematic_cond feature draws
    (Delta_s, IB_duration, Delta_t) from -- used by ALL methods that go through the
    _draw_additive path (additive, additive_r, gated, softgate, softblend,
    wall_vs_neighbor, AND social_focus / social_track). It also now builds the
    (r, dHH, |phi|) 3-D cells and the (r)-only marginal (see n_phi_bins); the
    sampler picks the resolution. Unlike
    build_radial_bin_distributions (1-D, in r only, typically built from single-
    fish data), the step distribution here is conditioned on dHH, so it carries
    the socially-modulated step statistics (e.g. a larger inward Delta_r, or a
    different step length / turn / pause when a neighbour is close). Each step
    also stores its frame-portable body-frame turn Delta_theta (the change in
    displacement heading) and step length Delta_s_mm.
    (It was also the candidate pool for the removed 'weighted_radial_dHH' social
    method; see pair_fish_archived_methods.py.)

    Steps are read from IBI_properties (Delta_r_mm, Delta_gamma, Delta_s_mm,
    Delta_theta, Delta_t_s, IB_duration_s), as produced by get_IBI_properties, and
    binned by the starting (r_mm_mean, head_head_distance_mm_mean) of the same IBI.
    All stored IBIs are used (non-finite rows dropped). Requires Nfish == 2.

    Inputs
    ------
    datasets : list of dataset dicts, each with "IBI_properties" (Nfish==2) and
               sub-keys r_mm_mean, head_head_distance_mm_mean, Delta_t_s,
               IB_duration_s, Delta_r_mm, Delta_gamma, Delta_s_mm, Delta_theta.
    arena_radius_mm : float; radial bins span [0, arena] in steps of bin_size_mm.
    bin_size_mm : float; radial bin width (mm).
    dHH_bin_size_mm : float; head-head-distance bin width (mm). Coarser than the
               radial bins by default, since the 2-D grid needs more data per bin.
    dHH_max_mm : float or None; upper dHH edge. If None, 2*arena_radius_mm (the
               maximum possible separation).
    max_bout_speed_mm_s : float or None; drop bouts with Delta_s/Delta_t > cap
               (ID-swap tracking jumps). None -> keep all.
    max_bout_turn_angle_rad_s, fps : float or None, float; drop bouts with
               |Delta_theta| > max_bout_turn_angle_rad_s / fps (impossible large
               turns from tracking errors). None -> keep all.

    Returns
    -------
    radial_dHH_bins : dict with
        "bins"      : 2-D list (n_r x n_dHH) of dicts, each with "r_edges",
                      "dHH_edges", "Delta_r_mm", "Delta_gamma", "Delta_t_s",
                      "IB_duration_s", "Delta_s_mm", "Delta_theta", "N"
        "r_edges"   : 1D array of radial bin edges (mm), length n_r + 1
        "dHH_edges" : 1D array of dHH bin edges (mm), length n_dHH + 1
        "bins_phi"  : 3-D list (n_r x n_dHH x n_phi_bins) of dicts with the drawn
                      triple ("Delta_s_mm", "IB_duration_s", "Delta_t_s", "N"),
                      resolved by |phi| (folded neighbour bearing); "bins" is its
                      phi-marginal / fallback.
        "bins_r"    : 1-D list (n_r) of the same triple, the dHH- and phi-marginal
                      (the 'average' resolution).
        "phi_edges" : 1D array of |phi| bin edges (rad, 0..pi), length n_phi_bins + 1
        "n_phi_bins": int
    The kinematic sampler picks the level via its `resolution`
    ('average' | 'dHH' | 'dHH_phi'); see sample_kinematics_from_radial_dHH_bin.
    """
    step_subkeys = ["Delta_r_mm", "Delta_gamma", "Delta_s_mm", "Delta_theta"]
    for j, ds in enumerate(datasets):
        if ds["Nfish"] != 2:
            raise ValueError('build_radial_dHH_bin_distributions requires '
                             f'Nfish==2; dataset {j} has Nfish={ds["Nfish"]}.')
        if "IBI_properties" not in ds:
            raise KeyError(f'"IBI_properties" missing from dataset {j}.')
        miss = [key for key in step_subkeys if key not in ds["IBI_properties"]]
        if miss:
            raise KeyError(
                f'IBI_properties in dataset {j} is missing step-difference '
                f'sub-key(s) {miss}. Regenerate with '
                'revise_datasets(keys_to_modify=["IBI_properties"]) from IO_toolkit.')

    if dHH_max_mm is None:
        dHH_max_mm = 2.0 * arena_radius_mm

    r_edges = np.arange(0.0, arena_radius_mm + bin_size_mm, bin_size_mm)
    dHH_edges = np.arange(0.0, dHH_max_mm + dHH_bin_size_mm, dHH_bin_size_mm)
    n_r = len(r_edges) - 1
    n_d = len(dHH_edges) - 1

    # Pool per-step quantities across datasets / fish
    r_start_all, dHH_start_all = [], []
    Dr_all, Dg_all, Dt_all, ib_all = [], [], [], []
    Ds_all, Dtheta_all, phi_all = [], [], []
    for ds in datasets:
        ibi = ds["IBI_properties"]
        for k in range(2):
            # Per-IBI means / step differences read from IBI_properties; the step
            # length Delta_s and body-frame turn Delta_theta therefore match the
            # 1-D radial_bins (both from get_IBI_properties), aligned by index.
            r = np.asarray(ibi["r_mm_mean"][k], dtype=float)
            dt = np.asarray(ibi["Delta_t_s"][k], dtype=float)
            ibd = np.asarray(ibi["IB_duration_s"][k], dtype=float)
            dHH = np.asarray(ibi["head_head_distance_mm_mean"][k], dtype=float)
            Dr = np.asarray(ibi["Delta_r_mm"][k], dtype=float)
            Dg = np.asarray(ibi["Delta_gamma"][k], dtype=float)
            Ds = np.asarray(ibi["Delta_s_mm"][k], dtype=float)
            Dth = np.asarray(ibi["Delta_theta"][k], dtype=float)
            ph = np.asarray(ibi["relative_orientation_mean"][k], dtype=float)
            # max_bout_speed_mm_s: skip ID-swap tracking jumps (Delta_s/Delta_t too
            # fast); max_bout_turn_angle_rad_s: skip impossible large turns
            # (|Delta_theta| > cap/fps). Both corrupt the close-range kinematics.
            too_fast = ~(_bout_speed_ok(Ds, dt, max_bout_speed_mm_s)
                         & _bout_turn_ok(Dth, max_bout_turn_angle_rad_s, fps))
            for i in range(len(r)):
                if not (np.isfinite(r[i]) and np.isfinite(dHH[i])
                        and np.isfinite(Dr[i]) and np.isfinite(Dg[i])
                        and np.isfinite(Ds[i]) and np.isfinite(Dth[i])):
                    continue
                if too_fast[i]:
                    continue
                r_start_all.append(r[i])
                dHH_start_all.append(dHH[i])
                Dr_all.append(Dr[i])
                Dg_all.append(Dg[i])
                Dt_all.append(dt[i])
                ib_all.append(ibd[i])
                Ds_all.append(Ds[i])
                Dtheta_all.append(Dth[i])
                phi_all.append(ph[i])

    r_start_all = np.asarray(r_start_all)
    dHH_start_all = np.asarray(dHH_start_all)
    Dr_all = np.asarray(Dr_all)
    Dg_all = np.asarray(Dg_all)
    Dt_all = np.asarray(Dt_all)
    ib_all = np.asarray(ib_all)
    Ds_all = np.asarray(Ds_all)
    Dtheta_all = np.asarray(Dtheta_all)
    phi_all = np.asarray(phi_all)

    # 2-D bin assignment (np.digitize is 1-indexed; clamp to the valid range)
    i_r_all = np.clip(np.digitize(r_start_all, r_edges) - 1, 0, n_r - 1)
    j_d_all = np.clip(np.digitize(dHH_start_all, dHH_edges) - 1, 0, n_d - 1)

    bins2D = []
    for i in range(n_r):
        row = []
        for jj in range(n_d):
            mask = (i_r_all == i) & (j_d_all == jj)
            row.append({
                "r_edges": (r_edges[i], r_edges[i + 1]),
                "dHH_edges": (dHH_edges[jj], dHH_edges[jj + 1]),
                "Delta_r_mm": Dr_all[mask],
                "Delta_gamma": Dg_all[mask],
                "Delta_t_s": Dt_all[mask],
                "IB_duration_s": ib_all[mask],
                "Delta_s_mm": Ds_all[mask],
                "Delta_theta": Dtheta_all[mask],
                "N": int(mask.sum()),
            })
        bins2D.append(row)

    # [dHH-KIN, phi] Additional (r, dHH, |phi|) 3-D cells and (r)-only marginals, so
    # the kinematic draw can optionally resolve by neighbour bearing (|phi|, folded
    # 45-deg bins) -- the sim slow-brakes when a neighbour is ahead / darts when behind
    # -- or coarsen to (r)-only ('average'). Only the drawn triple (Delta_s,
    # IB_duration, Delta_t) is stored at these extra levels. |phi| = |relative
    # orientation|; the 2-D "bins" above is the (r, dHH) phi-marginal / fallback.
    phi_edges = np.linspace(0.0, np.pi, n_phi_bins + 1)
    absphi_all = np.abs((phi_all + np.pi) % (2.0*np.pi) - np.pi)
    k_p_all = np.clip(np.digitize(absphi_all, phi_edges) - 1, 0, n_phi_bins - 1)
    finphi = np.isfinite(absphi_all)
    bins_phi = []
    for i in range(n_r):
        plane = []
        for jj in range(n_d):
            col = []
            base = (i_r_all == i) & (j_d_all == jj) & finphi
            for kk in range(n_phi_bins):
                mask = base & (k_p_all == kk)
                col.append({"Delta_t_s": Dt_all[mask],
                            "IB_duration_s": ib_all[mask],
                            "Delta_s_mm": Ds_all[mask],
                            "N": int(mask.sum())})
            plane.append(col)
        bins_phi.append(plane)
    bins_r = []
    for i in range(n_r):
        mask = (i_r_all == i)
        bins_r.append({"r_edges": (r_edges[i], r_edges[i + 1]),
                       "Delta_t_s": Dt_all[mask],
                       "IB_duration_s": ib_all[mask],
                       "Delta_s_mm": Ds_all[mask],
                       "N": int(mask.sum())})

    radial_dHH_bins = {"bins": bins2D, "r_edges": r_edges, "dHH_edges": dHH_edges,
                       "bins_phi": bins_phi, "bins_r": bins_r,
                       "phi_edges": phi_edges, "n_phi_bins": n_phi_bins}

    n_empty = sum(1 for i in range(n_r) for jj in range(n_d)
                  if bins2D[i][jj]["N"] == 0)
    n_empty_phi = sum(1 for i in range(n_r) for jj in range(n_d)
                      for kk in range(n_phi_bins) if bins_phi[i][jj][kk]["N"] == 0)
    print(f'\nBuilt (r, dHH) step bins from pair data: {len(r_start_all)} steps '
          f'in a {n_r} x {n_d} grid (radial {bin_size_mm} mm, dHH '
          f'{dHH_bin_size_mm} mm); {n_empty} of {n_r*n_d} bins empty. '
          f'(r, dHH, |phi|): {n_empty_phi} of {n_r*n_d*n_phi_bins} cells empty '
          f'({n_phi_bins} phi bins; phi-marginal fallback).')

    return radial_dHH_bins


def _density_and_sem(arrays, edges):
    """
    Given a list of 1D sample arrays (one per replicate -- per experimental
    dataset, or per simulation trial), histogram each on the shared bin `edges`
    as a normalized density, and return (mean_density, sem_density) where the
    mean is over the POOLED samples (all arrays concatenated) and the sem is the
    across-replicate standard error std(per-replicate densities) / sqrt(Nrep)
    -- the same across-subunit s.e.m. used by the IO_toolkit distribution plots.
    Returns (pooled_density, None) if fewer than 2 non-empty replicates are
    available (no meaningful across-replicate spread).
    """
    per_rep = []
    for a in arrays:
        a = np.asarray(a, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        h, _ = np.histogram(a, bins=edges, density=True)
        per_rep.append(h)
    pooled = np.concatenate([np.asarray(a, dtype=float).ravel()
                             for a in arrays]) if len(arrays) else np.array([])
    pooled = pooled[np.isfinite(pooled)]
    pooled_density, _ = np.histogram(pooled, bins=edges, density=True)
    if len(per_rep) >= 2:
        stack = np.vstack(per_rep)
        sem = np.std(stack, axis=0) / np.sqrt(stack.shape[0])
    else:
        sem = None
    return pooled_density, sem
