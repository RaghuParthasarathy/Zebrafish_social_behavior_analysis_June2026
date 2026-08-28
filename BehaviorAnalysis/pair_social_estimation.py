"""
pair_social_estimation.py

Estimation of the pair-social model parameters from data -- the layer between the
sim-free data utilities (IBI_properties_utils) and the simulation / main pipeline
(random_displacement_analysis.py and pair_simulate_and_compare.py). These functions
turn experimental pair (and single-fish) trajectories into the quantities the pair
simulation consumes:

  - compute_within_condition_turn_std : within-condition turn std sigma(dHH, |phi|)
        and its far-field asymptote -> the focus factor f for the focus_ratio spread.
  - estimate_social_blend_weight_vs_distance : the social blend weight w(dHH) and the
        excess w_excess = w_A - w_null (a WLS projection of unit turn vectors) -> the
        mean-steering weight.
  - build_real_paired_null_bouts : the genuinely-asocial "two real single fish paired"
        null bouts that estimate_social_blend_weight_vs_distance(null_bouts=...) uses.

Moved here from IBI_diagnostics.py: these are model-parameter estimators, not
data-only diagnostics. Imports only from IBI_properties_utils (the data layer), so
there is no import cycle with the diagnostics or simulation modules.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from IBI_properties_utils import (_bout_speed_ok, _bout_turn_ok, _fit_sigma_asymptote)


def _relative_orientation_focal(heading, dx, dy):
    """Signed relative orientation phi of a FOCAL fish (body heading `heading`, rad)
    toward a partner offset (dx, dy) = partner - focal, matching the EXACT convention
    of calc_relative_orientation / get_relative_orientation (dot-product magnitude,
    cross-product sign; phi = -unsigned where the cross-z >= 0). Vectorized. Returns
    NaN where the offset is degenerate (dHH == 0)."""
    heading = np.asarray(heading, dtype=float)
    dx = np.asarray(dx, dtype=float); dy = np.asarray(dy, dtype=float)
    norm = np.hypot(dx, dy)
    with np.errstate(invalid='ignore', divide='ignore'):
        ux, uy = dx/norm, dy/norm
    vx, vy = np.cos(heading), np.sin(heading)
    dot = np.clip(vx*ux + vy*uy, -1.0, 1.0)
    unsigned = np.arccos(dot)
    cross = vx*uy - vy*ux
    phi = np.where(cross >= 0.0, -unsigned, unsigned)
    phi = np.where(norm > 0.0, phi, np.nan)
    return phi




def compute_within_condition_turn_std(datasets_A, dHH_edges=None,
                                      n_r_bins=3, n_psi_bins=4, n_phi_bins=6,
                                      min_cell_N=8, min_delta_s=0.0,
                                      max_bout_speed_mm_s=None,
                                      max_bout_turn_angle_rad_s=None, fps=25.0):
    """
    [SOCIAL_TRACK spread -- A-only, B-INDEPENDENT] Within-condition circular std of
    the inter-bout turn (-Delta_theta) vs inter-fish distance dHH, computed from a
    SINGLE pair dataset A (no subtrahend B). Conditions out the (r, psi, phi, dHH)-
    cell mean (residual circular std), exactly as compare_pair_turn_std_vs_distance
    does for its sigma_within_A -- but the coarse r-bin EDGES are quantiles of A's
    OWN radial positions, so the result does not depend on any second dataset. This
    is the spread source for social_method='turn_sampling_social_track' in 'sigmaA' /
    'sigmaA_r' mode, making that model genuinely B-independent (it needs only the
    minuend pair pickle pairstats_2).

    Inputs mirror compare_pair_turn_std_vs_distance (same nuisance-grid resolution
    and min_cell_N defaults), but take only datasets_A. min_delta_s (mm, default 0)
    drops small bouts whose noise-dominated displacement direction inflates the
    spread (see diagnose_delta_theta_vs_heading_turn); ~1-2 mm is the adopted standard.
    max_bout_speed_mm_s (mm/s, default None) drops implausibly fast bouts (Delta_s /
    Delta_t) -- ID-swap tracking jumps at close range that inject spurious large turns.

    Returns
    -------
    dict with dHH_centers (mm), sigma_within (n_dhh, rad), sigma_within_rdHH
    (n_r_bins x n_dhh, rad; NaN where under-populated), r_edges (mm),
    sigma_within_absphidHH (n_absphi x n_dhh, rad; |phi| in hard-wired 45-deg bins),
    absphi_edges (rad), and the per-bin residual counts N (n_dhh), N_rdHH, N_absphidHH.
    For the simulation:
      social_focus_sigma_abs  = (dHH_centers, sigma_within)              ['sigmaA']
      social_track_sigma_rmap = (dHH_centers, r_edges, sigma_within_rdHH) ['sigmaA_r']
      social_focus_sigma_phi  = (dHH_centers, absphi_edges,
                                 sigma_within_absphidHH, sigma_within)   [SIGMA-PHI]
    """
    def _wrap(a):
        return (np.asarray(a, dtype=float) + np.pi) % (2.0*np.pi) - np.pi

    def _circ_mean(a):
        return np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))

    def _circ_std(a):
        if a.size == 0:
            return np.nan
        R = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
        return np.sqrt(-2.0*np.log(R)) if R > 0.0 else np.inf

    # Pool A's per-IBI (turn, r, psi, phi, dHH, Delta_s) into finite 1-D arrays.
    turn, r, psi, phi, dhh, dsz, dts = [], [], [], [], [], [], []
    for d in datasets_A:
        if d["Nfish"] != 2:
            raise ValueError('compute_within_condition_turn_std needs pair '
                             '(Nfish==2) data.')
        ip = d["IBI_properties"]
        for k in range(d["Nfish"]):
            turn.append(-np.asarray(ip["Delta_theta"][k], dtype=float))
            r.append(np.asarray(ip["r_mm_mean"][k], dtype=float))
            th = np.asarray(ip["theta"][k], dtype=float)
            gm = np.asarray(ip["gamma_mean"][k], dtype=float)
            psi.append(_wrap(th - gm))
            phi.append(np.asarray(ip["relative_orientation_mean"][k], dtype=float))
            dhh.append(np.asarray(ip["head_head_distance_mm_mean"][k], dtype=float))
            dsz.append(np.asarray(ip["Delta_s_mm"][k], dtype=float))
            dts.append(np.asarray(ip["Delta_t_s"][k], dtype=float))
    turn = np.concatenate(turn); r = np.concatenate(r)
    psi = np.concatenate(psi); phi = np.concatenate(phi); dhh = np.concatenate(dhh)
    dsz = np.concatenate(dsz); dts = np.concatenate(dts)
    # min_delta_s: drop small bouts (noise-dominated displacement direction inflates
    # the within-condition turn spread as well as the mean). max_bout_speed_mm_s: drop
    # implausibly FAST bouts (Delta_s / Delta_t) -- ID-swap tracking jumps at close
    # range that inject spurious large turns (see the dHH/bout-speed diagnosis).
    # max_bout_turn_angle_rad_s: also drop impossible large turns (|Delta_theta|>cap/fps).
    ok = (np.isfinite(turn) & np.isfinite(r) & np.isfinite(psi)
          & np.isfinite(phi) & np.isfinite(dhh) & np.isfinite(dsz)
          & (dsz > min_delta_s) & _bout_speed_ok(dsz, dts, max_bout_speed_mm_s)
          & _bout_turn_ok(turn, max_bout_turn_angle_rad_s, fps))
    turn, r, psi, phi, dhh = turn[ok], r[ok], psi[ok], phi[ok], dhh[ok]

    # Bin edges. r EDGES FROM A ALONE (the B-independence point); psi, phi linear.
    if dHH_edges is None:
        dHH_edges = np.linspace(0.0, 40.0, 14)
    dHH_edges = np.asarray(dHH_edges, dtype=float)
    n_dhh = len(dHH_edges) - 1
    dHH_centers = 0.5*(dHH_edges[:-1] + dHH_edges[1:])
    r_edges = np.quantile(r, np.linspace(0.0, 1.0, n_r_bins + 1))
    r_edges = np.maximum.accumulate(r_edges)
    for i in range(1, len(r_edges)):
        if r_edges[i] <= r_edges[i - 1]:
            r_edges[i] = np.nextafter(r_edges[i - 1], np.inf)
    psi_edges = np.linspace(-np.pi, np.pi, n_psi_bins + 1)
    phi_edges = np.linspace(-np.pi, np.pi, n_phi_bins + 1)

    # Angular residual of each step about its (r, psi, phi, dHH) cell circular mean
    # (cells with >= min_cell_N steps; others dropped).
    ir = np.clip(np.digitize(r, r_edges) - 1, 0, n_r_bins - 1)
    jp = np.clip(np.digitize(psi, psi_edges) - 1, 0, n_psi_bins - 1)
    kp = np.clip(np.digitize(phi, phi_edges) - 1, 0, n_phi_bins - 1)
    ld = np.digitize(dhh, dHH_edges) - 1
    in_dhh = (ld >= 0) & (ld < n_dhh)
    cid = (((ir*n_psi_bins + jp)*n_phi_bins + kp)*n_dhh + ld)
    cid = np.where(in_dhh, cid, -1)
    resid = np.full(turn.shape, np.nan)
    order = np.argsort(cid, kind='stable')
    cid_s = cid[order]
    uniq, start = np.unique(cid_s, return_index=True)
    start = np.append(start, len(cid_s))
    for m in range(len(uniq)):
        if uniq[m] < 0:
            continue
        idx = order[start[m]:start[m + 1]]
        if idx.size >= min_cell_N:
            resid[idx] = _wrap(turn[idx] - _circ_mean(turn[idx]))
    keep = np.isfinite(resid)
    resid_k, ld_k, ir_k = resid[keep], ld[keep], ir[keep]
    # |phi| bin of each kept step: hard-wired 45-deg bins (0-45, 45-90, 90-135,
    # 135-180) for the phi-resolved social-focus spread [SIGMA-PHI].
    absphi_edges = np.radians([0.0, 45.0, 90.0, 135.0, 180.0])
    n_absphi = len(absphi_edges) - 1
    iabs_k = np.clip(np.digitize(np.abs(_wrap(phi))[keep], absphi_edges) - 1,
                     0, n_absphi - 1)

    # Collapse the residual spread over dHH (1-D), (r, dHH) (2-D), and (|phi|, dHH).
    sigma = np.full(n_dhh, np.nan)
    N = np.zeros(n_dhh, dtype=int)
    for b in range(n_dhh):
        a = resid_k[ld_k == b]
        N[b] = a.size
        if a.size >= min_cell_N:
            sigma[b] = _circ_std(a)
    sigma_rdHH = np.full((n_r_bins, n_dhh), np.nan)
    N_rdHH = np.zeros((n_r_bins, n_dhh), dtype=int)
    for ii in range(n_r_bins):
        for b in range(n_dhh):
            a = resid_k[(ir_k == ii) & (ld_k == b)]
            N_rdHH[ii, b] = a.size
            if a.size >= min_cell_N:
                sigma_rdHH[ii, b] = _circ_std(a)
    sigma_absphidHH = np.full((n_absphi, n_dhh), np.nan)
    N_absphidHH = np.zeros((n_absphi, n_dhh), dtype=int)
    for ii in range(n_absphi):
        for b in range(n_dhh):
            a = resid_k[(iabs_k == ii) & (ld_k == b)]
            N_absphidHH[ii, b] = a.size
            if a.size >= min_cell_N:
                sigma_absphidHH[ii, b] = _circ_std(a)

    # Far-field (neighbour-absent) spread sigma_far from the exponential asymptote fit
    # of the phi-marginal sigma(dHH); the normaliser for the focus factor
    # f = sigma_A(dHH,|phi|)/sigma_far [FOCUS-RATIO].
    sigma_far, sigma_fit_popt = _fit_sigma_asymptote(dHH_centers, sigma, N)

    print('\n[SOCIAL_TRACK spread] within-condition turn std from dataset A ONLY '
          f'(r-edges from A; {n_r_bins}x{n_psi_bins}x{n_phi_bins} cells, '
          f'min_cell_N={min_cell_N}): '
          f'{int(np.sum(np.isfinite(sigma)))}/{n_dhh} dHH bins defined; '
          f'sigma_far={np.degrees(sigma_far):.1f} deg.')
    return {"dHH_centers": dHH_centers, "sigma_within": sigma,
            "sigma_within_rdHH": sigma_rdHH, "r_edges": r_edges,
            "sigma_within_absphidHH": sigma_absphidHH, "absphi_edges": absphi_edges,
            "sigma_far": sigma_far, "sigma_fit_popt": sigma_fit_popt,
            "N": N, "N_rdHH": N_rdHH, "N_absphidHH": N_absphidHH}




def estimate_social_blend_weight_vs_distance(
        datasets_A, radial_psi_bins, datasets_B=None, null_bouts=None, dHH_edges=None,
        magnitude_weighted=False, n_boot=300, min_N=8, min_delta_s=0.0,
        max_bout_speed_mm_s=None, max_bout_turn_angle_rad_s=None, fps=25.0,
        k_focus_dHH_threshold=None, k_focus_floor=0.7, target='tangential',
        labelA='real pairs (A)', labelB='time-shifted (B)',
        showLateralResolved = False,
        outputFileName1='social_blend_weight_vs_dHH.png',
        outputFileName2='social_blend_wexcess_vs_dHH.png',
        csvFileName='social_blend_weight_vs_dHH.csv',
        closeFigure=False, seed=0):
    """
    [SOCIAL-BLEND WEIGHT -- diagnostic] Data-driven estimate of the asocial/social
    mixing weight w(dHH) for 'turn_sampling_social_track', the principled replacement
    for the ad-hoc k_focus = clip(dHH/dHH_threshold, floor, 1.0).

    NOTE: This function uses "w" for the SOCIAL weight (the prefactor of z_track) and
    (1-w) for the intrinsic (asocial) weight.

    Model (per bout, on UNIT turn vectors z = exp(i*turn); turn in the -Delta_theta
    heading convention the sim uses):
        z_exp = (1 - w(dHH)) * z_int + w(dHH) * z_track
      z_exp   : observed pair turn, exp(i*(-Delta_theta)).
      z_int   : asocial expectation, exp(i*ti_mean(r,psi)) from radial_psi_bins
                (single-fish (r,psi) map, looked up at the bout's own r, psi). With
                magnitude_weighted=True, R_int*exp(i*ti_mean), R_int = exp(-ti_std^2/2)
                (accounts for asocial dispersion -- the rigorous mixture).
      z_track : the SOCIAL target, set by `target` (matching social_track_target):
                'tangential' -> exp(i*turn_track), turn_track = psi -
                (pi/2)*sign(sin(psi - phi)) (along-wall tracking); 'full' ->
                exp(i*phi), FACING the neighbour (radial + tangential). Use 'full' to
                get the weight CONSISTENT with social_track_target='full' -- the
                tangential projection under-reads a toward-neighbour (radial) turn.
    This is the mean-direction blend the sim's mu uses (w == 1 - k_focus): w weights
    the social (target) turn, 1 - w the asocial turn. NOTE: the intrinsic-free side-
    split diagnostic below stays a TANGENTIAL measure regardless of `target` (the radial
    component is side-independent, so a +/-side difference cancels it).

    w(dHH) marginalises over phi, psi (and r) by pooling all bouts in a dHH bin and
    solving the WEIGHTED LEAST-SQUARES projection (down-weights bouts where
    z_int ~ z_track, i.e. asocial and tracking coincide -- the otherwise 0/0 cells):
        w(dHH) = Re[ sum (z_exp - z_int) * conj(z_track - z_int) ]
                 / sum |z_track - z_int|^2
    If the mixture holds, w in [0, 1]. The leftover imaginary part of the numerator
    (reported as imag_frac = |Im num| / |num|) and any excursion of w outside [0, 1]
    flag model mismatch or a real phi/psi dependence the marginalisation hides.

    TIME-SHIFTED CONTROL. The ABSOLUTE w(dHH) is biased toward ~0.5 - 0.5*R when the
    neighbour is far (phi, hence z_track, becomes uninformative noise) -- so w does
    NOT go to 0 at large dHH even with no real interaction. Pass datasets_B (the
    time-shifted pairs) to get w_B(dHH) on the SAME geometric/noise baseline; the
    genuine social tracking is then the EXCESS w_A - w_B (real pairs assign more
    weight to tangential tracking, i.e. higher w). That difference -- peaking at
    contact, ~0 at large dHH -- is the parameter-free replacement for 1 - k_focus.

    A-only when datasets_B is None (needs just the minuend pair datasets and the
    single-fish radial_psi_bins).

    Inputs
    ------
    datasets_A : minuend pair (Nfish==2) datasets with IBI_properties.
    radial_psi_bins : single-fish (r,psi) map (build_radial_psi_bin_distributions),
        carrying per-bin "ti_mean"/"ti_std"/"N" and "r_edges"/"psi_edges".
    datasets_B : optional time-shifted (subtrahend) pair datasets for the control
        baseline and the social excess w_A - w_B. None -> A-only (no difference).
    null_bouts : optional PLUGGABLE null as a dict of per-bout arrays with keys
        "turn", "r", "psi", "phi", "dHH" (and optionally "delta_s"/"delta_t" for the
        physical filters) -- e.g. an asocial two-single-fish null (focal single-fish
        bouts paired with an independent partner's geometry). Takes precedence over
        datasets_B; flows through the SAME _arrays_to_D as A. NOTE for comparability:
        phi here must use the SAME convention as A's z_track (see caller) -- for a
        sim/data two-single-fish null, phi should be built as the toward-neighbour
        turn wrap(bearing_to_partner - heading) consistent with datasets_A's phi.
    dHH_edges : 1-D dHH bin EDGES (mm). Default np.linspace(0, 40, 14).
    magnitude_weighted : weight z_int by R_int = exp(-ti_std^2/2) (default False ->
        unit vectors, matching the sim's mu blend).
    n_boot : bootstrap resamples for the CI bands (0 to skip). A and B are resampled
        independently; the difference CI uses paired (per-iteration) resamples.
    min_N : minimum bouts in a dHH bin to estimate w (default 8).
    min_delta_s : keep only bouts with Delta_s_mm > this (mm, default 0.0 = all).
        Small bouts have a noise-dominated displacement direction (body barely turns,
        but -Delta_theta swings ~20-30 deg), which dilutes the social signal in the
        displacement-turn channel; raise (e.g. 2.0) to restrict to substantive bouts.
    k_focus_dHH_threshold, k_focus_floor : retained for call compatibility; no longer
        plotted (the 1 - k_focus overlay was removed).
    labelA, labelB : legend labels.
    showLateralResolved : if True, show in excess figure axial and lateral phi
    outputFileName1 : weight figure (w_A / w_B vs dHH). outputFileName2 : excess figure
        (w_excess vs dHH; only drawn when datasets_B / null_bouts given). csvFileName :
        CSV of dHH, w_A, w_B, w_excess, w_excess_lo, w_excess_hi (None to skip any).
    closeFigure, seed : plotting / RNG controls.

    ORIENTATION-RESOLVED CHECKS (when datasets_B given). To test whether the pooled
    w averages away a phi-structured weight (intrinsic z_int has no phi, while z_exp
    and z_track do), the excess is ALSO reported:
      - w_excess_lateral / w_excess_axial : the pooled excess restricted to lateral
        (|phi| in [45,135] deg) vs axial bouts -- a strong lateral tracking hidden by
        axial bouts would show as w_excess_lateral >> pooled.
      - tracking_excess_intrinsic_free : the INTRINSIC-FREE side-split estimate. Per
        coarse (r, psi) cell, [mean(+tangential turn component | neighbour on +side)
        - (| -side)] / 2 removes the asocial turn by construction (it is side-
        independent); w_if_A - w_if_B is then the undiluted excess tracking weight.
        NOTE: this estimate is ALWAYS TANGENTIAL and target-INDEPENDENT -- the
        radial/face-the-neighbour part of a 'full' turn is side-independent and
        CANCELS in the +/- side difference, and the reference (psi - pi/2) must stay
        side-independent for the asocial cancellation to hold. So it is a tangential
        cross-check regardless of `target`; the pooled/lateral/axial w_excess ARE
        target-aware (they use z_track = exp(i*phi) when target='full').
        [Commented out from plot]

    Returns
    -------
    dict with dHH_centers; for A: w_A, wA_lo, wA_hi, N_A, imag_frac_A; if datasets_B
    given, the same for B (w_B, wB_lo, wB_hi, N_B, imag_frac_B), the social excess
    w_excess = w_A - w_B with excess_lo/excess_hi (paired-bootstrap 95% CI), and the
    orientation-resolved checks w_excess_lateral, w_excess_axial, and
    tracking_excess_intrinsic_free (all "excess tracking weight", directly comparable).
    """
    rng = np.random.default_rng(seed)

    def _wrap(a):
        return (np.asarray(a, dtype=float) + np.pi) % (2.0*np.pi) - np.pi

    # Single-fish (r,psi) intrinsic mean/std grids for a vectorised per-bout lookup.
    bins = radial_psi_bins["bins"]
    r_edges_psi = np.asarray(radial_psi_bins["r_edges"], dtype=float)
    psi_edges = np.asarray(radial_psi_bins["psi_edges"], dtype=float)
    n_r = len(bins); n_psi = len(bins[0])
    ti_mean_grid = np.full((n_r, n_psi), np.nan)
    ti_std_grid = np.full((n_r, n_psi), np.nan)
    for i in range(n_r):
        for j in range(n_psi):
            b = bins[i][j]
            if b["N"] > 0:
                ti_mean_grid[i, j] = b.get("ti_mean", np.nan)
                ti_std_grid[i, j] = b.get("ti_std", np.nan)

    if dHH_edges is None:
        dHH_edges = np.linspace(0.0, 40.0, 14)
    dHH_edges = np.asarray(dHH_edges, dtype=float)
    n_dhh = len(dHH_edges) - 1
    dHH_centers = 0.5*(dHH_edges[:-1] + dHH_edges[1:])

    def _arrays_to_D(turn, r, psi, phi, dhh, dsz, dts):
        """Turn per-bout arrays into the projection dict D (z_exp, z_int, z_track, ld,
        phi, psi, r, turn). Shared by the real-data pooling (_prep) and any pluggable
        null passed as bout arrays (null_bouts), so both go through identical intrinsic
        lookup, target construction, and filtering."""
        turn = np.asarray(turn, dtype=float); r = np.asarray(r, dtype=float)
        psi = np.asarray(psi, dtype=float); phi = np.asarray(phi, dtype=float)
        dhh = np.asarray(dhh, dtype=float)
        dsz = np.asarray(dsz, dtype=float); dts = np.asarray(dts, dtype=float)
        ir = np.clip(np.digitize(r, r_edges_psi) - 1, 0, n_r - 1)
        jp = np.clip(np.digitize(_wrap(psi), psi_edges) - 1, 0, n_psi - 1)
        ti_mean = ti_mean_grid[ir, jp]
        ti_std = ti_std_grid[ir, jp]
        psi_tgt = np.where(np.sin(psi - phi) >= 0.0, np.pi/2.0, -np.pi/2.0)
        turn_track = psi - psi_tgt
        # Social target the projection resolves against (matches the sim's
        # social_track_target): 'tangential' -> along-wall turn_track = psi - psi_tgt;
        # 'full' -> FACE the neighbour, turn = phi (so z_track below = exp(i*phi), the
        # radial+tangential toward-neighbour direction). One variable turn_soc keeps
        # _curve/_w_of and the lateral-axial split unchanged.
        turn_soc = phi if target == 'full' else turn_track
        # min_delta_s: drop small bouts whose displacement DIRECTION is dominated by
        # tracking noise (the body barely reorients but -Delta_theta swings wildly);
        # they dilute the social signal in the displacement-turn channel.
        # max_bout_speed_mm_s: drop ID-swap tracking jumps (Delta_s/Delta_t too fast).
        # max_bout_turn_angle_rad_s: drop impossible large turns (|Delta_theta|>cap/fps).
        good = (np.isfinite(turn) & np.isfinite(ti_mean) & np.isfinite(turn_soc)
                & np.isfinite(dhh) & np.isfinite(dsz) & (dsz > min_delta_s)
                & _bout_speed_ok(dsz, dts, max_bout_speed_mm_s)
                & _bout_turn_ok(turn, max_bout_turn_angle_rad_s, fps))
        turn, dhh = turn[good], dhh[good]
        ti_mean, ti_std, turn_soc = ti_mean[good], ti_std[good], turn_soc[good]
        z_exp = np.exp(1j*turn)
        R_int = (np.exp(-0.5*np.where(np.isfinite(ti_std), ti_std, 0.0)**2)
                 if magnitude_weighted else 1.0)
        z_int = R_int*np.exp(1j*ti_mean)
        z_track = np.exp(1j*turn_soc)
        ld = np.digitize(dhh, dHH_edges) - 1
        phi = phi[good]; psi = psi[good]; r = r[good]
        return {"z_exp": z_exp, "z_int": z_int, "z_track": z_track, "ld": ld,
                "phi": phi, "psi": psi, "r": r, "turn": turn}

    def _prep(datasets):
        """Pool one pair dataset-list to per-bout arrays, then -> D via _arrays_to_D.
        Intrinsic from the (r,psi) grid, tracking from the stored geometry (phi, dHH)."""
        turn, r, psi, phi, dhh, dsz, dts = [], [], [], [], [], [], []
        for d in datasets:
            if d["Nfish"] != 2:
                raise ValueError('estimate_social_blend_weight_vs_distance needs '
                                 'pair (Nfish==2) data.')
            ip = d["IBI_properties"]
            for k in range(d["Nfish"]):
                turn.append(-np.asarray(ip["Delta_theta"][k], dtype=float))
                r.append(np.asarray(ip["r_mm_mean"][k], dtype=float))
                th = np.asarray(ip["theta"][k], dtype=float)
                gm = np.asarray(ip["gamma_mean"][k], dtype=float)
                psi.append(_wrap(th - gm))
                phi.append(np.asarray(ip["relative_orientation_mean"][k],
                                      dtype=float))
                dhh.append(np.asarray(ip["head_head_distance_mm_mean"][k],
                                      dtype=float))
                dsz.append(np.asarray(ip["Delta_s_mm"][k], dtype=float))
                dts.append(np.asarray(ip["Delta_t_s"][k], dtype=float))
        return _arrays_to_D(np.concatenate(turn), np.concatenate(r),
                            np.concatenate(psi), np.concatenate(phi),
                            np.concatenate(dhh), np.concatenate(dsz),
                            np.concatenate(dts))

    def _w_of(zx, zi, zt):
        """WLS-projection w (the SOCIAL weight, prefactor of z_track) from per-bout
        vectors; plus |Im num|/|num|. Projects (z_exp - z_int) onto (z_track - z_int),
        i.e. z_exp = (1 - w)*z_int + w*z_track. The denominator |z_track - z_int|^2 is
        unchanged in magnitude from the old (intrinsic-weight) form, so bouts where
        z_int ~ z_track (asocial and tracking coincide) are still down-weighted."""
        num = np.sum((zx - zi)*np.conj(zt - zi))
        den = np.sum(np.abs(zt - zi)**2)
        if den <= 0.0:
            return np.nan, np.nan
        w = float(np.real(num)/den)
        imf = float(np.abs(np.imag(num))/np.abs(num)) if np.abs(num) > 0 else np.nan
        return w, imf

    def _curve(D, sub=None):
        """Per-dHH-bin w, imag_frac, N, and a (n_boot x n_dhh) bootstrap array.
        `sub` is an optional boolean mask selecting a subset of bouts (e.g. lateral)."""
        z_exp, z_int, z_track, ld = D["z_exp"], D["z_int"], D["z_track"], D["ld"]
        if sub is not None:
            z_exp, z_int, z_track, ld = (z_exp[sub], z_int[sub], z_track[sub],
                                         ld[sub])
        w = np.full(n_dhh, np.nan); imf = np.full(n_dhh, np.nan)
        N = np.zeros(n_dhh, dtype=int)
        wb = np.full((max(n_boot, 1), n_dhh), np.nan)
        for b in range(n_dhh):
            m = np.where(ld == b)[0]
            N[b] = m.size
            if m.size < min_N:
                continue
            zx, zi, zt = z_exp[m], z_int[m], z_track[m]
            w[b], imf[b] = _w_of(zx, zi, zt)
            for t in range(n_boot):
                idx = rng.integers(0, m.size, m.size)
                wb[t, b] = _w_of(zx[idx], zi[idx], zt[idx])[0]
        return w, imf, N, wb

    # Coarse (r, psi) cells for the intrinsic-free side-split estimator (r-edges from
    # A so it is B-consistent). Within a cell the asocial part is ~constant, so the
    # +side vs -side difference of the +tangential turn component cancels it.
    def _cells(D, r_edges_c):
        irc = np.clip(np.digitize(D["r"], r_edges_c) - 1, 0, len(r_edges_c) - 2)
        jpc = np.clip(np.digitize(_wrap(D["psi"]), psi_edges_c) - 1, 0, n_psi_c - 1)
        return irc*n_psi_c + jpc

    def _intrinsic_free_tracking(D, r_edges_c, min_side=8):
        """Intrinsic-free tangential-tracking weight w_if(dHH) (the SOCIAL weight,
        computed directly without _w_of): per (r, psi) cell,
        [mean(+tangential turn component | neighbour on +side) - (| -side)] / 2, which
        removes the asocial turn (side-independent). Count-weighted pool over cells."""
        # +tangential turn component q = cos(turn - (psi - pi/2)); s = neighbour side.
        q = np.cos(D["turn"] - (D["psi"] - np.pi/2.0))
        s = np.where(np.sin(D["psi"] - D["phi"]) >= 0.0, 1.0, -1.0)
        cell = _cells(D, r_edges_c)
        ld = D["ld"]
        w_if = np.full(n_dhh, np.nan)
        for b in range(n_dhh):
            mb = (ld == b)
            num = 0.0; den = 0.0
            for c in np.unique(cell[mb]):
                inc = mb & (cell == c)
                qp = q[inc & (s > 0)]; qm = q[inc & (s < 0)]
                if qp.size >= min_side and qm.size >= min_side:
                    wgt = qp.size + qm.size
                    num += wgt*(qp.mean() - qm.mean())
                    den += wgt*2.0
            if den > 0:
                w_if[b] = num/den
        return w_if

    n_psi_c = 6
    psi_edges_c = np.linspace(-np.pi, np.pi, n_psi_c + 1)

    DA = _prep(datasets_A)
    r_edges_c = np.quantile(DA["r"], np.linspace(0.0, 1.0, 4))   # 3 r cells from A
    r_edges_c = np.maximum.accumulate(r_edges_c)
    for i in range(1, len(r_edges_c)):
        if r_edges_c[i] <= r_edges_c[i - 1]:
            r_edges_c[i] = np.nextafter(r_edges_c[i - 1], np.inf)

    wA, imfA, NA, wbA = _curve(DA)
    wA_lo = wA_hi = None
    if n_boot > 0:
        wA_lo = np.nanpercentile(wbA, 2.5, axis=0)
        wA_hi = np.nanpercentile(wbA, 97.5, axis=0)

    # Lateral (|phi| in [45, 135] deg) vs axial (neighbour ~ahead/behind) subsets:
    # tracking is only identifiable at lateral bearings, so a strong lateral excess
    # hidden by axial bouts would show here.
    def _lateral_mask(D):
        ap = np.abs(_wrap(D["phi"]))
        return (ap >= np.pi/4.0) & (ap <= 3.0*np.pi/4.0)

    latA = _lateral_mask(DA)
    wA_lat = _curve(DA, sub=latA)[0]
    wA_ax = _curve(DA, sub=~latA)[0]
    wA_if = _intrinsic_free_tracking(DA, r_edges_c)

    # The null (subtrahend) is either time-shifted PAIR datasets (datasets_B) or a
    # PLUGGABLE null passed as bout arrays (null_bouts: turn, r, psi, phi, dHH, and
    # optionally delta_s/delta_t for the filters) -- e.g. an asocial two-single-fish
    # null. null_bouts takes precedence. Both flow through _arrays_to_D, so w_B is
    # computed identically to w_A.
    has_B = (null_bouts is not None) or (datasets_B is not None)
    wB = imfB = NB = wbB = None
    wB_lo = wB_hi = w_excess = excess_lo = excess_hi = None
    w_excess_lat = w_excess_ax = tracking_excess_if = None
    if has_B:
        if null_bouts is not None:
            _nb = null_bouts
            _n = np.asarray(_nb["turn"], dtype=float).size
            DB = _arrays_to_D(
                _nb["turn"], _nb["r"], _nb["psi"], _nb["phi"], _nb["dHH"],
                _nb.get("delta_s", np.full(_n, np.inf)),
                _nb.get("delta_t", np.full(_n, np.nan)))
        else:
            DB = _prep(datasets_B)
        wB, imfB, NB, wbB = _curve(DB)
        w_excess = wA - wB
        latB = _lateral_mask(DB)
        w_excess_lat = wA_lat - _curve(DB, sub=latB)[0]
        w_excess_ax = wA_ax - _curve(DB, sub=~latB)[0]
        # Intrinsic-free EXCESS tracking weight = w_if_A - w_if_B (social weight).
        tracking_excess_if = wA_if - _intrinsic_free_tracking(DB, r_edges_c)
        if n_boot > 0:
            wB_lo = np.nanpercentile(wbB, 2.5, axis=0)
            wB_hi = np.nanpercentile(wbB, 97.5, axis=0)
            # Paired (per-iteration) difference of the independent A and B resamples.
            diff_boot = wbA - wbB
            excess_lo = np.nanpercentile(diff_boot, 2.5, axis=0)
            excess_hi = np.nanpercentile(diff_boot, 97.5, axis=0)

    # ---- report ----
    tag = ' (magnitude-weighted intrinsic)' if magnitude_weighted else ''
    print(f'\n[SOCIAL-BLEND WEIGHT] w(dHH) = {target} social weight (1 - w = asocial '
          f'intrinsic), WLS projection on unit turn vectors{tag}:')
    if has_B:
        print(f'  excess {target} tracking (positive = real pairs track more): pooled '
              f'w_A-w_B and lateral/axial |phi| split are {target}; excIF = '
              f'intrinsic-free side-split, ALWAYS TANGENTIAL (target-independent).')
        print(f'  {"dHH":>6} | {"w_A":>6} {"w_B":>6} | {"exc":>6} {"excLat":>7} '
              f'{"excAx":>6} {"excIF":>6}')
        for b in range(n_dhh):
            print(f'  {dHH_centers[b]:6.1f} | {wA[b]:6.2f} {wB[b]:6.2f} | '
                  f'{w_excess[b]:6.2f} {w_excess_lat[b]:7.2f} {w_excess_ax[b]:6.2f} '
                  f'{tracking_excess_if[b]:6.2f}')
    else:
        print(f'  {"dHH":>6} | {"w":>6} | {"imagfrac":>8} {"N":>7}')
        for b in range(n_dhh):
            print(f'  {dHH_centers[b]:6.1f} | {wA[b]:6.2f} | '
                  f'{imfA[b]:8.2f} {NA[b]:7d}')

    # ---- plot: two SEPARATE figures (weight; excess) ----
    def _band(axis, w, lo, hi, color, lab):
        m = np.isfinite(w)
        axis.plot(dHH_centers[m], w[m], '-o', color=color, label=lab)
        if lo is not None:
            mb = m & np.isfinite(lo) & np.isfinite(hi)
            if np.any(mb):
                axis.fill_between(dHH_centers[mb], lo[mb], hi[mb], color=color,
                                  alpha=0.2, linewidth=0)

    # Figure 1: the social weight w_A (and w_B) vs dHH.
    fig1, ax = plt.subplots(figsize=(8, 5.5))
    _band(ax, wA, wA_lo, wA_hi, 'C3', f'w_A ({labelA})')
    if has_B:
        _band(ax, wB, wB_lo, wB_hi, 'C0', f'w_B ({labelB})')
    ax.axhline(0.0, color='k', lw=0.6); ax.axhline(1.0, color='k', lw=0.6)
    ax.axhspan(0.0, 1.0, color='green', alpha=0.05)
    ax.set_xlabel('Inter-fish distance dHH (mm)')
    ax.set_ylabel(f'{target} social weight w (1 - w = asocial intrinsic)')
    ax.set_title('Data-driven social-blend weight w(dHH)\n'
                 f'exp turn = (1-w)*intrinsic + w*{target}-tracking (unit vectors)')
    ax.legend(fontsize=8)
    fig1.tight_layout()
    if outputFileName1 is not None:
        fig1.savefig(outputFileName1, dpi=150)
        print(f'[SOCIAL-BLEND WEIGHT] wrote {outputFileName1}')

    # Figure 2: the social EXCESS w_excess = w_A - w_B vs dHH (only with a B/null).
    fig2 = None
    if has_B:
        fig2, ax2 = plt.subplots(figsize=(8, 5.5))
        def _line(w, color, lab, style='-o'):
            m = np.isfinite(w)
            ax2.plot(dHH_centers[m], w[m], style, color=color, label=lab)
        me = np.isfinite(w_excess)
        ax2.plot(dHH_centers[me], w_excess[me], '-o', color='darkorange',
                 label='Excess w_A - w_B')
        if excess_lo is not None:
            mb = me & np.isfinite(excess_lo) & np.isfinite(excess_hi)
            if np.any(mb):
                ax2.fill_between(dHH_centers[mb], excess_lo[mb], excess_hi[mb],
                                 color='darkorange', alpha=0.2, linewidth=0,
                                 label='95% CI')
        if showLateralResolved:
            _line(w_excess_lat, 'violet', f'lateral |phi| in [45,135] deg ({target})', '-s')
            _line(w_excess_ax, 'gold', f'axial |phi| ({target})', '-D')
        #_line(tracking_excess_if, 'C4',
        #      'intrinsic-free side-split (TANGENTIAL, target-indep)', '--')
        ax2.axhline(0.0, color='k', lw=0.8)
        ax2.set_xlabel('Inter-fish distance dHH (mm)')
        ax2.set_ylabel(f'Social tracking weight')
        titleStr = 'Genuine social tracking weight'
        if showLateralResolved:
            titleStr = titleStr + f'\n vs lateral/axial ({target})'
        # f'vs intrinsic-free (tangential cross-check)'
        ax2.set_title(titleStr)
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        if outputFileName2 is not None:
            fig2.savefig(outputFileName2, dpi=150)
            print(f'[SOCIAL-BLEND WEIGHT] wrote {outputFileName2}')

    if closeFigure:
        plt.close(fig1)
        if fig2 is not None:
            plt.close(fig2)
    else:
        plt.show(block=False)

    # ---- CSV: dHH, w_A, w_B, w_excess, w_excess_lo, w_excess_hi ----
    if csvFileName is not None:
        def _col(a):
            return a if a is not None else np.full(n_dhh, np.nan)
        cols = [dHH_centers, wA, _col(wB), _col(w_excess),
                _col(excess_lo), _col(excess_hi)]
        with open(csvFileName, 'w', newline='') as _f:
            _wr = csv.writer(_f)
            _wr.writerow(['dHH_mm', 'w_A', 'w_B', 'w_excess',
                          'w_excess_lo', 'w_excess_hi'])
            for i in range(n_dhh):
                _wr.writerow([('' if not np.isfinite(c[i]) else f'{c[i]:.6g}')
                              for c in cols])
        print(f'[SOCIAL-BLEND WEIGHT] wrote {csvFileName}')

    out = {"dHH_centers": dHH_centers, "w_A": wA, "wA_lo": wA_lo, "wA_hi": wA_hi,
           "N_A": NA, "imag_frac_A": imfA}
    if has_B:
        out.update({"w_B": wB, "wB_lo": wB_lo, "wB_hi": wB_hi, "N_B": NB,
                    "imag_frac_B": imfB, "w_excess": w_excess,
                    "excess_lo": excess_lo, "excess_hi": excess_hi,
                    "w_excess_lateral": w_excess_lat, "w_excess_axial": w_excess_ax,
                    "tracking_excess_intrinsic_free": tracking_excess_if})
    return out




def build_real_paired_null_bouts(single_fish_datasets, rng, avoid_self=True):
    """Build an ASOCIAL turn-projection null from REAL single-fish data: each focal
    single-fish bout keeps its own (turn, r, psi, Delta_s, Delta_t) and is paired with
    an INDEPENDENT partner position drawn (by random permutation) from the pooled
    single-fish occupancy. phi is computed from the focal fish's BODY heading
    (IBI_properties["heading_angle_mean"]) via _relative_orientation_focal -- the SAME
    body-heading convention as A's stored relative_orientation_mean, so w_null is
    comparable to w_A. psi still uses the DISPLACEMENT direction theta (matching the
    projection's intrinsic lookup). No social content: the partner is independent.

    Returns a null_bouts dict (turn, r, psi, phi, dHH, delta_s, delta_t) for
    estimate_social_blend_weight_vs_distance(null_bouts=...), or None if no usable
    single-fish bouts. All physical filtering (min_delta_s, speed, turn caps) is left
    to the estimator, so it is applied identically to A and the null."""
    turn, r, psi, head, x, y, dsz, dts = [], [], [], [], [], [], [], []
    for ds in single_fish_datasets:
        ip = ds.get("IBI_properties")
        if ip is None or "heading_angle_mean" not in ip:
            continue
        for k in range(ds.get("Nfish", 1)):
            th = np.asarray(ip["theta"][k], dtype=float)
            gm = np.asarray(ip["gamma_mean"][k], dtype=float)
            rr = np.asarray(ip["r_mm_mean"][k], dtype=float)
            turn.append(-np.asarray(ip["Delta_theta"][k], dtype=float))
            r.append(rr)
            psi.append((th - gm + np.pi) % (2.0*np.pi) - np.pi)
            head.append(np.asarray(ip["heading_angle_mean"][k], dtype=float))
            x.append(rr*np.cos(gm)); y.append(rr*np.sin(gm))
            dsz.append(np.asarray(ip["Delta_s_mm"][k], dtype=float))
            dts.append(np.asarray(ip["Delta_t_s"][k], dtype=float))
    if not turn or sum(a.size for a in turn) == 0:
        return None
    turn = np.concatenate(turn); r = np.concatenate(r); psi = np.concatenate(psi)
    head = np.concatenate(head); x = np.concatenate(x); y = np.concatenate(y)
    dsz = np.concatenate(dsz); dts = np.concatenate(dts)
    n = turn.size
    # Independent partner = a random permutation of the occupancy pool (each position
    # used once). Fix any accidental self-pairing so no bout pairs with itself.
    perm = rng.permutation(n)
    if avoid_self:
        self_hit = np.where(perm == np.arange(n))[0]
        if self_hit.size:
            perm[self_hit] = perm[(self_hit + 1) % n]
    dx = x[perm] - x; dy = y[perm] - y
    dHH = np.hypot(dx, dy)
    phi = _relative_orientation_focal(head, dx, dy)
    return {"turn": turn, "r": r, "psi": psi, "phi": phi, "dHH": dHH,
            "delta_s": dsz, "delta_t": dts}


