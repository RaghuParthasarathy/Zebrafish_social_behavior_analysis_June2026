"""
IBI_diagnostics.py -- DATA-ONLY diagnostic plots for zebrafish inter-bout /
bout properties (turn spreads, kinematics-vs-dHH, circulation & approach,
p(dHH) real-vs-time-shifted / null comparisons, the social-blend weight
w_excess, etc.). Extracted from random_displacement_analysis.py (2026-07).

These operate on experimental datasets (single-fish or pair) and do NOT need
the simulation, so they can be run on any dataset (e.g. mutant fish). Simulated
trajectories, where a function overlays them, are passed IN as arguments -- this
module imports no simulation code. Sits on IBI_properties_utils; imported by
random_displacement_analysis for the pipeline runs (no import cycle).
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from IBI_properties_utils import (_good_frame_mask, _bout_speed_ok,
                                  _bout_turn_ok, _density_and_sem)


def diagnose_delta_theta_vs_heading_turn(datasets, exptName='',
                                         ds_split_mm=2.0, nbins=61,
                                         outputFileName=None, closeFigure=False):
    """
    [DIAGNOSTIC] Relationship between the two IBI turning-angle definitions and how
    it depends on bout size Delta_s:
      x = -Delta_theta        (change in DISPLACEMENT direction -- the sim's turn)
      y = turning_angle_IBI   (change in BODY HEADING)
    both in the same heading-turn sign convention (radians, wrapped to [-pi, pi]).

    Motivation: for small bouts the displacement DIRECTION is poorly defined (a tiny
    step's direction is mostly tracking noise), so -Delta_theta may decouple from the
    body-heading turn there while agreeing for substantive bouts. This quantifies that
    coupling for 'all bouts', 'Delta_s > ds_split_mm', and 'Delta_s <= ds_split_mm'.

    For each subset it computes and prints:
      - circular correlation r_c (Jammalamadaka-Sarma) between x and y,
      - mean alignment <cos(x - y)> (1 = identical turns, 0 = unrelated),
      - median |x| and |y| (deg),
    and plots the 2-D joint histogram of (x, y) per subset (log color), with the y=x
    diagonal. Works for single-fish (Nfish==1) or pair (Nfish==2) data (pools all
    fish/datasets); needs "Delta_theta", "turning_angle_IBI", "Delta_s_mm" in
    IBI_properties.

    Returns
    -------
    dict keyed 'all' / 'large' / 'small', each with r_circ, align, N, and the
    (x, y, ds) arrays for that subset (radians / mm).
    """
    deg = 180.0/np.pi

    def _wrap(a):
        return (np.asarray(a, dtype=float) + np.pi) % (2.0*np.pi) - np.pi

    def _circ_mean(a):
        return np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))

    def _circ_corr(a, b):
        # Jammalamadaka-Sarma circular correlation coefficient.
        a0 = a - _circ_mean(a); b0 = b - _circ_mean(b)
        sa, sb = np.sin(a0), np.sin(b0)
        den = np.sqrt(np.sum(sa**2)*np.sum(sb**2))
        return float(np.sum(sa*sb)/den) if den > 0 else np.nan

    # Pool x = -Delta_theta, y = turning_angle_IBI, ds = Delta_s_mm.
    x, y, ds = [], [], []
    for d in datasets:
        ip = d["IBI_properties"]
        for k in range(d["Nfish"]):
            x.append(-np.asarray(ip["Delta_theta"][k], dtype=float))
            y.append(np.asarray(ip["turning_angle_IBI"][k], dtype=float))
            ds.append(np.asarray(ip["Delta_s_mm"][k], dtype=float))
    x = _wrap(np.concatenate(x)); y = _wrap(np.concatenate(y))
    ds = np.concatenate(ds)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(ds)
    x, y, ds = x[ok], y[ok], ds[ok]

    subsets = {
        'all':   np.ones(x.shape, dtype=bool),
        'large': ds > ds_split_mm,
        'small': ds <= ds_split_mm,
    }
    labels = {'all': 'all bouts',
              'large': f'Delta_s > {ds_split_mm:g} mm',
              'small': f'Delta_s <= {ds_split_mm:g} mm'}

    out = {}
    print(f'\n[dTheta vs heading turn] {exptName}: x = -Delta_theta (displacement), '
          f'y = turning_angle_IBI (heading).')
    print(f'  {"subset":<16} {"N":>8} | {"r_circ":>7} {"<cos(x-y)>":>11} | '
          f'{"med|x|":>7} {"med|y|":>7} (deg)')
    for key in ('all', 'large', 'small'):
        m = subsets[key]
        xs, ys = x[m], y[m]
        rc = _circ_corr(xs, ys) if xs.size > 2 else np.nan
        al = float(np.mean(np.cos(xs - ys))) if xs.size else np.nan
        out[key] = {"r_circ": rc, "align": al, "N": int(xs.size),
                    "x": xs, "y": ys, "ds": ds[m]}
        print(f'  {labels[key]:<16} {xs.size:>8d} | {rc:>7.3f} {al:>11.3f} | '
              f'{np.median(np.abs(xs))*deg:>7.1f} {np.median(np.abs(ys))*deg:>7.1f}')

    # ---- plot: 2-D joint histogram per subset ----
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    edges = np.linspace(-180.0, 180.0, nbins)
    for ax, key in zip(axes, ('all', 'large', 'small')):
        xs, ys = out[key]["x"]*deg, out[key]["y"]*deg
        h = ax.hist2d(xs, ys, bins=[edges, edges],
                      norm=LogNorm(), cmap='magma')[3]
        ax.plot([-180, 180], [-180, 180], '--', color='cyan', lw=1)
        ax.set_xlabel('-Delta_theta (displacement turn, deg)')
        ax.set_ylabel('turning_angle_IBI (heading turn, deg)')
        ax.set_title(f'{labels[key]}\nr_circ={out[key]["r_circ"]:.3f}, '
                     f'<cos(x-y)>={out[key]["align"]:.3f}, N={out[key]["N"]}',
                     fontsize=10)
        ax.set_aspect('equal')
        fig.colorbar(h, ax=ax, label='count')
    fig.suptitle(f'{exptName}: displacement turn vs body-heading turn', fontsize=12)
    if outputFileName is not None:
        fig.savefig(outputFileName, dpi=140)
        print(f'[dTheta vs heading turn] wrote {outputFileName}')
    if closeFigure:
        plt.close(fig)
    else:
        plt.show(block=False)

    return out


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


def phi_resolved_turn_std_vs_distance(datasets_A, dHH_edges=None,
                                      n_r_bins=3, n_psi_bins=4, n_phi_bins=8,
                                      min_cell_N=8, min_delta_s=1.0,
                                      max_bout_speed_mm_s=None,
                                      max_bout_turn_angle_rad_s=None, fps=25.0,
                                      make_line_plots=False, heatmap_clim=None,
                                      outputFileName='turn_std_phi_dHH.png',
                                      closeFigure=False):
    """
    [phi-RESOLVED social focus spread -- diagnostic] Within-condition circular std of
    the inter-bout turn (-Delta_theta) resolved by BOTH inter-fish distance dHH AND
    relative orientation phi. This is the phi-aware companion to
    compute_within_condition_turn_std: identical residualization (the
    (r, psi, phi, dHH)-cell circular mean is conditioned out, so mean structure is
    removed and only the SPREAD remains), but the residuals are then collapsed over
    (phi, dHH) rather than (r, dHH). It answers the question the dHH-only focus spread
    cannot: at a fixed neighbour distance, does the fish narrow its turning MORE for
    some relative orientations than others (e.g. neighbour forward [|phi|<45 deg],
    beside [lateral, 45-135 deg], or behind [|phi|>135 deg])?

    Convention note: phi here is relative_orientation (theta - beta), matching the
    lateral/axial split used in estimate_social_blend_weight_vs_distance, where
    'lateral' is |phi| in [45, 135] deg and 'axial' is the complement.

    Inputs mirror compute_within_condition_turn_std (n_phi_bins=8 -> 45-deg phi bins).
    min_delta_s (mm, default 1.0) drops small, noise-dominated bouts. Produces the
    signed sigma(phi, dHH) heatmap (degrees, 45-deg phi bins) as its OWN figure
    (saved to outputFileName). If make_line_plots is True, ALSO makes a separate
    2-panel LINE figure -- (top) sigma vs dHH for the forward / lateral / behind
    folded-|phi| groups plus the phi-marginal, (middle) the focus factor f -- saved to
    outputFileName with "_line" inserted before the extension. heatmap_clim = (vmin,
    vmax) in DEGREES fixes the heatmap colour scale; None -> autoscale.

    Returns
    -------
    dict with dHH_centers (mm), phi_centers (rad), phi_edges (rad),
    sigma_within_phidHH (n_phi x n_dhh, rad; NaN where under-populated),
    N_phidHH (n_phi x n_dhh), sigma_within (n_dhh, the phi-marginal), N_marg (n_dhh),
    the folded-group spreads sigma_forward/lateral/behind (n_dhh), the far-field
    normaliser sigma_far (rad, the asymptote of the exponential fit to the marginal;
    fit_popt = [sigma_far, sigma_0, lambda] or None), and the focus factor
    f = sigma / sigma_far for the marginal (f_marg) and folded groups
    (f_forward/lateral/behind). For the simulation (if wired into
    turn_sampling_social_focus): social_focus_sigma_phi = (dHH_centers, phi_edges,
    sigma_within_phidHH).
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

    # Pool A's per-IBI (turn, r, psi, phi, dHH, Delta_s, Delta_t), as the A-only spread.
    turn, r, psi, phi, dhh, dsz, dts = [], [], [], [], [], [], []
    for d in datasets_A:
        if d["Nfish"] != 2:
            raise ValueError('phi_resolved_turn_std_vs_distance needs pair '
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
    ok = (np.isfinite(turn) & np.isfinite(r) & np.isfinite(psi)
          & np.isfinite(phi) & np.isfinite(dhh) & np.isfinite(dsz)
          & (dsz > min_delta_s) & _bout_speed_ok(dsz, dts, max_bout_speed_mm_s)
          & _bout_turn_ok(turn, max_bout_turn_angle_rad_s, fps))
    turn, r, psi, phi, dhh = turn[ok], r[ok], psi[ok], phi[ok], dhh[ok]
    phi = _wrap(phi)

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
    phi_centers = 0.5*(phi_edges[:-1] + phi_edges[1:])

    # Residual about each (r, psi, phi, dHH) cell circular mean (same as A-only spread).
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
    resid_k, ld_k, kp_k = resid[keep], ld[keep], kp[keep]

    # Collapse the residual spread over (phi, dHH) [2-D] and over dHH alone [marginal].
    sigma = np.full(n_dhh, np.nan)
    for b in range(n_dhh):
        a = resid_k[ld_k == b]
        if a.size >= min_cell_N:
            sigma[b] = _circ_std(a)
    sigma_phidHH = np.full((n_phi_bins, n_dhh), np.nan)
    N_phidHH = np.zeros((n_phi_bins, n_dhh), dtype=int)
    for kk in range(n_phi_bins):
        for b in range(n_dhh):
            a = resid_k[(kp_k == kk) & (ld_k == b)]
            N_phidHH[kk, b] = a.size
            if a.size >= min_cell_N:
                sigma_phidHH[kk, b] = _circ_std(a)

    # Forward (|phi| < 45 deg), lateral (45-135 deg), behind (|phi| > 135 deg) groups
    # (folded in the sign of phi).
    ap = np.abs(_wrap(phi[keep]))
    fwd = ap < np.pi/4.0
    lat = (ap >= np.pi/4.0) & (ap <= 3.0*np.pi/4.0)
    beh = ap > 3.0*np.pi/4.0
    sig_fwd = np.full(n_dhh, np.nan); sig_lat = np.full(n_dhh, np.nan)
    sig_beh = np.full(n_dhh, np.nan)
    for b in range(n_dhh):
        for grp, out in ((fwd, sig_fwd), (lat, sig_lat), (beh, sig_beh)):
            a = resid_k[(ld_k == b) & grp]
            if a.size >= min_cell_N:
                out[b] = _circ_std(a)

    # Far-field (neighbour-absent) spread sigma_far: the asymptote of the exponential
    # approach fit to the phi-MARGINAL sigma(dHH) (high-count -> low-variance). It is
    # the single normaliser for the focus factor f(dHH,|phi|) = sigma_A/sigma_far
    # (-> 1 far away, < 1 where pairs focus, > 1 in close-range jockeying). Same
    # _fit_sigma_asymptote used by compute_within_condition_turn_std for the sim, so the
    # diagnostic and the model share one sigma_far.
    N_marg = np.array([int(np.sum(ld_k == b)) for b in range(n_dhh)])
    sigma_far, fit_popt = _fit_sigma_asymptote(dHH_centers, sigma, N_marg)
    # Focus factor f = sigma / sigma_far (marginal and folded-|phi| groups).
    f_marg = sigma/sigma_far
    f_fwd = sig_fwd/sigma_far; f_lat = sig_lat/sigma_far; f_beh = sig_beh/sigma_far

    print('\n[phi-RESOLVED spread] within-condition turn std from dataset A ONLY, '
          f'resolved by (phi, dHH) ({n_r_bins}x{n_psi_bins}x{n_phi_bins} cells, '
          f'min_cell_N={min_cell_N}, min_delta_s={min_delta_s:g} mm):')
    _lam = fit_popt[2] if fit_popt is not None else np.nan
    print(f'  sigma_far (asymptote) = {np.degrees(sigma_far):.1f} deg'
          + (f' (fit: sigma_0={np.degrees(fit_popt[1]):.1f} deg, '
             f'lambda={_lam:.1f} mm)' if fit_popt is not None
             else ' (plateau mean, fit failed)'))
    print(f'  {"dHH":>6} | {"marg":>6} {"fwd":>6} {"lat":>6} {"behind":>6}  (deg) | '
          f'{"f_marg":>6}')
    for b in range(n_dhh):
        print(f'  {dHH_centers[b]:6.1f} | {np.degrees(sigma[b]):6.1f} '
              f'{np.degrees(sig_fwd[b]):6.1f} {np.degrees(sig_lat[b]):6.1f} '
              f'{np.degrees(sig_beh[b]):6.1f} | {f_marg[b]:6.2f}')

    # ---- plot ----
    _groups = ((sig_fwd, f_fwd, 'C2', '-s', 'forward (|phi|<45 deg)'),
               (sig_lat, f_lat, 'C0', '-^', 'lateral (45-135 deg)'),
               (sig_beh, f_beh, 'C3', '-v', 'behind (|phi|>135 deg)'))
    # LINE figure (optional): sigma vs dHH (top) + focus factor f (bottom).
    if make_line_plots:
        figL, (ax, axf) = plt.subplots(2, 1, figsize=(8, 9))
        # (top) sigma vs dHH: marginal + groups + fitted approach curve + sigma_far.
        m = np.isfinite(sigma)
        ax.plot(dHH_centers[m], np.degrees(sigma[m]), '-o', color='k',
                lw=2, label='phi-marginal')
        for sg, _fg, col, mk, lab in _groups:
            mm = np.isfinite(sg)
            ax.plot(dHH_centers[mm], np.degrees(sg[mm]), mk, color=col, label=lab)
        if fit_popt is not None:
            _xg = np.linspace(dHH_centers[0], dHH_centers[-1], 200)
            ax.plot(_xg, np.degrees(_sigma_approach(_xg, *fit_popt)), '--',
                    color='0.5', lw=1.5, label='marginal fit')
        ax.axhline(np.degrees(sigma_far), color='0.5', ls=':', lw=1.2,
                   label=f'sigma_far = {np.degrees(sigma_far):.0f} deg')
        ax.set_ylabel('within-condition turn std (deg)')
        ax.set_title('phi-resolved turn spread vs inter-fish distance (dataset A)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        # (bottom) focus factor f = sigma / sigma_far.
        axf.axhline(1.0, color='0.5', ls=':', lw=1.2)
        mm = np.isfinite(f_marg)
        axf.plot(dHH_centers[mm], f_marg[mm], '-o', color='k', lw=2,
                 label='phi-marginal')
        for _sg, fg, col, mk, lab in _groups:
            mm = np.isfinite(fg)
            axf.plot(dHH_centers[mm], fg[mm], mk, color=col, label=lab)
        axf.set_ylabel('focus factor  f = sigma / sigma_far')
        axf.set_xlabel('inter-fish distance dHH (mm)')
        axf.legend(fontsize=8); axf.grid(alpha=0.3)
        figL.tight_layout()
        if outputFileName:
            _base, _ext = os.path.splitext(outputFileName)
            lineFileName = f'{_base}_line{_ext}'
            figL.savefig(lineFileName, dpi=150, bbox_inches='tight')
            print(f'[phi-RESOLVED spread] saved {lineFileName}')
        if closeFigure:
            plt.close(figL)

    # HEATMAP figure (own window): full signed sigma(phi, dHH). heatmap_clim=(vmin,
    # vmax) in DEGREES fixes the colour scale; None -> autoscale.
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    _clim = ({} if heatmap_clim is None
             else {'vmin': heatmap_clim[0], 'vmax': heatmap_clim[1]})
    im = ax2.imshow(np.degrees(sigma_phidHH), origin='lower', aspect='auto',
                    extent=[dHH_edges[0], dHH_edges[-1],
                            np.degrees(phi_edges[0]), np.degrees(phi_edges[-1])],
                    cmap='viridis', **_clim)
    ax2.set_xlabel('Inter-fish distance dHH (mm)')
    ax2.set_ylabel(r'Relative orientation $\phi$ (deg)')
    ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    fig2.colorbar(im, ax=ax2, label='Turn std (deg)')
    fig2.tight_layout()
    if outputFileName:
        fig2.savefig(outputFileName, dpi=150, bbox_inches='tight')
        print(f'[phi-RESOLVED spread] saved {outputFileName}')
    if closeFigure:
        plt.close(fig2)

    return {"dHH_centers": dHH_centers, "phi_centers": phi_centers,
            "phi_edges": phi_edges, "sigma_within_phidHH": sigma_phidHH,
            "N_phidHH": N_phidHH, "sigma_within": sigma, "N_marg": N_marg,
            "sigma_forward": sig_fwd, "sigma_lateral": sig_lat,
            "sigma_behind": sig_beh, "sigma_far": sigma_far, "fit_popt": fit_popt,
            "f_marg": f_marg, "f_forward": f_fwd, "f_lateral": f_lat,
            "f_behind": f_beh}


def compare_pair_turn_std_vs_distance(
        datasets_A, datasets_B,
        dHH_edges=None,
        n_r_bins=3, n_psi_bins=4, n_phi_bins=6,
        min_cell_N=8, n_boot=300, min_delta_s=0.0, max_bout_speed_mm_s=None,
        max_bout_turn_angle_rad_s=None, fps=25.0,
        labelA='pairs (light)', labelB='pairs (light, time-shifted)',
        outputFileName='turn_std_vs_dHH.png', closeFigure=False, seed=0):
    """
    [PRECISION FEATURE -- diagnostic.] Compare the WITHIN-CONDITION spread (circular
    std) of the bout-to-bout turn (-Delta_theta) as a function of inter-fish distance
    dHH, between a minuend pair dataset A (real interacting pairs, e.g. pairs in
    light) and a subtrahend pair dataset B (the asocial control, e.g. the
    time-shifted pairs). This is the companion to diagnose_intrinsic_vs_social_precision
    for the empirical question "is the social signal a CHANGE in the turn spread?"
    -- so it is written separately because it needs the raw A/B datasets, not the
    pre-binned arrays that function takes.

    Why not just std(Delta_theta | dHH)?
    ------------------------------------
    The mean turn varies strongly with the neighbour bearing phi and the wall
    alignment psi (that variation IS the social/wall steering signal). Pooling the
    raw turns over r, psi, phi at a given dHH folds the BETWEEN-cell spread of those
    means into the "std", so a naive marginal std mixes mean-structure with genuine
    spread. Since the social effect of interest lives in the turn VARIANCE at fixed
    bearing (not in the mean), the mean structure must be held fixed before the
    spread is measured.

    Method (marginalize the SPREAD, not the angles)
    -----------------------------------------------
    1. Pool per-IBI (turn = -Delta_theta, r = r_mm_mean, psi = wrap(theta-gamma_mean),
       phi = relative_orientation_mean, dHH = head_head_distance_mm_mean) for A and B.
    2. Build a SHARED coarse nuisance grid (r, psi, phi) -- identical edges for A and
       B so the comparison is apples-to-apples. r edges are quantiles of the pooled
       A+B r (equal occupancy); psi, phi are linear over [-pi, pi].
    3. Within each (r, psi, phi, dHH) cell (per dataset, cells with >= min_cell_N
       steps) take the circular mean turn and form each step's angular RESIDUAL about
       its own cell mean. This removes the (r, psi, phi)- and dHH-dependent mean.
    4. For each dHH bin, sigma_within(dHH) = circular std of the pooled residuals
       across all cells. Crucially this never needs a single 4-D cell to be well
       populated; only the 1-D collapse over dHH must be.
    5. A percentile bootstrap (resampling residuals within each dHH bin) gives a CI
       band. The naive marginal std(turn | dHH) is also computed and drawn dashed,
       to expose how much the mean structure inflates it.

    Inputs
    ------
    datasets_A, datasets_B : lists of pair (Nfish==2) dataset dicts with
        "IBI_properties" carrying per-fish ragged "Delta_theta", "r_mm_mean",
        "theta", "gamma_mean", "relative_orientation_mean",
        "head_head_distance_mm_mean".
    dHH_edges : 1-D dHH bin EDGES (mm). Default np.linspace(0, 40, 14) (~3 mm bins).
    n_r_bins, n_psi_bins, n_phi_bins : coarse nuisance-grid resolution. Keep coarse
        so each (r, psi, phi, dHH) cell clears min_cell_N.
    min_cell_N : minimum steps in a cell for its residuals to be used (default 8).
    n_boot : bootstrap resamples for the CI band (default 300; 0 to skip).
    labelA, labelB : legend labels.
    outputFileName : figure filename (None to skip saving).
    closeFigure : close the figure after saving.
    seed : RNG seed for the bootstrap.

    Returns
    -------
    dict with dHH_centers, sigma_within_A/B (rad), sigma_within_lo/hi_A/B (rad CI),
    sigma_naive_A/B (rad), sigma_ratio (= sigma_within_A/sigma_within_B), the per-bin
    residual counts N_A/N_B, and -- for the r-resolved spread --
    sigma_within_A_rdHH (n_r_bins x n_dHH, rad; NaN where under-populated), its counts
    N_A_rdHH, the coarse r_edges (mm), and n_r_bins. The (r, dHH) map is what
    social_track uses when binning the data-driven spread by r as well as dHH.
    """
    deg = 180.0/np.pi
    rng = np.random.default_rng(seed)

    def _wrap(a):
        return (np.asarray(a, dtype=float) + np.pi) % (2.0*np.pi) - np.pi

    def _circ_mean(a):
        return np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))

    def _circ_std(a):
        if a.size == 0:
            return np.nan
        R = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
        return np.sqrt(-2.0*np.log(R)) if R > 0.0 else np.inf

    def _pool(datasets):
        turn, r, psi, phi, dhh, dsz, dts = [], [], [], [], [], [], []
        for d in datasets:
            if d["Nfish"] != 2:
                raise ValueError('compare_pair_turn_std_vs_distance needs pair '
                                 '(Nfish==2) data.')
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
        turn = np.concatenate(turn); r = np.concatenate(r)
        psi = np.concatenate(psi); phi = np.concatenate(phi)
        dhh = np.concatenate(dhh); dsz = np.concatenate(dsz); dts = np.concatenate(dts)
        # min_delta_s: drop small bouts (noise-dominated displacement direction).
        # max_bout_speed_mm_s: drop ID-swap tracking jumps (Delta_s/Delta_t too fast).
        # max_bout_turn_angle_rad_s: drop impossible large turns (|Delta_theta|>cap/fps).
        ok = (np.isfinite(turn) & np.isfinite(r) & np.isfinite(psi)
              & np.isfinite(phi) & np.isfinite(dhh) & np.isfinite(dsz)
              & (dsz > min_delta_s) & _bout_speed_ok(dsz, dts, max_bout_speed_mm_s)
              & _bout_turn_ok(turn, max_bout_turn_angle_rad_s, fps))
        return turn[ok], r[ok], psi[ok], phi[ok], dhh[ok]

    turnA, rA, psiA, phiA, dhhA = _pool(datasets_A)
    turnB, rB, psiB, phiB, dhhB = _pool(datasets_B)

    # Shared bin edges (A and B identical). r edges = quantiles of pooled r;
    # psi, phi linear over [-pi, pi]; dHH from the argument / default.
    if dHH_edges is None:
        dHH_edges = np.linspace(0.0, 40.0, 14)
    dHH_edges = np.asarray(dHH_edges, dtype=float)
    n_dhh = len(dHH_edges) - 1
    dHH_centers = 0.5*(dHH_edges[:-1] + dHH_edges[1:])
    r_pool = np.concatenate([rA, rB])
    r_edges = np.quantile(r_pool, np.linspace(0.0, 1.0, n_r_bins + 1))
    r_edges = np.maximum.accumulate(r_edges)
    for i in range(1, len(r_edges)):
        if r_edges[i] <= r_edges[i - 1]:
            r_edges[i] = np.nextafter(r_edges[i - 1], np.inf)
    psi_edges = np.linspace(-np.pi, np.pi, n_psi_bins + 1)
    phi_edges = np.linspace(-np.pi, np.pi, n_phi_bins + 1)

    def _residuals_and_dhhbin(turn, r, psi, phi, dhh):
        """Angular residual of each step about its (r,psi,phi,dHH) cell circular
        mean, plus the step's dHH-bin index and coarse r-bin index. Steps in cells
        with < min_cell_N (or outside the dHH range) are dropped. Returns
        (residuals, dHH_bin_index, r_bin_index)."""
        ir = np.clip(np.digitize(r, r_edges) - 1, 0, n_r_bins - 1)
        jp = np.clip(np.digitize(psi, psi_edges) - 1, 0, n_psi_bins - 1)
        kp = np.clip(np.digitize(phi, phi_edges) - 1, 0, n_phi_bins - 1)
        # dHH bin: -1 / n_dhh outside the requested range -> dropped below.
        ld = np.digitize(dhh, dHH_edges) - 1
        in_dhh = (ld >= 0) & (ld < n_dhh)
        cid = (((ir*n_psi_bins + jp)*n_phi_bins + kp)*n_dhh + ld)
        cid = np.where(in_dhh, cid, -1)          # mark out-of-range cells
        resid = np.full(turn.shape, np.nan)
        order = np.argsort(cid, kind='stable')
        cid_s = cid[order]
        uniq, start = np.unique(cid_s, return_index=True)
        start = np.append(start, len(cid_s))
        for m in range(len(uniq)):
            if uniq[m] < 0:                       # out-of-dHH-range group
                continue
            idx = order[start[m]:start[m + 1]]
            if idx.size >= min_cell_N:
                resid[idx] = _wrap(turn[idx] - _circ_mean(turn[idx]))
        keep = np.isfinite(resid)
        return resid[keep], ld[keep], ir[keep]

    residA, ldA, irA = _residuals_and_dhhbin(turnA, rA, psiA, phiA, dhhA)
    residB, ldB, irB = _residuals_and_dhhbin(turnB, rB, psiB, phiB, dhhB)

    def _curve(resid, ld):
        sig = np.full(n_dhh, np.nan)
        lo = np.full(n_dhh, np.nan)
        hi = np.full(n_dhh, np.nan)
        N = np.zeros(n_dhh, dtype=int)
        for b in range(n_dhh):
            a = resid[ld == b]
            N[b] = a.size
            if a.size >= min_cell_N:
                sig[b] = _circ_std(a)
                if n_boot > 0:
                    bs = np.array([_circ_std(a[rng.integers(0, a.size, a.size)])
                                   for _ in range(n_boot)])
                    bs = bs[np.isfinite(bs)]
                    if bs.size:
                        lo[b], hi[b] = np.percentile(bs, [2.5, 97.5])
        return sig, lo, hi, N

    sigA, loA, hiA, NA = _curve(residA, ldA)
    sigB, loB, hiB, NB = _curve(residB, ldB)

    # r-RESOLVED real-pair spread sigma_within_A(r, dHH): same residuals, but kept
    # split by the coarse r bin instead of collapsed over it. This lets social_track
    # use a wall-aware spread (smaller near the wall, where fish glide) instead of
    # the r-marginal sigA, so reproducing p(dHH) need not cost p(r). NaN where a
    # (r, dHH) cell has < min_cell_N residuals; the sim falls back along dHH there.
    sigA_rdHH = np.full((n_r_bins, n_dhh), np.nan)
    NA_rdHH = np.zeros((n_r_bins, n_dhh), dtype=int)
    for ii in range(n_r_bins):
        for b in range(n_dhh):
            a = residA[(irA == ii) & (ldA == b)]
            NA_rdHH[ii, b] = a.size
            if a.size >= min_cell_N:
                sigA_rdHH[ii, b] = _circ_std(a)

    def _naive(turn, dhh):
        ld = np.digitize(dhh, dHH_edges) - 1
        sig = np.full(n_dhh, np.nan)
        for b in range(n_dhh):
            a = turn[ld == b]
            if a.size >= min_cell_N:
                sig[b] = _circ_std(a)
        return sig
    naiveA = _naive(turnA, dhhA)
    naiveB = _naive(turnB, dhhB)

    # ---- report ----
    print('\n[TURN-STD vs dHH] within-condition turn spread (circular std, deg), '
          f'conditioning out (r x psi x phi) = {n_r_bins}x{n_psi_bins}x{n_phi_bins} '
          f'cells, min_cell_N={min_cell_N}:')
    print(f'  {"dHH":>6} | {"sig_A":>7} {"sig_B":>7} {"A-B":>7} | '
          f'{"N_A":>7} {"N_B":>7}')
    for b in range(n_dhh):
        d = sigA[b] - sigB[b]
        print(f'  {dHH_centers[b]:6.1f} | '
              f'{sigA[b]*deg:7.1f} {sigB[b]*deg:7.1f} '
              f'{d*deg if np.isfinite(d) else float("nan"):7.1f} | '
              f'{NA[b]:7d} {NB[b]:7d}')

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for sig, lo, hi, naive, color, lab in (
            (sigA, loA, hiA, naiveA, 'C0', labelA),
            (sigB, loB, hiB, naiveB, 'C1', labelB)):
        m = np.isfinite(sig)
        ax.plot(dHH_centers[m], sig[m]*deg, '-o', color=color,
                label=f'{lab} (within-condition)')
        mb = m & np.isfinite(lo) & np.isfinite(hi)
        if np.any(mb):
            ax.fill_between(dHH_centers[mb], lo[mb]*deg, hi[mb]*deg,
                            color=color, alpha=0.2, linewidth=0)
        mn = np.isfinite(naive)
        ax.plot(dHH_centers[mn], naive[mn]*deg, '--', color=color, alpha=0.6,
                label=f'{lab} (naive marginal)')
    ax.set_xlabel('head-head distance dHH (mm)')
    ax.set_ylabel('turn-angle spread, circular std (deg)')
    ax.set_title('Bout-to-bout turn spread vs inter-fish distance\n'
                 '(mean structure in r, psi, phi conditioned out)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    if outputFileName is not None:
        fig.savefig(outputFileName, dpi=150)
        print(f'[TURN-STD vs dHH] wrote {outputFileName}')
    if closeFigure:
        plt.close(fig)
    else:
        plt.show(block=False)

    # Focusing factor rho(dHH) = sigma_within_A / sigma_within_B (real pairs over the
    # time-shifted control): the empirically observed turn-spread narrowing, directly
    # usable as the spread multiplier in turn_sampling_social_focus (via
    # social_focus_sigma_ratio). NaN where either curve is undefined.
    with np.errstate(invalid='ignore', divide='ignore'):
        sigma_ratio = sigA / sigB
    return {"dHH_centers": dHH_centers,
            "sigma_within_A": sigA, "sigma_within_B": sigB,
            "sigma_within_lo_A": loA, "sigma_within_hi_A": hiA,
            "sigma_within_lo_B": loB, "sigma_within_hi_B": hiB,
            "sigma_naive_A": naiveA, "sigma_naive_B": naiveB,
            "sigma_ratio": sigma_ratio, "N_A": NA, "N_B": NB,
            "sigma_within_A_rdHH": sigA_rdHH, "N_A_rdHH": NA_rdHH,
            "r_edges": r_edges, "n_r_bins": n_r_bins}


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


def diagnose_pair_circulation_and_approach(
        datasets, fps=25.0, arena_radius_mm=None, dHH_edges=None,
        dHH_bin_mm=2.0, sim_datasets=None, circ_window_s=1.0,
        label_data='real pairs', label_sim='simulated pairs',
        outputFileName='pair_circulation_approach.png',
        closeFigure=False):
    """
    [DIAGNOSTIC] Test, in the EXPERIMENTAL pair data, the two long-range coupling
    signatures the w_excess turn estimator cannot see -- the ones that would break
    (or confirm) the simulated antipodal co-rotating "trap" behind the spurious
    40-45 mm p(dHH) peak:

      1. CIRCULATION CORRELATION c(dHH) = < sign(gamma_dot_0) * sign(gamma_dot_1) >,
         binned by inter-fish distance. c = +1 means the two fish co-rotate (same
         CW/CCW sense along the wall -> phase-locked, few encounters); c = -1 means
         they counter-rotate (sweep through encounters). A REAL pair that arranges to
         meet should sit BELOW the time-shifted control (more negative c), especially
         at large dHH -- a coupling that is NOT a turn toward the neighbour and so is
         invisible to w_excess.
      2. APPROACH DRIFT < d(dHH)/dt | dHH > (mm/s), binned by dHH. A genuine long-
         range attraction shows up as a NEGATIVE drift at large dHH for real pairs
         relative to the control, even when no single bout "turns toward" the other --
         the integrated, non-turn approach signal w_excess misses.

    Each quantity is shown for the real pairing and an internally-computed TIME-
    SHIFTED control (fish 1 rolled by half the movie), the same null the social-blend
    weight uses. If sim_datasets is given (dataset-like dicts built from simulated
    trajectories, e.g. simulate_pair_dHH_trials(..., collect_trajectories=True)), the
    SIMULATED curves are overlaid so the sim's circulation lock can be compared to the
    data directly. Frame-level; needs per-fish "radial_position_mm" and
    "polar_angle_rad" (Nframes x 2) and, optionally, "frameArray"/"bad_bodyTrack_frames"
    for good-frame masking. Curves are the mean over datasets/trials with a +/- s.e.m.
    band across datasets/trials.

    Inputs
    ------
    datasets : list of pair (Nfish==2) dataset dicts (real pairs; e.g. pairstats_2).
    fps : frame rate (Hz), for the mm/s approach rate.
    arena_radius_mm : arena radius (mm); sets the default dHH range [0, 2*R]. If None,
        the range is taken from the data.
    dHH_edges : explicit dHH bin edges (mm); overrides arena_radius_mm/dHH_bin_mm.
    dHH_bin_mm : dHH bin width (mm) for the default edges.
    sim_datasets : optional list of simulated dataset-like dicts (same frame-level
        fields) to overlay; None -> data only.
    circ_window_s : window (s) over which the circulation sign and dHH change are
        measured (sign(gamma[t+W]-gamma[t]) etc., W = circ_window_s*fps), applied
        identically to data and sim. ~1 IBI (default 1.0 s) is ESSENTIAL for the
        simulated trajectories, whose nearest-neighbour (step) interpolation makes the
        per-frame dgamma sparse; it also puts data and sim on the same bout scale. Set
        near 1/fps to recover the raw per-frame difference.
    label_data, label_sim : legend labels for the two sources.

    Returns
    -------
    dict with dHH_centers, "data" (and "sim" if provided): each a dict with, for tag
    in {'real','shift'}: c (circulation corr), c_sem, drift (mm/s), drift_sem, and
    N_c/N_d (pooled per-bin counts).
    """
    def _select(dslist):
        return [d for d in (dslist or [])
                if d.get("Nfish", None) == 2
                and "radial_position_mm" in d and "polar_angle_rad" in d]

    pair = _select(datasets)
    sim_pair = _select(sim_datasets)
    if not pair and not sim_pair:
        print('\ndiagnose_pair_circulation_and_approach: no pair datasets with '
              'radial_position_mm / polar_angle_rad; skipping.')
        return None

    if dHH_edges is None:
        if arena_radius_mm is not None:
            dmax = 2.0*float(arena_radius_mm)
        else:
            dmax = 0.0
            for d in pair + sim_pair:
                hh = np.asarray(d.get("head_head_distance_mm", []), dtype=float)
                if hh.size:
                    dmax = max(dmax, float(np.nanmax(hh)))
            dmax = dmax if dmax > 0 else 50.0
        dHH_edges = np.arange(0.0, dmax + dHH_bin_mm, dHH_bin_mm)
    dHH_edges = np.asarray(dHH_edges, dtype=float)
    centers = 0.5*(dHH_edges[:-1] + dHH_edges[1:])
    nb = len(centers)
    n_win = max(1, int(round(float(circ_window_s)*float(fps))))

    def _mean_sem(rows):
        if not rows:
            return np.full(nb, np.nan), None
        M = np.vstack(rows)
        n = np.sum(np.isfinite(M), axis=0)
        mean = np.full(nb, np.nan); sd = np.full(nb, np.nan)
        has = n > 0                                  # avoid all-NaN-slice warnings
        if np.any(has):
            with np.errstate(invalid='ignore'):
                mean[has] = np.nanmean(M[:, has], axis=0)
                sd[has] = np.nanstd(M[:, has], axis=0)
        sem = np.where(n > 1, sd/np.sqrt(np.maximum(n, 1)), np.nan)
        return mean, sem

    def _curves(pair_list):
        """Real+shift circulation/drift curves (mean +/- across-unit s.e.m.) for one
        source (a list of dataset/trial dicts). Returns None if empty."""
        if not pair_list:
            return None
        per = {'real': {'c': [], 'd': []}, 'shift': {'c': [], 'd': []}}
        # Per-UNIT (per-movie / per-trial) SCALAR circulation correlation, pooled over
        # all dHH: c_unit = sum(prod)/count for that unit. Its SPREAD across units is
        # the trap discriminator -- a conserved-circulation (locked) sim gives c_unit
        # near +-1 (std ~1) that cancel to ~0 in the ensemble mean, whereas a mixing
        # source gives c_unit tightly clustered near its (small) mean.
        unit_c = {'real': [], 'shift': []}
        pool = {t: {"c_sum": np.zeros(nb), "c_cnt": np.zeros(nb),
                    "d_sum": np.zeros(nb), "d_cnt": np.zeros(nb)}
                for t in ('real', 'shift')}
        for d in pair_list:
            r_mm = np.asarray(d["radial_position_mm"], dtype=float)
            gamma = np.asarray(d["polar_angle_rad"], dtype=float)
            if r_mm.ndim != 2 or r_mm.shape[1] < 2 or gamma.shape != r_mm.shape:
                continue
            nframes = r_mm.shape[0]
            good = _good_frame_mask(d, nframes)
            good = np.ones(nframes, dtype=bool) if good is None else good
            n_shift = nframes // 2
            if n_shift < 2:
                continue
            one = _circulation_drift_one_dataset(r_mm[:, :2], gamma[:, :2], good,
                                                 fps, dHH_edges, n_shift,
                                                 n_win=n_win)
            for t in ('real', 'shift'):
                o = one[t]
                with np.errstate(invalid='ignore', divide='ignore'):
                    cc = np.where(o["c_cnt"] > 0, o["c_sum"]/o["c_cnt"], np.nan)
                    dd = np.where(o["d_cnt"] > 0, o["d_sum"]/o["d_cnt"], np.nan)
                per[t]['c'].append(cc); per[t]['d'].append(dd)
                tot = float(np.sum(o["c_cnt"]))
                unit_c[t].append(float(np.sum(o["c_sum"])/tot) if tot > 0 else np.nan)
                for kk in ("c_sum", "c_cnt", "d_sum", "d_cnt"):
                    pool[t][kk] += o[kk]
        res = {}
        for t in ('real', 'shift'):
            cm, cs = _mean_sem(per[t]['c'])
            dm, ds_ = _mean_sem(per[t]['d'])
            res[t] = {"c": cm, "c_sem": cs, "drift": dm, "drift_sem": ds_,
                      "N_c": pool[t]["c_cnt"], "N_d": pool[t]["d_cnt"],
                      "unit_c": np.asarray(unit_c[t], dtype=float)}
        return res

    data_res = _curves(pair)
    sim_res = _curves(sim_pair)

    far = centers >= 0.7*centers.max()
    def _wavg(vals, w):
        m = np.isfinite(vals) & (w > 0)
        return float(np.sum(vals[m]*w[m])/np.sum(w[m])) if np.any(m) else float('nan')

    def _summary(name, res):
        if res is None:
            return
        uc = res["real"]["unit_c"]; uc = uc[np.isfinite(uc)]
        umean = float(np.mean(uc)) if uc.size else float('nan')
        ustd = float(np.std(uc)) if uc.size else float('nan')
        uabs = float(np.mean(np.abs(uc))) if uc.size else float('nan')
        print(f'  [{name}] c(all)={_wavg(res["real"]["c"], res["real"]["N_c"]):+.3f}'
              f'  c(far)={_wavg(res["real"]["c"][far], res["real"]["N_c"][far]):+.3f}'
              f'  (shift c(all)={_wavg(res["shift"]["c"], res["shift"]["N_c"]):+.3f})'
              f'  drift(far)={_wavg(res["real"]["drift"][far], res["real"]["N_d"][far]):+.3f} mm/s')
        print(f'         per-unit c: mean={umean:+.3f}  std={ustd:.3f}  '
              f'|c|={uabs:.3f}  (N={uc.size})   <- spread ~1 = conserved/locked, '
              f'~0 = mixing')
    print(f'\n--- Pair circulation & approach diagnostic '
          f'(circulation window {n_win/float(fps):.2f} s) ---')
    _summary(label_data, data_res)
    _summary(label_sim, sim_res)

    # Plot: circulation vs dHH (top), approach drift vs dHH (middle), and the
    # per-unit c DISTRIBUTION (bottom) -- the trap discriminator the ensemble mean
    # hides. Each source's real curve is solid; its time-shifted control is dashed.
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 11))
    sources = [(data_res, 'black', label_data)]
    if sim_res is not None:
        sources.append((sim_res, 'darkorange', label_sim))
    for res, col, lab in sources:
        if res is None:
            continue
        ax0.plot(centers, res["real"]["c"], '-', color=col, lw=2, label=lab)
        if res["real"]["c_sem"] is not None:
            ax0.fill_between(centers, res["real"]["c"] - res["real"]["c_sem"],
                             res["real"]["c"] + res["real"]["c_sem"],
                             color=col, alpha=0.22, linewidth=0)
        ax0.plot(centers, res["shift"]["c"], '--', color=col, lw=1, alpha=0.7,
                 label=f'{lab} (time-shifted)')
        ax1.plot(centers, res["real"]["drift"], '-', color=col, lw=2, label=lab)
        if res["real"]["drift_sem"] is not None:
            ax1.fill_between(centers, res["real"]["drift"] - res["real"]["drift_sem"],
                             res["real"]["drift"] + res["real"]["drift_sem"],
                             color=col, alpha=0.22, linewidth=0)
        ax1.plot(centers, res["shift"]["drift"], '--', color=col, lw=1, alpha=0.7)
    ax0.axhline(0.0, color='gray', lw=0.8, ls=':')
    ax1.axhline(0.0, color='gray', lw=0.8, ls=':')
    ax0.set_ylabel('circulation corr  c = <sign(gamma_dot_0) sign(gamma_dot_1)>')
    ax0.set_title('Pair circulation & approach vs inter-fish distance '
                  f'({n_win/float(fps):.2f} s window)\n'
                  '(solid = real pairing, dashed = time-shifted control)')
    ax0.legend(fontsize=8)
    ax1.set_ylabel('approach drift  <d(dHH)/dt>  (mm/s)')
    ax1.set_xlabel('inter-fish distance dHH (mm)')
    # Bottom: per-unit (per-movie / per-trial) circulation correlation, jittered
    # points + mean. A conserved/locked source spreads toward +-1; a mixing source
    # clusters near its mean.
    rng_j = np.random.default_rng(0)
    for xpos, (res, col, lab) in enumerate(sources):
        if res is None:
            continue
        uc = res["real"]["unit_c"]; uc = uc[np.isfinite(uc)]
        if uc.size == 0:
            continue
        xj = xpos + (rng_j.uniform(-0.12, 0.12, uc.size))
        ax2.plot(xj, uc, 'o', color=col, alpha=0.6, ms=5)
        ax2.plot([xpos - 0.2, xpos + 0.2], [uc.mean()]*2, '-', color=col, lw=2.5)
    ax2.axhline(0.0, color='gray', lw=0.8, ls=':')
    ax2.set_xticks(range(len(sources)))
    ax2.set_xticklabels([lab for _, _, lab in sources])
    ax2.set_ylabel('per-unit circulation corr c\n(each point = one movie / trial)')
    ax2.set_title('Per-unit c: spread ~1 = conserved/locked circulation, '
                  '~0 = mixing', fontsize=10)
    ax2.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    if outputFileName is not None:
        fig.savefig(outputFileName, dpi=140)
        print(f'  Saved circulation/approach diagnostic: {outputFileName}')
    plt.show(block=False)
    if closeFigure:
        plt.close(fig)
    return {"dHH_centers": centers, "data": data_res, "sim": sim_res}


def plot_pair_dHH_real_vs_timeshift(
        datasets, sim_datasets=None, arena_radius_mm=None, bin_width_mm=1.0,
        dHH_max_mm=None, show_time_shifted=True,
        label_data='real pairs', label_sim='simulated pairs',
        outputFileName='pair_dHH_real_vs_timeshift.png', closeFigure=False):
    """
    [DIAGNOSTIC] p(dHH) for the REAL pairing vs the TIME-SHIFTED control (fish 1
    rolled by half the movie -- the "independent pairing" null), for the experimental
    data and, if given, the simulation. Where real and shifted DIFFER is where the
    source has genuine inter-fish coupling: a real, aggregating pair sits ABOVE the
    shifted null at short dHH and BELOW it at large dHH (depleting the 2R geometric
    pileup); if real ~ shifted everywhere (as expected for the weakly-coupled sim),
    the large-dHH peak is just the two-independent-wall-followers geometry.

    Distances are reconstructed body-CENTER separations (consistent between real and
    shifted). show_time_shifted=False drops the dashed time-shifted curves (leaving
    just the real p(dHH) for each source) -- the usual setting once the null has been
    inspected.

    Returns dict with dHH_centers and, per source ('data'/'sim'), the real & shifted
    densities; None if no usable pair data.
    """
    def _select(dl):
        return [d for d in (dl or []) if d.get("Nfish", None) == 2
                and "radial_position_mm" in d and "polar_angle_rad" in d]
    data = _select(datasets); sim = _select(sim_datasets)
    if not data and not sim:
        print('\nplot_pair_dHH_real_vs_timeshift: no usable pair data; skipping.')
        return None

    if dHH_max_mm is None:
        dHH_max_mm = 2.0*float(arena_radius_mm) if arena_radius_mm else 50.0
    edges = np.arange(0.0, dHH_max_mm + bin_width_mm, bin_width_mm)
    centers = 0.5*(edges[:-1] + edges[1:])

    def _curves(pairlist):
        if not pairlist:
            return None
        reals, shifts = [], []
        for d in pairlist:
            n = np.asarray(d["radial_position_mm"], dtype=float).shape[0]
            n_shift = n // 2
            if n_shift < 2:
                continue
            dre, dsh = _pair_dHH_real_shifted_frames(d, n_shift)
            if dre is None:
                continue
            reals.append(dre[np.isfinite(dre)])
            shifts.append(dsh[np.isfinite(dsh)])
        rd, rsem = _density_and_sem(reals, edges)
        sd, ssem = _density_and_sem(shifts, edges)
        return {"real": (rd, rsem), "shift": (sd, ssem)}

    dres = _curves(data); sres = _curves(sim)

    def _pfar(dens):
        far = centers >= 0.7*centers.max()
        return float(np.nansum(dens[far])*bin_width_mm)
    print('\n--- p(dHH): real pairing vs time-shifted (independent) null ---')
    for nm, res in (('data', dres), ('sim', sres)):
        if res is None:
            continue
        print(f'  [{nm}] P(dHH>0.7*max): real={_pfar(res["real"][0]):.3f}  '
              f'shift={_pfar(res["shift"][0]):.3f}  '
              f'(real<shift => genuine far-tail depletion / aggregation)')

    fig = plt.figure(figsize=(9, 5.5))
    for res, col, lab in [(dres, 'black', label_data),
                          (sres, 'darkorange', label_sim)]:
        if res is None:
            continue
        rd, rsem = res["real"]
        plt.plot(centers, rd, '-', color=col, lw=2, label=lab)
        if rsem is not None:
            plt.fill_between(centers, rd - rsem, rd + rsem, color=col,
                             alpha=0.25, linewidth=0)
        if show_time_shifted:
            sd, ssem = res["shift"]
            plt.plot(centers, sd, '--', color=col, lw=1.3, alpha=0.8,
                     label=f'{lab} (time-shifted)')
            if ssem is not None:
                plt.fill_between(centers, sd - ssem, sd + ssem, color=col,
                                 alpha=0.12, linewidth=0)
    plt.ylim(bottom=0)
    plt.xlabel('inter-fish distance dHH (mm)', fontsize=12)
    plt.ylabel('probability density', fontsize=12)
    ttl = 'p(dHH): real pairing'
    ttl += ' vs time-shifted null' if show_time_shifted else ''
    plt.title(ttl, fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    if outputFileName is not None:
        plt.savefig(outputFileName, dpi=140)
        print(f'  Saved p(dHH) real-vs-timeshift figure: {outputFileName}')
    plt.show(block=False)
    if closeFigure:
        plt.close(fig)
    return {"dHH_centers": centers, "data": dres, "sim": sres}


def plot_pair_dHH_autocorrelation(
        datasets, sim_datasets=None, fps=25.0, max_lag_s=60.0, n_lags=80,
        label_data='real pairs', label_sim='simulated pairs',
        outputFileName='pair_dHH_autocorrelation.png', closeFigure=False):
    """
    [DIAGNOSTIC] Within-unit temporal autocorrelation of the inter-fish distance
    dHH(t), for the experimental data and (if given) the simulation. This measures
    the SLOW-MODE timescale that governs how many independent p(dHH) samples each
    movie/trial provides -- the direct test of whether the run-to-run p(dHH) variance
    is just finite sampling of a long-autocorrelation observable.

    Per unit (movie/trial), the demeaned dHH is correlated at each lag over good-frame
    pairs; the ACF is averaged across units (with a +/- s.e.m. band). Reports the
    1/e decay time tau_1e and the integrated time tau_int (dt * sum of the ACF up to
    its first zero crossing) for each source. Compare tau to the trial length: tau <<
    T means the variance is ordinary sampling (more trials converge it); tau ~ T means
    a genuine slow mode remains. Uses the stored "head_head_distance_mm".

    Returns dict with lag_s and, per source, acf/acf_sem/tau_1e/tau_int.
    """
    def _select(dl):
        return [d for d in (dl or []) if d.get("Nfish", None) == 2
                and np.asarray(d.get("head_head_distance_mm", [])).size > 0]
    data = _select(datasets); sim = _select(sim_datasets)
    if not data and not sim:
        print('\nplot_pair_dHH_autocorrelation: no usable pair data; skipping.')
        return None

    lags = np.unique(np.linspace(1, max(2, int(max_lag_s*fps)),
                                 n_lags).astype(int))
    lag_s = lags/float(fps)

    def _acf_one(dHH, good):
        good = good & np.isfinite(dHH)
        if int(np.sum(good)) < 100:
            return None
        x = np.where(good, dHH - np.mean(dHH[good]), np.nan)
        var0 = np.mean(x[good]**2)
        if not np.isfinite(var0) or var0 <= 0:
            return None
        out = np.full(len(lags), np.nan)
        for i, L in enumerate(lags):
            if L >= x.size:
                break
            a = x[:-L]; b = x[L:]
            m = np.isfinite(a) & np.isfinite(b)
            if int(np.sum(m)) > 20:
                out[i] = np.mean(a[m]*b[m])/var0
        return out

    def _curves(pairlist):
        if not pairlist:
            return None
        rows = []
        for d in pairlist:
            dHH = np.asarray(d.get("head_head_distance_mm", []), dtype=float).ravel()
            if dHH.size == 0:
                continue
            good = _good_frame_mask(d, dHH.shape[0])
            good = np.ones(dHH.shape[0], dtype=bool) if good is None else good
            a = _acf_one(dHH, good)
            if a is not None:
                rows.append(a)
        if not rows:
            return None
        M = np.vstack(rows)
        n = np.sum(np.isfinite(M), axis=0)
        with np.errstate(invalid='ignore'):
            mean = np.where(n > 0, np.nanmean(M, axis=0), np.nan)
            sd = np.where(n > 0, np.nanstd(M, axis=0), np.nan)
        sem = np.where(n > 1, sd/np.sqrt(np.maximum(n, 1)), np.nan)
        # 1/e decay time (first downward crossing of 1/e), linearly interpolated.
        tau_1e = np.nan
        thr = 1.0/np.e
        for i in range(len(mean)):
            if np.isfinite(mean[i]) and mean[i] < thr:
                if i == 0:
                    tau_1e = lag_s[0]
                else:
                    y0, y1 = mean[i-1], mean[i]
                    f = (y0 - thr)/(y0 - y1) if y1 != y0 else 0.0
                    tau_1e = lag_s[i-1] + f*(lag_s[i] - lag_s[i-1])
                break
        # Integrated time: dt * sum of ACF up to the first zero (or non-finite).
        dt = 1.0/float(fps)
        tau_int = 0.0
        prev_lag = 0.0
        for i in range(len(mean)):
            if not np.isfinite(mean[i]) or mean[i] <= 0:
                break
            tau_int += mean[i]*(lag_s[i] - prev_lag)
            prev_lag = lag_s[i]
        return {"acf": mean, "acf_sem": sem, "tau_1e": tau_1e, "tau_int": tau_int}

    dres = _curves(data); sres = _curves(sim)
    print('\n--- dHH autocorrelation time (within movie/trial) ---')
    for nm, res in (('data', dres), ('sim', sres)):
        if res is None:
            continue
        print(f'  [{nm}] tau_1e={res["tau_1e"]:.1f} s   tau_int={res["tau_int"]:.1f} s')

    fig = plt.figure(figsize=(9, 5.5))
    for res, col, lab in [(dres, 'black', label_data),
                          (sres, 'darkorange', label_sim)]:
        if res is None:
            continue
        plt.plot(lag_s, res["acf"], '-', color=col, lw=2,
                 label=f'{lab} (tau_1e={res["tau_1e"]:.1f}s)')
        if res["acf_sem"] is not None:
            plt.fill_between(lag_s, res["acf"] - res["acf_sem"],
                             res["acf"] + res["acf_sem"], color=col,
                             alpha=0.22, linewidth=0)
    plt.axhline(1.0/np.e, color='gray', lw=0.8, ls='--', label='1/e')
    plt.axhline(0.0, color='gray', lw=0.6, ls=':')
    plt.xlabel('lag (s)', fontsize=12)
    plt.ylabel('dHH autocorrelation', fontsize=12)
    plt.title('Within-trial dHH autocorrelation (slow-mode timescale)', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    if outputFileName is not None:
        plt.savefig(outputFileName, dpi=140)
        print(f'  Saved dHH autocorrelation figure: {outputFileName}')
    plt.show(block=False)
    if closeFigure:
        plt.close(fig)
    return {"lag_s": lag_s, "data": dres, "sim": sres}


def diagnose_kinematics_vs_dHH(
        datasets_A, datasets_B=None, single_fish_IB=None,
        n_r_strata=2, n_psi_strata=2, dHH_bin_mm=5.0, dHH_max_mm=None,
        arena_radius_mm=None, min_delta_s=0.0, max_bout_speed_mm_s=None,
        max_bout_turn_angle_rad_s=None, fps=25.0, min_cell_N=20,
        outputFileName='kinematics_vs_dHH.png', closeFigure=False):
    """
    [DIAGNOSTIC] Deconfounded characterization of whether the pair KINEMATICS
    (Delta_s, Delta_t, IB_duration) are altered by the neighbour -- separating a
    genuine change from the reweighting artifact (social bouts occur at particular
    (r, psi), so a pooled (r, psi)-marginal looks altered even if the intrinsic
    kinematics are not). Everything is compared WITHIN (r, |psi|) strata, so the
    reweighting confound cancels.

    Two tests, per (r, |psi|) stratum and kinematic:
      A. INTRINSIC alteration: does the neighbour-FAR real value converge to the
         SINGLE-FISH value (horizontal reference)? If yes, the asocial kinematics are
         unchanged and any pooled difference is purely social reweighting; if no, a
         genuine non-local (arousal/state) change.
      B. SOCIAL modulation (deconfounded): the dHH-dependence of the REAL pairing
         (solid) vs the TIME-SHIFTED control (dashed). Time-shifting scrambles the
         dHH label while preserving each fish's own (r, psi, kinematics) and the
         geometric r-dHH correlation, so REAL minus TIME-SHIFTED at fixed (r, psi) is
         the genuine fish-fish kinematic coupling -- the same null logic as w_excess.

    Inputs
    ------
    datasets_A : real pair (Nfish==2) datasets (e.g. pairstats_2).
    datasets_B : time-shifted pair datasets (e.g. pairstats_2b) for test B; None ->
        real only (no deconfounded social curve).
    single_fish_IB : single-fish pooled_IB_properties dict for test A; None -> no
        single-fish reference line.
    n_r_strata, n_psi_strata : number of r (radial-quantile) and |psi| (0..pi) strata
        -> n_r_strata*n_psi_strata panel columns; 3 kinematic rows.
    dHH_bin_mm, dHH_max_mm, arena_radius_mm : dHH binning (default 0..2R).
    min_cell_N : minimum bouts per (stratum, dHH) cell to plot a point.

    Returns dict with the strata, dHH centres, and per-source binned means.
    """
    poolA = _pool_pair_kinematics(datasets_A, min_delta_s, max_bout_speed_mm_s,
                                  max_bout_turn_angle_rad_s, fps)
    if poolA is None:
        print('\ndiagnose_kinematics_vs_dHH: no usable pair bouts; skipping.')
        return None
    poolB = (_pool_pair_kinematics(datasets_B, min_delta_s, max_bout_speed_mm_s,
                                   max_bout_turn_angle_rad_s, fps)
             if datasets_B else None)
    poolS = (_pool_single_kinematics(single_fish_IB, min_delta_s,
                                     max_bout_speed_mm_s,
                                     max_bout_turn_angle_rad_s, fps)
             if single_fish_IB is not None else None)

    # Strata edges: r by quantiles of the real pair data; |psi| linear over [0, pi].
    r_edges = np.quantile(poolA["r"], np.linspace(0.0, 1.0, n_r_strata + 1))
    r_edges[0] = -np.inf; r_edges[-1] = np.inf
    psi_edges = np.linspace(0.0, np.pi, n_psi_strata + 1)
    psi_edges[0] = -0.1; psi_edges[-1] = np.pi + 0.1
    if dHH_max_mm is None:
        dHH_max_mm = (2.0*float(arena_radius_mm) if arena_radius_mm
                      else float(np.nanmax(poolA["dHH"])))
    dHH_edges = np.arange(0.0, dHH_max_mm + dHH_bin_mm, dHH_bin_mm)
    dcent = 0.5*(dHH_edges[:-1] + dHH_edges[1:])

    kins = [("Delta_s", "step size  <Δs>  (mm)"),
            ("Delta_t", "bout duration  <Δt>  (s)"),
            ("IB", "IB duration  <IBI>  (s)")]

    def _stratum_mask(pool, ri, pi):
        return ((pool["r"] >= r_edges[ri]) & (pool["r"] < r_edges[ri+1])
                & (pool["psi"] >= psi_edges[pi]) & (pool["psi"] < psi_edges[pi+1]))

    def _binned(pool, sel, key):
        dv = pool["dHH"][sel]; yv = pool[key][sel]
        idx = np.digitize(dv, dHH_edges) - 1
        nb = len(dcent)
        m = np.full(nb, np.nan); s = np.full(nb, np.nan); n = np.zeros(nb)
        for i in range(nb):
            yy = yv[idx == i]; yy = yy[np.isfinite(yy)]
            n[i] = yy.size
            if yy.size >= min_cell_N:
                m[i] = yy.mean(); s[i] = yy.std()/np.sqrt(yy.size)
        return m, s, n

    strata = [(ri, pi) for ri in range(n_r_strata) for pi in range(n_psi_strata)]
    ncols = len(strata); nrows = len(kins)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, 3.0*nrows),
                             squeeze=False, sharex=True, sharey='row')
    far = dcent >= 0.7*dcent.max()
    res = {"dHH_centers": dcent, "strata": [], "data": {}, "shift": {}, "single": {}}
    print('\n--- Kinematics vs dHH within (r, |psi|) strata '
          '(A: far vs single-fish; B: real vs time-shifted) ---')
    for ci, (ri, pi) in enumerate(strata):
        lab = (f'r[{r_edges[ri] if np.isfinite(r_edges[ri]) else 0:.0f}-'
               f'{r_edges[ri+1] if np.isfinite(r_edges[ri+1]) else 999:.0f}]mm, '
               f'|psi|[{np.degrees(max(psi_edges[pi],0)):.0f}-'
               f'{np.degrees(min(psi_edges[pi+1],np.pi)):.0f}]deg')
        res["strata"].append(lab)
        selA = _stratum_mask(poolA, ri, pi)
        selB = _stratum_mask(poolB, ri, pi) if poolB else None
        selS = _stratum_mask(poolS, ri, pi) if poolS else None
        for row, (key, ylab) in enumerate(kins):
            ax = axes[row][ci]
            mA, sA, nA = _binned(poolA, selA, key)
            ax.plot(dcent, mA, '-', color='black', lw=1.8, label='real')
            ax.fill_between(dcent, mA - sA, mA + sA, color='black',
                            alpha=0.2, linewidth=0)
            res["data"].setdefault(key, []).append(mA)
            if poolB is not None:
                mB, sB, nB = _binned(poolB, selB, key)
                ax.plot(dcent, mB, '--', color='darkorange', lw=1.5,
                        label='time-shifted')
                res["shift"].setdefault(key, []).append(mB)
            if poolS is not None:
                sval = float(np.nanmean(poolS[key][selS])) if np.any(selS) else np.nan
                ax.axhline(sval, color='green', lw=1.3, ls=':', label='single-fish')
                res["single"].setdefault(key, []).append(sval)
            if row == 0:
                ax.set_title(lab, fontsize=8)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=9)
            if row == nrows - 1:
                ax.set_xlabel('dHH (mm)', fontsize=9)
            if row == 0 and ci == 0:
                ax.legend(fontsize=7)
        # Compact summary for Delta_s (the primary kinematic).
        mAs, _, _ = _binned(poolA, selA, "Delta_s")
        near = float(np.nanmean(mAs[dcent < 12.0]))
        farv = float(np.nanmean(mAs[far]))
        sref = (float(np.nanmean(poolS["Delta_s"][selS]))
                if poolS is not None and np.any(selS) else float('nan'))
        print(f'  {lab}: <ds> near={near:.2f} far={farv:.2f} mm  '
              f'single-fish={sref:.2f} mm  (A:far-vs-single={farv - sref:+.2f}, '
              f'B:near-vs-far={near - farv:+.2f})')

    fig.suptitle('Pair kinematics vs neighbour distance, within (r, |psi|) strata\n'
                 'solid=real, dashed=time-shifted (social null), dotted=single-fish',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if outputFileName is not None:
        fig.savefig(outputFileName, dpi=140)
        print(f'  Saved kinematics-vs-dHH diagnostic: {outputFileName}')
    plt.show(block=False)
    if closeFigure:
        plt.close(fig)
    return res


def plot_pair_dHH_null_comparison(
        pair_datasets, single_fish_datasets=None, sim_sim_dHH_list=None,
        arena_radius_mm=None, bin_width_mm=1.0, dHH_max_mm=None,
        outputFileName='pair_dHH_null_comparison.png', closeFigure=False):
    """
    [DIAGNOSTIC] Compare the candidate ASOCIAL nulls for p(dHH), to decide which
    baseline to subtract for the social turning signal (and to check whether the
    single-fish SIM occupancy is trustworthy). Overlays, as normalized densities:
      - REAL pairs                     (black)        -- the target;
      - TIME-SHIFTED pairs             (grey dashed)  -- current null (real occupancy,
                                                         but social-shaped behaviour);
      - two SINGLE-FISH SIMULATIONS    (orange)       -- model asocial null;
      - two REAL single fish, paired   (green)        -- data asocial null (no model).

    Reading: if the two REAL-single-fish-paired curve matches the SIM-SIM curve, the
    flat time-shifted-pair shape is a genuine SOCIAL occupancy change and a two-single-
    fish null is the right baseline. If instead real-single-paired matches the flat
    time-shifted pair, the single-fish SIM occupancy is wrong (too edge-peaked) and
    must be fixed before it can serve as a null. All distances are body-CENTER
    separations, consistent across sources.

    Inputs
    ------
    pair_datasets : real pair (Nfish==2) datasets (radial_position_mm/polar_angle_rad).
    single_fish_datasets : single-fish (Nfish==1) datasets for the real-paired null;
        None -> skip that curve.
    sim_sim_dHH_list : list of dHH arrays from an ASOCIAL (two-single-fish) simulation
        (e.g. simulate_pair_dHH_trials(..., social_method='turn_sampling_additive',
        mean_angle_multiplier=0.0)[0]); None -> skip that curve.
    """
    def _select_pair(dl):
        return [d for d in (dl or []) if d.get("Nfish", None) == 2
                and "radial_position_mm" in d and "polar_angle_rad" in d]
    pair = _select_pair(pair_datasets)

    if dHH_max_mm is None:
        dHH_max_mm = 2.0*float(arena_radius_mm) if arena_radius_mm else 50.0
    edges = np.arange(0.0, dHH_max_mm + bin_width_mm, bin_width_mm)
    centers = 0.5*(edges[:-1] + edges[1:])

    # Real pairs -> real + time-shifted per-dataset frame dHH.
    real_arrs, shift_arrs = [], []
    for d in pair:
        n = np.asarray(d["radial_position_mm"], dtype=float).shape[0]
        if n // 2 < 2:
            continue
        dre, dsh = _pair_dHH_real_shifted_frames(d, n // 2)
        if dre is None:
            continue
        real_arrs.append(dre[np.isfinite(dre)])
        shift_arrs.append(dsh[np.isfinite(dsh)])

    rng = np.random.default_rng(0)
    single_arrs = (_paired_single_fish_dHH(
        [d for d in single_fish_datasets if d.get("Nfish", None) == 1], rng)
        if single_fish_datasets else [])
    sim_arrs = ([np.asarray(a, dtype=float).ravel() for a in sim_sim_dHH_list]
                if sim_sim_dHH_list else [])

    curves = []   # (label, color, linestyle, density, sem)
    def _add(label, color, ls, arrs):
        if arrs:
            dens, sem = _density_and_sem(arrs, edges)
            curves.append((label, color, ls, dens, sem))
    _add('real pairs', 'black', '-', real_arrs)
    _add('time-shifted pairs', 'grey', '--', shift_arrs)
    _add('two single-fish sims', 'darkorange', '-', sim_arrs)
    _add('two real single fish (paired)', 'green', '-', single_arrs)
    if not curves:
        print('\nplot_pair_dHH_null_comparison: no usable data; skipping.')
        return None

    def _pfar(dens):
        far = centers >= 0.7*centers.max()
        return float(np.nansum(dens[far])*bin_width_mm)
    print('\n--- p(dHH): candidate asocial nulls ---')
    for label, _c, _ls, dens, _s in curves:
        print(f'  {label:32s} P(dHH>0.7*max)={_pfar(dens):.3f}')

    fig = plt.figure(figsize=(9, 5.5))
    for label, color, ls, dens, sem in curves:
        lw = 2.2 if ls == '-' else 1.5
        plt.plot(centers, dens, ls, color=color, lw=lw, label=label)
        if sem is not None:
            plt.fill_between(centers, dens - sem, dens + sem, color=color,
                             alpha=0.18, linewidth=0)
    plt.ylim(bottom=0)
    plt.xlabel('inter-fish distance dHH (mm)', fontsize=12)
    plt.ylabel('probability density', fontsize=12)
    plt.title('p(dHH): real pairs vs candidate asocial nulls', fontsize=12)
    plt.legend(fontsize=9)
    plt.tight_layout()
    if outputFileName is not None:
        plt.savefig(outputFileName, dpi=140)
        print(f'  Saved p(dHH) null-comparison figure: {outputFileName}')
    plt.show(block=False)
    if closeFigure:
        plt.close(fig)
    return {"dHH_centers": centers,
            "curves": {label: dens for label, _c, _ls, dens, _s in curves}}


def _circulation_drift_one_dataset(r_mm, gamma_rad, good, fps, edges, n_shift,
                                   n_win=1):
    """
    Frame-level circulation correlation and approach drift for ONE pair dataset,
    for BOTH the real pairing and a time-shifted control (fish 1 rolled by n_shift
    frames). Inputs are per-fish frame arrays r_mm, gamma_rad of shape
    (Nframes, 2) and a good-frame boolean mask (Nframes,). Returns a dict with, for
    tag in {'real','shift'}: c_sum/c_cnt (binned sum & count of the circulation-sign
    product sign(dgamma_0)*sign(dgamma_1)) and d_sum/d_cnt (binned sum & count of
    the approach rate d(dHH)/dt in mm/s), all over the dHH bins defined by `edges`.

    n_win : window (frames) over which the net rotation and dHH change are taken --
        sign(gamma[t+n_win] - gamma[t]), (dHH[t+n_win] - dHH[t])/(n_win*dt). n_win=1
        is the per-frame difference. A window of ~one inter-bout interval is ESSENTIAL
        for SIMULATED trajectories: interpolate_pair_rsim uses nearest-neighbour
        (STEP) interpolation, so per-frame dgamma is zero except at the sparse bout
        transitions, and requiring BOTH fish nonzero on the same frame leaves <1% of
        frames usable -> a hugely noisy c. Averaging the net rotation over ~1 IBI
        makes gamma move within the window (dense sampling) and measures circulation
        at the bout scale, comparable between real and simulated data.

    The two fish positions are reconstructed as x=r cos(gamma), y=r sin(gamma) (a
    body-CENTER separation, consistent between real and control; it differs slightly
    from the stored head-head distance but the real-vs-control CONTRAST is the point).
    """
    def _wrap(a):
        return (np.asarray(a, dtype=float) + np.pi) % (2.0*np.pi) - np.pi

    x = r_mm*np.cos(gamma_rad); y = r_mm*np.sin(gamma_rad)   # (Nframes, 2)
    nb = len(edges) - 1
    W = max(1, int(n_win))
    out = {}
    for tag in ('real', 'shift'):
        if tag == 'real':
            x1, y1, g1 = x[:, 1], y[:, 1], gamma_rad[:, 1]
            valid = good.copy()
        else:
            # Time-shifted control: roll fish 1 by half the movie so any real-time
            # coupling is destroyed while each fish keeps its own statistics.
            x1 = np.roll(x[:, 1], n_shift)
            y1 = np.roll(y[:, 1], n_shift)
            g1 = np.roll(gamma_rad[:, 1], n_shift)
            valid = good & np.roll(good, n_shift)
        dHH = np.hypot(x[:, 0] - x1, y[:, 0] - y1)
        g0 = gamma_rad[:, 0]
        # Net change over a W-frame window t -> t+W (length Nframes-W), indexed at t.
        s0 = np.sign(_wrap(g0[W:] - g0[:-W]))
        s1 = np.sign(_wrap(g1[W:] - g1[:-W]))
        prod = s0*s1                                   # +1 co-rotate, -1 counter
        ddHH = (dHH[W:] - dHH[:-W])*fps/float(W)       # mm/s (approach if < 0)
        base = dHH[:-W]                                # bin by dHH at frame t
        vv = valid[:-W] & valid[W:] & np.isfinite(base)
        idx = np.digitize(base, edges) - 1
        inrange = vv & (idx >= 0) & (idx < nb)
        c_sum = np.zeros(nb); c_cnt = np.zeros(nb)
        d_sum = np.zeros(nb); d_cnt = np.zeros(nb)
        okc = inrange & (s0 != 0) & (s1 != 0)
        okd = inrange & np.isfinite(ddHH)
        np.add.at(c_sum, idx[okc], prod[okc]); np.add.at(c_cnt, idx[okc], 1.0)
        np.add.at(d_sum, idx[okd], ddHH[okd]); np.add.at(d_cnt, idx[okd], 1.0)
        out[tag] = {"c_sum": c_sum, "c_cnt": c_cnt, "d_sum": d_sum, "d_cnt": d_cnt}
    return out


def _fit_sigma_asymptote(dHH_centers, sigma, N_marg, d_far_plateau=20.0):
    """Fit the phi-marginal within-condition turn std sigma(dHH) to
    _sigma_approach, weighted by 1/sqrt(N), and return (sigma_far, popt) with
    popt = [s_far, s_0, lambda] (or None if the fit failed). The marginal is high-
    count so s_far is low-variance. Falls back to the count-weighted plateau mean over
    dHH >= d_far_plateau, then the last finite value."""
    dHH_centers = np.asarray(dHH_centers, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    N_marg = np.asarray(N_marg)
    mfit = np.isfinite(sigma) & (N_marg > 0)
    sigma_far = np.nan; popt = None
    if int(np.sum(mfit)) >= 3:
        try:
            from scipy.optimize import curve_fit
            _x = dHH_centers[mfit]; _y = sigma[mfit]
            _w = 1.0/np.sqrt(np.maximum(N_marg[mfit], 1))
            popt, _ = curve_fit(
                _sigma_approach, _x, _y,
                p0=[float(np.min(_y)), float(np.max(_y)), 5.0],
                sigma=_w, absolute_sigma=False, maxfev=20000,
                bounds=([0.0, 0.0, 0.5], [np.pi, np.pi, 100.0]))
            sigma_far = float(popt[0])
        except Exception:
            popt = None
    if not np.isfinite(sigma_far):
        pl = mfit & (dHH_centers >= d_far_plateau)
        if np.any(pl):
            sigma_far = float(np.average(sigma[pl], weights=N_marg[pl]))
        elif np.any(mfit):
            sigma_far = float(sigma[mfit][-1])
    return sigma_far, popt


def _pair_dHH_real_shifted_frames(d, n_shift):
    """(dHH_real, dHH_shift) frame arrays (bad/undefined frames set NaN) for ONE pair
    dataset, reconstructed as the body-CENTER separation from radial_position_mm and
    polar_angle_rad. The shifted array rolls fish 1 by n_shift frames (the same time-
    shifted control as the circulation diagnostic: an "independent pairing" null).
    Returns (None, None) if the per-fish frame arrays are missing/misshapen."""
    r_mm = np.asarray(d.get("radial_position_mm", []), dtype=float)
    gamma = np.asarray(d.get("polar_angle_rad", []), dtype=float)
    if r_mm.ndim != 2 or r_mm.shape[1] < 2 or gamma.shape != r_mm.shape:
        return None, None
    n = r_mm.shape[0]
    good = _good_frame_mask(d, n)
    good = np.ones(n, dtype=bool) if good is None else good
    x = r_mm[:, :2]*np.cos(gamma[:, :2]); y = r_mm[:, :2]*np.sin(gamma[:, :2])
    d_real = np.hypot(x[:, 0] - x[:, 1], y[:, 0] - y[:, 1])
    d_real = np.where(good, d_real, np.nan)
    x1 = np.roll(x[:, 1], n_shift); y1 = np.roll(y[:, 1], n_shift)
    gs = good & np.roll(good, n_shift)
    d_shift = np.hypot(x[:, 0] - x1, y[:, 0] - y1)
    d_shift = np.where(gs, d_shift, np.nan)
    return d_real, d_shift


def _paired_single_fish_dHH(single_datasets, rng):
    """Pair INDEPENDENT single-fish trajectories (all distinct dataset pairs) and
    return a list of per-pair frame-level dHH arrays -- the genuinely-asocial
    'two real single fish' null. Positions are the body-CENTER (x=r cos g,
    y=r sin g) from radial_position_mm / polar_angle_rad (Nfish==1 -> column 0). A
    random circular offset decorrelates any incidental synchrony. Distance only, so
    there is no orientation-convention ambiguity."""
    trajs = []
    for d in single_datasets:
        r = np.asarray(d.get("radial_position_mm", []), dtype=float)
        g = np.asarray(d.get("polar_angle_rad", []), dtype=float)
        if r.ndim != 2 or r.shape[1] < 1 or g.shape != r.shape:
            continue
        n = r.shape[0]
        good = _good_frame_mask(d, n)
        good = np.ones(n, dtype=bool) if good is None else good
        x = np.where(good, r[:, 0]*np.cos(g[:, 0]), np.nan)
        y = np.where(good, r[:, 0]*np.sin(g[:, 0]), np.nan)
        trajs.append((x, y))
    out = []
    for i in range(len(trajs)):
        for j in range(i + 1, len(trajs)):
            xi, yi = trajs[i]; xj, yj = trajs[j]
            n = min(len(xi), len(xj))
            if n < 4:
                continue
            off = int(rng.integers(0, n))
            d = np.hypot(xi[:n] - np.roll(xj[:n], off),
                         yi[:n] - np.roll(yj[:n], off))
            d = d[np.isfinite(d)]
            if d.size:
                out.append(d)
    return out


def _pool_pair_kinematics(datasets, min_delta_s, max_bout_speed_mm_s,
                          max_bout_turn_angle_rad_s, fps):
    """Pool per-bout (r, |psi|, dHH, Delta_s, Delta_t, IB_duration) from PAIR datasets
    (each fish, each IBI), with the standard physical filters. dHH is the neighbour
    distance at that bout. Returns None if nothing usable."""
    def _wrap(a):
        return (np.asarray(a, dtype=float) + np.pi) % (2.0*np.pi) - np.pi
    R, P, D, DS, DT, IB = [], [], [], [], [], []
    for ds in datasets:
        ip = ds.get("IBI_properties")
        if ip is None:
            continue
        for k in range(ds.get("Nfish", 2)):
            r = np.asarray(ip["r_mm_mean"][k], dtype=float)
            th = np.asarray(ip["theta"][k], dtype=float)
            gm = np.asarray(ip["gamma_mean"][k], dtype=float)
            psi = np.abs(_wrap(th - gm))
            dhh = np.asarray(ip["head_head_distance_mm_mean"][k], dtype=float)
            dsz = np.asarray(ip["Delta_s_mm"][k], dtype=float)
            dt = np.asarray(ip["Delta_t_s"][k], dtype=float)
            ibd = np.asarray(ip["IB_duration_s"][k], dtype=float)
            dth = np.asarray(ip["Delta_theta"][k], dtype=float)
            ok = (np.isfinite(r) & np.isfinite(psi) & np.isfinite(dhh)
                  & np.isfinite(dsz) & (dsz > min_delta_s)
                  & _bout_speed_ok(dsz, dt, max_bout_speed_mm_s)
                  & _bout_turn_ok(dth, max_bout_turn_angle_rad_s, fps))
            R.append(r[ok]); P.append(psi[ok]); D.append(dhh[ok])
            DS.append(dsz[ok]); DT.append(dt[ok]); IB.append(ibd[ok])
    if not R or sum(a.size for a in R) == 0:
        return None
    return {"r": np.concatenate(R), "psi": np.concatenate(P),
            "dHH": np.concatenate(D), "Delta_s": np.concatenate(DS),
            "Delta_t": np.concatenate(DT), "IB": np.concatenate(IB)}


def _pool_single_kinematics(pooled_IB, min_delta_s, max_bout_speed_mm_s,
                            max_bout_turn_angle_rad_s, fps):
    """Pool per-bout (r, |psi|, Delta_s, Delta_t, IB_duration) from the SINGLE-fish
    pooled_IB_properties dict (no dHH), with the same physical filters."""
    def _wrap(a):
        return (np.asarray(a, dtype=float) + np.pi) % (2.0*np.pi) - np.pi
    r = np.asarray(pooled_IB["r_mm_mean"], dtype=float)
    th = np.asarray(pooled_IB["theta"], dtype=float)
    gm = np.asarray(pooled_IB["gamma_mean"], dtype=float)
    psi = np.abs(_wrap(th - gm))
    dsz = np.asarray(pooled_IB["Delta_s_mm"], dtype=float)
    dt = np.asarray(pooled_IB["Delta_t_s"], dtype=float)
    ibd = np.asarray(pooled_IB["IB_duration_s"], dtype=float)
    dth = np.asarray(pooled_IB["Delta_theta"], dtype=float)
    ok = (np.isfinite(r) & np.isfinite(psi) & np.isfinite(dsz)
          & (dsz > min_delta_s) & _bout_speed_ok(dsz, dt, max_bout_speed_mm_s)
          & _bout_turn_ok(dth, max_bout_turn_angle_rad_s, fps))
    return {"r": r[ok], "psi": psi[ok], "Delta_s": dsz[ok],
            "Delta_t": dt[ok], "IB": ibd[ok]}


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


def _sigma_approach(x, s_far, s_0, lam):
    """Exponential approach-to-asymptote sigma(dHH) = s_far + (s_0 - s_far)*exp(-x/lam)
    used to read off the far-field (neighbour-absent) turn std s_far."""
    return s_far + (s_0 - s_far)*np.exp(-np.asarray(x, dtype=float)/lam)
