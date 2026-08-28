# pair_simulate_and_compare.py

A self-contained, agent-based simulation of the behaviour of a **pair** of larval
zebrafish, with a direct comparison to experimental data. It is the minimal,
publishable subset of a larger pipeline that contains abandoned and alternative variants -- contact Raghuveer Parthasarathy for details. 

Each fish is a memoryless random walk of linear bouts separated by inter-bout
intervals. Bout kinematics are sampled from data; the turning angle combines a
neighbour-directed mean bias (weight `w_excess(dHH)`) with a distance/orientation-
dependent focusing of the turn variance (factor `f`). Both `w_excess` and `f` are
derived from the data, so **the model has no free (adjustable) parameters**. Running
it produces the inter-fish-distance distribution `p(dHH)` and the radial-position
distribution `p(r)`, overlaid on the experimental curves.

This minimal pipeline was extracted by AI (Claude Opus 4.8) from the full simulation and analysis pipeline. 

Raghuveer Parthasarathy
August 27, 2026

---

## 1. Requirements

### Python packages
`numpy`, `matplotlib`, `scipy`, and — pulled in transitively through `IO_toolkit`
(used here only to load pickles) — `pandas`, `pyyaml`, `tifffile`, `imageio`,
`scikit-learn`, and `tkinter` (part of the standard library on most installs).

```
pip install numpy matplotlib scipy pandas pyyaml tifffile imageio scikit-learn
```

### Local modules (must be on the Python path, alongside the script)
- `pair_social_estimation.py`  — the model-parameter estimators (`w_excess`, the
  focus ratio, the asocial null).
- `IBI_properties_utils.py`    — data layer (bout/IBI extraction and binning).
- `IBI_diagnostics.py`         — the turn-std diagnostic figure.
- `IO_toolkit.py`              — pickle loading.
- `toolkit.py`                 — low-level helpers used by `IO_toolkit`.

> `random_displacement_analysis.py` is **not** required.

---

## 2. Input data — FOUR pickle files (two datasets × two files each)

The script needs **two experimental datasets**, and *each dataset is stored as two
pickle files* — a **position-data** pickle (`..._positionData.pickle`) and an analysis
**"datasets"** pickle (`..._datasets.pickle`). That is **four pickle files in total**:

| Dataset | Fish | Files | Provides |
|---|---|---|---|
| **Single-fish** | `Nfish == 1` | `..._positionData.pickle` + `..._datasets.pickle` | the `(r)` and `(r, psi)` bout/turn distributions (the backbone), the arena radius, and the "two real single fish paired" asocial null for `w_excess` |
| **Pair** | `Nfish == 2` | `..._positionData.pickle` + `..._datasets.pickle` | `w_excess`'s real-pair term, the focus-ratio `sigma(dHH, |phi|)`, the `(r, dHH)` kinematic bins, and the experimental `p(dHH)` / `p(r)` |

Requirements on the pickles:
- The **datasets** pickles must already contain per-dataset `IBI_properties` (they are
  produced by the main analysis pipeline, `behaviors_main.py`). If a pickle predates
  that step, add them without re-analysing the CSVs via
  `IO_toolkit.revise_datasets(keys_to_modify=["IBI_properties"])`.
- The **single-fish** datasets pickle's `expt_config` must contain `arena_radius_mm`
  (the arena radius is read from it, not hard-coded).
- The single-fish pickle must be `Nfish == 1`; the pair pickle `Nfish == 2`
  (the script checks and errors otherwise).

Note that four pickle files from our experiments are provided in TwoWk_Light_pickles.zip in the GitHub repository.
---

## 3. Running

Edit the two path constants near the top of `pair_simulate_and_compare.py`, giving
each dataset as a `(positionData, datasets)` pair:

```python
SINGLE_FISH_PICKLE = (r"...\TwoWkSingle_..._positionData.pickle",
                      r"...\TwoWkSingle_..._Analysis\TwoWkSingle_..._datasets.pickle")
PAIR_PICKLE        = (r"...\TwoWk_..._positionData.pickle",
                      r"...\TwoWk_..._Analysis\TwoWk_..._datasets.pickle")
```

Leave any entry as `None` to be prompted for that path at the terminal. Then:

```
python pair_simulate_and_compare.py
```

Other run settings live in the same **RUN CONFIG** block: `Ntrials` (number of
simulated trials), `T_total_s` (duration per trial), the quality-control cuts
(`fps`, `max_bout_speed_mm_s`), the boundary method (`edgeMethod`), the
distance-conditioned-kinematics options, and the output toggles.

---

## 4. Outputs

Figures written to the working directory (PNG):
- `compare_dHH_exp_vs_sim_<label>.png` — experimental vs simulated `p(dHH)`.
- `compare_r_exp_vs_sim_<label>.png`   — experimental vs simulated `p(r)`.
- `social_blend_weight_vs_dHH_<label>.png` / `social_blend_wexcess_vs_dHH_<label>.png`
  (+ a `.csv`) — the fitted social weight and the excess weight `w_excess(dHH)`.
- `turn_std_phi_dHH_<label>.png` — the turn-standard-deviation diagnostic
  (if `plot_phi_resolved_turn_std` is on).

The `<label>` suffix is set by `run_label` in the config block.

---

## 5. Notes

- **No free parameters:** every quantity (`w_excess`, the focus ratio, the kinematic
  and turn distributions) is estimated from the data.
- **No silent fallbacks:** if a pickle is missing or too sparse to estimate
  `w_excess` or the focus ratio, the script raises a clear error rather than
  substituting a placeholder.
- The asocial null for `w_excess` is the **pseudopair** ("two real single fish paired")
  constructed from the single-fish data — hence the single-fish dataset is required
  even though the model simulates pairs.
