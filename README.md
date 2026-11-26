# Vehicle-level multi-platoon reconfiguration in mixed traffic

This repository contains the code of the paper "Communication-aware planning and distributed control for vehicle-level multi-platoon reconfiguration in mixed traffic". Several platoons of connected and automated vehicles (CAVs) in different lanes are reconfigured into new platoons while human-driven vehicles (HDVs) travel ahead of and inside the platoons. The planning layer decides the target platoon, the interleaving order and the new predecessor-follower links of all vehicles by exact dynamic programming. The execution layer carries out the plan by sequential distributed model predictive control with individually triggered lane changes and compaction.

Chinese description: [README_zh.md](README_zh.md)

This repository contains the code for manuscript version v12. Precomputed experiment data (`results/`) and manuscript files are not included. The reproduction commands below generate the data needed for analysis and figures. Run the planning and `exec_fixed` experiments before `python run_exec.py --selftest`, which compares a new run with the generated reference results. `python verify.py` can be run without experiment data.

## Requirements

Python 3.13 and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
```

`plot_figures.py` typesets the figure text with XeLaTeX through the PGF backend of Matplotlib and needs TeX Live with the TeX Gyre Termes fonts. The Chinese figures use Noto Serif CJK and fall back to SimSun.

## Files

| File | Content | Section of the paper |
| --- | --- | --- |
| `code/scenario.py` | Scenario generator, parameters and link-quality model | 3, 6.1 |
| `code/driver_model.py` | Intelligent driver model with calibrated parameter distributions | 3 |
| `code/sequence.py` | Dynamic programming over order-preserving interleavings, `optimize_joint`, and the terms of the objective | 4 |
| `code/plan_methods.py` | Planning methods SB, MDA, DP-M, DP-MS and DP-MSL behind one interface | 6.2 |
| `code/plan_exact.py` | Exact dynamic programming without beam truncation | 6.2 |
| `code/plan_mixed.py` | Planning with HDVs inside the source platoons | 4 |
| `code/dmpc.py`, `code/event_mpc.py` | Local QP, references, shared plans, acceptance test and lane-change admission | 5.1, 5.2 |
| `code/lag_qp.py`, `code/robust_qp.py` | Lag-aware prediction, constraint tightening and soft recovery | 5.1 |
| `code/timing.py` | Comfort envelope and reference durations | 5.2 |
| `code/run_exec.py` | Closed loop with the three execution policies, the rank-difference reservation and the single-gap ablation | 5.1, 5.2 |
| `code/run_mixed.py` | HDVs ahead of the group and group speed adaptation | 5.3 |
| `code/run_inner.py` | Execution with HDVs inside the platoons | 5.3 |
| `code/audit_continuous_safety.py` | Exact separation margins between sampling instants, independent of the controller | 5.4 |
| `code/cai2022.py` | TAPPA comparison method with target assignment and conflict-based path search | 6.2 |
| `code/traffic_impact.py` | Upstream traffic and delay statistics | 6.3 |
| `code/platoon_comm.py` | Car following under Bernoulli and Gilbert-Elliott packet loss | 6.2 |
| `code/batch_exec.py`, `code/batch_mixed.py`, `code/batch_inner.py` | Batch runners of the closed-loop experiments | 6 |
| `code/analysis.py`, `code/analysis_lag.py` | Statistics and continuous-time checks of all runs | 6 |
| `code/paper_numbers.py` | Numbers quoted in the paper and the data rows of Table 2 | 6 |
| `code/plot_figures.py`, `code/make_tikz.py` | Figures of the paper | 6 |
| `code/verify.py` | Exhaustive check of the planner on a small instance | 4 |

## Experiments

All commands are run in `code/`. The experiment programs write their output to `results/` next to `code/`.

| Group | Command | Output |
| --- | --- | --- |
| EX1 planning methods | `python plan_methods.py --task main` | `results/planning/planning_main_0-99.json`, `results/planning/plans/` |
| EX2 exact solution and beam search | `python plan_exact.py --seeds 0 99`, `python plan_methods.py --task exact ...` | `results/planning/exact_main_0-99.json`, `exact_gap_*.json` |
| EX3 execution policies | `python batch_exec.py --groups exec_adaptive` | `results/exec_adaptive/` |
| EX4 closed-loop execution of the plans | `python batch_exec.py --groups exec_methods` | `results/exec_methods/` |
| EX5 single-gap ablation | `python batch_exec.py --groups ablation` | `results/ablation/` |
| EX6 comfort envelope and uniform duration | `python batch_exec.py --groups jsweep exec_fixed` | `results/jsweep/`, `results/exec_fixed/` |
| EX7 lag and disturbance | `python batch_exec.py --groups robust lagaware lagmismatch` | `results/robust/`, `results/lagaware/`, `results/lagmismatch/` |
| EX8 car following under packet loss | `python platoon_comm.py --mode bernoulli`, `python platoon_comm.py --mode ge` | `results/comm_loop/` |
| EX9 group size and computation time | `python batch_exec.py --groups scale` | `results/scale/` |
| EX10 HDV ahead | `python batch_mixed.py` | `results/mixed/batch/` |
| EX11 HDVs inside the platoons | `python batch_inner.py` | `results/inner/plans/`, `results/inner/runs/` |
| EX12 upstream traffic | `python traffic_impact.py` | `results/traffic/` |
| Rear-anchored layout | `python batch_exec.py --groups exec_rear` | `results/exec_rear/` |
| EX13 comparison with TAPPA | `python cai2022.py` | `results/cai/` |
| Statistics and continuous-time checks | `python analysis.py`, `python analysis_lag.py` | `results/analysis.json`, `results/robust_analysis.json`, `results/audit/`, `results/tables/` |

Each closed-loop run has its own folder with `summary.json` (metrics and settings), `scenario.json` (scenario) and `trajectory.npz` (states, commands and references). Scenario n uses random seed n-1, so `s0` in a file name is scenario 1. In file names, `barrier`, `eager` and `settled` denote the common-phase, immediate and settled policies, and `source_blocks`, `mda`, `motion_only`, `motion_speed` and `combined` denote SB, MDA, DP-M, DP-MS and DP-MSL. The DP plans used in the closed-loop runs come from a beam search of width 1600, and `plan_exact.py` compares them with the untruncated exact solution.

## Reproduction

```bash
cd code

# exhaustive check of the planner, a few seconds
python verify.py

# planning layer
python plan_methods.py --task main
python plan_exact.py --seeds 0 99
python plan_methods.py --task exact --m 4 6 8 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
python plan_methods.py --task exact --m 10 --seeds 0 1 2 3 4 5 6 7 8 9

# closed-loop runs, finished runs are skipped, use --worker k --workers n to split the work over n processes
python batch_exec.py --groups exec_adaptive exec_methods ablation jsweep exec_fixed exec_rear scale robust lagaware lagmismatch
python batch_mixed.py
python batch_inner.py

# rerun of scenario 1 compared with the stored result of the exec_fixed group, about 2 minutes
python run_exec.py --selftest

# comparison method, upstream traffic and car following under packet loss
python cai2022.py
python traffic_impact.py
python platoon_comm.py --mode bernoulli
python platoon_comm.py --mode ge

# statistics, numbers and figures
python analysis.py
python analysis_lag.py
python paper_numbers.py
python plot_figures.py en plan_stats main policies inner traffic
python make_tikz.py en
```

The batch programs skip a run whose `summary.json` exists, so the corresponding result folder has to be removed before a group is rerun. `paper_numbers.py` writes `results.tex` and `tab_cai_rows.tex` into `manuscript/zh` and `manuscript/en`, `plot_figures.py` writes the figures into `manuscript/<lang>/figs` and `make_tikz.py` writes the TikZ sources of the scenario figure and of an example figure that is not used in the paper into `manuscript/<lang>/figs_tikz`, where `<lang>` is `zh` or `en`. Without figure names, `plot_figures.py` draws ten figures, five of which are not used in the paper.

## Figures and tables of the paper

| Item | Program |
| --- | --- |
| Fig. 1 scenario | `make_tikz.py` |
| Fig. 3 planning statistics | `plot_figures.py plan_stats` |
| Fig. 4 complete reconfiguration | `plot_figures.py main` |
| Fig. 5 execution policies | `plot_figures.py policies` |
| Fig. 6 HDVs inside the platoons | `plot_figures.py inner` |
| Fig. 7 upstream traffic | `plot_figures.py traffic` |
| Table 2 comparison with TAPPA | `paper_numbers.py` |
| Numbers in the text | `paper_numbers.py` |

Fig. 2 is a block diagram and Table 1 lists the parameters set in `scenario.py` and the run scripts.

## Environment and run time

The results of the paper were obtained with Python 3.13, NumPy 2.5, SciPy 1.18, Clarabel 0.11.1, OSQP 1.1 and Matplotlib 3.11 on a single thread. All closed-loop runs together take about 12 hours. Computation times depend on the machine, so the planning and solve times and a few last digits that depend on solver precision may differ slightly after a rerun.
