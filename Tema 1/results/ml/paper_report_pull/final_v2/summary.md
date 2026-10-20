# Table 1b v2 final

Test rows use the Step 2 checkpoints. They are not the later horizon reselect.

| Model | Checkpoint | How it was chosen for the test eval |
|---|---|---|
| gns_baseline k=1 | `runs/gns_baseline_s42_5716ee5b_1791828699/epoch_06.pt` (epoch 6, step 460000) | capped 5 s MSE on 32 validation runs |
| gns_local | `runs/gns_local_s42_88dc163f_1791760366/best.pt` (epoch 13, step 460000) | capped 5 s MSE on 32 validation runs |

noise_k = 1 is the validation winner under both rules (max median TTF, then min capped 5 s MSE) among the selected checkpoint of each k. The k=1 row in that comparison is the post-hoc reselect `epoch_04.pt`, which was not re-evaluated on the test set.

## noise_k selection

Cap for every training run: `max_epochs: 15`. k = 0.1, 0.3, 3 also use early-stop patience 2.

| k | epochs trained | stop | best epoch | selected checkpoint | TTF median | capped 5 s MSE | wins max TTF | wins min 5 s MSE |
|---|---|---|---|---|---|---|---|---|
| 1 | 10 observed before the cost cap (post-hoc window) | cost cap $26; post-hoc early stop not reached | 9 (val rollout) | `epoch_04.pt` | 1.000 | 0.417 | yes | yes |
| 0.1 | 12 | early stop | 9 | `last.pt` | 0.285 | 19.78 | | |
| 0.3 | 14 | early stop | 11 | `epoch_12.pt` | 0.690 | 5.76 | | |
| 3 | 5 | early stop | 2 | `best.pt` | 0.290 | 2.69 | | |

gns_local training (separate from the noise sweep) stopped on `max_steps` after 14 epochs; best epoch 13 is the file evaluated on test.

## GPU / CPU hours

Rate used for the cost-cap conversion: $0.853624312/h. Checkpoint-selection wall time was not written as its own field; the gaps below are the clock time between `TRAIN_DONE` and the next `TRAIN` / `SELECT` line.

| Job | Hours | Commit |
|---|---|---|
| train gns_local | 11.56 | `88dc163f` |
| train gns_baseline k=1 | 30.46 (from the $26 cost cap; `train_time_s` was not stored) | `5716ee5b` |
| Step 2 eval gns_local | 3.53 (`EVAL_LOCAL_WALL_S=12705`) | `e4b560f8` |
| Step 2 eval gns_baseline epoch_06 | 7.86 (`EVAL_BASE_WALL_S=28309`) | `e4b560f8` |
| train k=0.1 | 34.38 | `e4b560f8` |
| horizon select after k=0.1 | ~0.23 (not logged) | `e4b560f8` |
| train k=0.3 | 40.08 | `e4b560f8` |
| horizon select after k=0.3 | ~0.23 (not logged) | `e4b560f8` |
| train k=3 | 14.51 | `e4b560f8` |
| horizon select k=1 window | ~0.51 (not logged) | `e4b560f8` |
| CV eval_v3 (c2-standard-30) | 3.16 (`EVAL_WALL_S_constant_velocity=11375`) | `5e1a56d` |
| FA eval_v3 (c2-standard-30) | 2.63 (`EVAL_WALL_S_follow_air=9480`) | `5e1a56d` |

VM `poliphysics-tema1-ml-train`: **TERMINATED**.  
VM `poliphysics-tema1-ml-baselines-cpu2` (CV/FA v3): created for this re-eval; stopped after assemble (see PROVENANCE).

## |ΔP| per grid point (split, N, σ_w, x)

Recomputed from stored per-run JSON. Sums match the split manifest: 590 / 150 / 1850 / 300. Cell counts match `splits.json` for data `5c60fa669a719404` (test_interp has 41 occupied cells, not a full 3×2×7 rectangle).

| Model | test_interp mean / max | unseenN_interp | unseenN | unseenSigma |
|---|---|---|---|---|
| CV (v3) | 0.139 / 0.450 | 0.207 / 0.320 | 0.259 / 0.625 | 0.167 / 0.280 |
| CV (old, pre-`0cb9a0e`) | 0.139 / 0.450 | 0.207 / 0.320 | 0.259 / 0.625 | 0.167 / 0.280 |
| FA (v3) | 0.430 / 0.900 | 0.240 / 0.460 | 0.376 / 0.700 | 0.250 / 0.420 |
| gns_local | 0.465 / 1.000 | 0.360 / 0.800 | 0.516 / 1.000 | 0.360 / 0.820 |
| gns_baseline k=1 | 0.455 / 1.000 | 0.367 / 0.800 | 0.336 / 0.560 | 0.383 / 0.840 |

CV \|ΔP\| old vs v3 agrees to numerical noise (spider path was already correct for CV). Files: `P/tab_P_grid_CV_v3_by_x.csv`, `P/tab_P_grid_FA_v3_by_x.csv`.

## FA (v3)

`FollowAir` is massless tracers for every node including the spider: `v_k = u(x_k^t, t)` for all three components (air at current absolute positions and current time). No `w_s` branch. Spider absolute position advances with `v_0`; thread positions are re-centered after each step. Per-run JSON schema v3 stores `z_0_series` on the same grid as `t_series`.

On the 20 diagnostic val runs, FA had **0 timeouts** (simulator also 0). Full test: FA `|ΔP|` is now written into `table1b_v2_final.csv`.

## Metric notes

RMS columns are `1000 * sqrt(capped mean MSE)` with the existing caps `ERROR_CAP_REL_MSE = ERROR_CAP_ABS_MSE = 100` m². Thread columns use `rel_mse_mean`; spider columns use `spider_abs_mse_mean`. Both means are already the capped means inside `metrics.json`.

`|ΔP|` in the table is the mean and max over `(N, σ_w, x)` points. The `abs_delta_P` field inside `metrics.json` is the older pool over `x` and is not copied into the table.

Speed-up is one number per split from `metrics.json`. Simulator hardware is `c2-standard-30`. GNS surrogate hardware is the L4. CV and FA surrogates are CPU.

## Stale artifacts superseded

Do not use these for Table 1b:

- `paper_report_pull/report/` — pre-`0cb9a0e` GNS rows. Full list in `Tema 1/results/ml/STALE_ARTIFACTS.md`.
- `final_v2/P/tab_P_grid_CV.csv`, `tab_P_grid_FA.csv`, … without `_by_x` — `q` rounded to 0.
- `metrics/CV.json` and `metrics/FA.json` (pre-fix, commit `3118364`) — superseded by `CV_v3.json` / `FA_v3.json`.
- `table1b_v2_final_prev.csv` — snapshot before CV/FA v3 rows.

Replacement: `final_v2/table1b_v2_final.csv` and `final_v2/P/tab_P_grid_*_by_x.csv` (CV/FA: `*_v3_by_x.csv`).

## Scripts

- CV and FA paper eval: `ml/cloud/run_baselines_cpu_vm.sh`. It writes `eval_v3/` only.
- Table 1b: `python3 -m ml.scripts.assemble_table1b --splits <splits.json>` from `Tema 1`. It reads `eval_v3/{constant_velocity,follow_air}/per_run` and `metrics/{CV_v3,FA_v3,gns_local,gns_baseline}.json`.
- `runs/_baseline_*_cpu_paper/` is the pre-fix eval. Do not use it for Table 1b.
