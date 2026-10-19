# Paper artifact provenance

Code commit is the git commit recorded in `metrics.json` (`git.commit`) or in `run_meta.json` (`git_commit`) for the checkout that wrote the file. It is not the short token embedded in a run directory name. Data version `5c60fa669a719404` is the directory named by the config used for that run. Bucket prefix: `gs://authorship-verification-poliphysics/tema1-ml`.

| Artifact | Local path | gs:// | Code commit | Config | Data version | Date | Status |
|---|---|---|---|---|---|---|---|
| gns_local test eval (`best.pt`, epoch 13) | `Tema 1/results/ml/paper_report_pull/final_v2/metrics/gns_local.json` | `runs/gns_local_s42_88dc163f_1791760366/eval/metrics.json` and `eval/per_run/` | `e4b560f8e1e64a18ffddad502a55fa9f4137e5f2` | `ml/configs/gns_local_gpu.yaml` | `5c60fa669a719404` | 2026-10-15 | valid |
| gns_baseline k=1 test eval (`epoch_06.pt`) | `Tema 1/results/ml/paper_report_pull/final_v2/metrics/gns_baseline.json` | `runs/gns_baseline_s42_5716ee5b_1791828699/eval/metrics.json` and `eval/per_run/` | `e4b560f8e1e64a18ffddad502a55fa9f4137e5f2` | `ml/configs/gns_baseline_gpu.yaml` | `5c60fa669a719404` | 2026-10-15 | valid |
| noise_k selection table | `Tema 1/results/ml/paper_report_pull/final_v2/selection/noise_k_selection_table.json` | `logs/table1b_final/noise_k_selection_table.json` | `e4b560f8e1e64a18ffddad502a55fa9f4137e5f2` | `ml/configs/gns_baseline_gpu.yaml` (k written into `gns_baseline_k{0.1,0.3,3}.yaml`) | `5c60fa669a719404` | 2026-10-19 | valid |
| ckpt select k=0.1 | `.../final_v2/selection/ckpt_select_k0.1.json` | `logs/table1b_final/ckpt_select_k0.1.json` | `e4b560f8e1e64a18ffddad502a55fa9f4137e5f2` | `gns_baseline_k0.1.yaml` | `5c60fa669a719404` | 2026-10-17 | valid |
| ckpt select k=0.3 | `.../final_v2/selection/ckpt_select_k0.3.json` | `logs/table1b_final/ckpt_select_k0.3.json` | `e4b560f8e1e64a18ffddad502a55fa9f4137e5f2` | `gns_baseline_k0.3.yaml` | `5c60fa669a719404` | 2026-10-19 | valid |
| ckpt select k=3 | `.../final_v2/selection/ckpt_select_k3.json` | `logs/table1b_final/ckpt_select_k3.json` | `e4b560f8e1e64a18ffddad502a55fa9f4137e5f2` | `gns_baseline_k3.yaml` | `5c60fa669a719404` | 2026-10-19 | valid |
| ckpt select k=1 (post-hoc window; selected `epoch_04.pt`, not the test eval) | `.../final_v2/selection/ckpt_select_k1.json` | `logs/table1b_final/ckpt_select_k1.json` | `e4b560f8e1e64a18ffddad502a55fa9f4137e5f2` | `ml/configs/gns_baseline_gpu.yaml` | `5c60fa669a719404` | 2026-10-19 | valid |
| CV metrics | `.../final_v2/metrics/CV.json` | `runs/_baseline_const_cpu_paper/eval/metrics.json` | `3118364d2aae8ed68fc49fa3819da7a7633f7fc5` | `ml/configs/baseline_const_cpu.yaml` | `5c60fa669a719404` | 2026-10-14 | trajectory metrics, TTF, physical fraction, and Δ(R/L) stale (written by `full_rollout_surrogate` before `0cb9a0e`); P and spider-absolute errors valid |
| FA metrics | `.../final_v2/metrics/FA.json` | `runs/_baseline_follow_air_cpu_paper/eval/metrics.json` | `3118364d2aae8ed68fc49fa3819da7a7633f7fc5` | `ml/configs/baseline_follow_air_cpu.yaml` | `5c60fa669a719404` | 2026-10-15 | stale (same pre-`0cb9a0e` rollout path, and spider vertical velocity forced to 0) |
| P-by-x CSV, CV | `.../final_v2/P/tab_P_grid_CV_by_x.csv` | not in bucket | `31848dc223087c025b4ffa98f5400f1e666de00f` | none (recomputed on the Mac from stored per-run JSON) | `5c60fa669a719404` | 2026-10-19 | valid |
| P-by-x CSV, gns_local | `.../final_v2/P/tab_P_grid_gns_local_by_x.csv` | not in bucket | `31848dc223087c025b4ffa98f5400f1e666de00f` | none | `5c60fa669a719404` | 2026-10-19 | valid |
| P-by-x CSV, gns_baseline | `.../final_v2/P/tab_P_grid_gns_baseline_by_x.csv` | not in bucket | `31848dc223087c025b4ffa98f5400f1e666de00f` | none | `5c60fa669a719404` | 2026-10-19 | valid |
| P-by-x CSV, FA | `.../final_v2/P/tab_P_grid_FA_by_x.csv` | not in bucket | `31848dc223087c025b4ffa98f5400f1e666de00f` | none | `5c60fa669a719404` | 2026-10-19 | stale (`p_usable=false`; surrogate status is timeout on every test run) |
| `table1b_v2_final.csv` | `.../final_v2/table1b_v2_final.csv` | not in bucket | unknown | none | `5c60fa669a719404` | 2026-10-19 | valid for columns taken from the valid rows above; FA `|ΔP|` left blank. Writer `final_v2/build_table1b.py` is untracked, so its commit is unknown |
| `summary.md` | `.../final_v2/summary.md` | not in bucket | unknown | none | `5c60fa669a719404` | 2026-10-19 | valid as the session note. Writer is untracked |

Pooled `final_v2/P/tab_P_grid_*.csv` files without `_by_x` are superseded (they group on charge `q`, which rounds to 0).
