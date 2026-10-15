# STALE artifacts (pre-fix `full_rollout_surrogate`)

Generated for commit `0cb9a0e` (spider-relative re-centering fix).
**Do not delete.** These outputs must not be used for Table 1b GNS rows or diagnose claims.

## Legend

| Category | Meaning |
|---|---|
| `diagnose_singular_pre_0cb9a0e` | `diagnose_baseline_failure` used buggy `full_rollout_surrogate` |
| `oracle_singular_pre_0cb9a0e` | Oracle rollouts through buggy singular path |
| `gns_table1b_or_eval_pre_0cb9a0e` | GNS full-eval / paper tables from before re-eval with fixed path (user: Table 1b GNS rows invalid) |
| `ckpt_select_pre_0cb9a0e_era` | Checkpoint-select JSON from before the fix commit; select used **batched** path (already correct). Superseded by `*_after_fix.json` |

## Replacement (post-fix, still local/partial)

- Diagnose: `Tema 1/results/ml/diagnostics/gns_baseline_epoch06_val20_after_fix/`
- Oracle: `Tema 1/results/ml/diagnostics/oracle_step1_after_fix/`
- Select: `Tema 1/results/ml/logs/ckpt_select_v2_gns_*_after_fix.json`
- Full Table 1b GNS re-eval: **blocked** pending GPU budget (see estimate below / report)

## Artifact list

| category | path | note |
|---|---|---|
| `diagnose_singular_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/edge_dev_vs_time_N1.png` | diagnose_baseline_failure → full_rollout_surrogate |
| `diagnose_singular_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/edge_dev_vs_time_N2.png` | diagnose_baseline_failure → full_rollout_surrogate |
| `diagnose_singular_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/edge_dev_vs_time_N4.png` | diagnose_baseline_failure → full_rollout_surrogate |
| `diagnose_singular_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/one_step_pre_fail_by_type.png` | diagnose_baseline_failure → full_rollout_surrogate |
| `diagnose_singular_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/spider_z_N1_sw0.15_x+4_i0099.png` | diagnose_baseline_failure → full_rollout_surrogate |
| `diagnose_singular_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/spider_z_N1_sw0.30_x+1_i0114.png` | diagnose_baseline_failure → full_rollout_surrogate |
| `diagnose_singular_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/spider_z_N2_sw0.15_x-1_i0114.png` | diagnose_baseline_failure → full_rollout_surrogate |
| `diagnose_singular_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/spider_z_N2_sw0.30_x+1_i0047.png` | diagnose_baseline_failure → full_rollout_surrogate |
| `diagnose_singular_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/spider_z_N4_sw0.15_x+1_i0029.png` | diagnose_baseline_failure → full_rollout_surrogate |
| `diagnose_singular_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/spider_z_N4_sw0.30_x+1_i0196.png` | diagnose_baseline_failure → full_rollout_surrogate |
| `diagnose_singular_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/summary.json` | diagnose_baseline_failure → full_rollout_surrogate |
| `diagnose_singular_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/summary.md` | diagnose_baseline_failure → full_rollout_surrogate |
| `oracle_singular_pre_0cb9a0e` | `Tema 1/results/ml/diagnostics/oracle_step1_t2/summary.json` | oracle through buggy full_rollout_surrogate |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/baseline/runs/gns_baseline_s42_5716ee5b_1791828699/ckpt_select_gns_baseline_s42_5716ee5b_1791828699.json` | pre-fix GNS baseline GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/baseline/runs/gns_baseline_s42_5716ee5b_1791828699/ckpt_select_v2_gns_baseline_s42_5716ee5b_1791828699.json` | pre-fix GNS baseline GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/baseline/runs/gns_baseline_s42_5716ee5b_1791828699/eval/metrics.json` | pre-fix GNS baseline GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/baseline/runs/gns_baseline_s42_5716ee5b_1791828699/eval/tab_accuracy.csv` | pre-fix GNS baseline GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/baseline/runs/gns_baseline_s42_5716ee5b_1791828699/eval/tab_cost.csv` | pre-fix GNS baseline GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/baseline/runs/gns_baseline_s42_5716ee5b_1791828699/metrics.json` | pre-fix GNS baseline GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/baseline/runs/gns_baseline_s42_5716ee5b_1791828699/run_meta.json` | pre-fix GNS baseline GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/baseline/runs/gns_baseline_s42_5716ee5b_1791828699/tab_accuracy.csv` | pre-fix GNS baseline GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/baseline/runs/gns_baseline_s42_5716ee5b_1791828699/tab_cost.csv` | pre-fix GNS baseline GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/local/runs/gns_local_s42_88dc163f_1791760366/ckpt_select_gns_local_s42_88dc163f_1791760366.json` | pre-fix GNS local GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/local/runs/gns_local_s42_88dc163f_1791760366/ckpt_select_v2_gns_local_s42_88dc163f_1791760366.json` | pre-fix GNS local GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/local/runs/gns_local_s42_88dc163f_1791760366/eval/metrics.json` | pre-fix GNS local GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/local/runs/gns_local_s42_88dc163f_1791760366/eval/tab_accuracy.csv` | pre-fix GNS local GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/local/runs/gns_local_s42_88dc163f_1791760366/eval/tab_cost.csv` | pre-fix GNS local GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/local/runs/gns_local_s42_88dc163f_1791760366/eval_gns_local.log` | pre-fix GNS local GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/local/runs/gns_local_s42_88dc163f_1791760366/gns_local_gpu_train.log` | pre-fix GNS local GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/local/runs/gns_local_s42_88dc163f_1791760366/metrics.json` | pre-fix GNS local GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/local/runs/gns_local_s42_88dc163f_1791760366/run_meta.json` | pre-fix GNS local GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/local/runs/gns_local_s42_88dc163f_1791760366/tab_accuracy.csv` | pre-fix GNS local GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/local/runs/gns_local_s42_88dc163f_1791760366/tab_cost.csv` | pre-fix GNS local GPU eval / select pull |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/report/tab_accuracy_selected.csv` | Table 1b source rows include stale GNS |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/report/tab_P_grid.csv` | P-grid includes stale GNS |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/report/tab_error_vs_time.csv` | error-vs-time includes stale GNS |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/report/tab_ckpt_select.csv` | ckpt select table pre-re-eval |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/report/selected_gns_baseline.txt` |  |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/paper_report_pull/report/selected_gns_local.txt` |  |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/final_gns_local_s42_88dc163f_1791760366/eval_gns_local.log` | older GNS local final eval dir |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/final_gns_local_s42_88dc163f_1791760366/fig_curves.png` | older GNS local final eval dir |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/final_gns_local_s42_88dc163f_1791760366/gns_local_gpu_train.log` | older GNS local final eval dir |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/final_gns_local_s42_88dc163f_1791760366/metrics.json` | older GNS local final eval dir |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/final_gns_local_s42_88dc163f_1791760366/rollout_refs.json` | older GNS local final eval dir |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/final_gns_local_s42_88dc163f_1791760366/run_meta.json` | older GNS local final eval dir |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/final_gns_local_s42_88dc163f_1791760366/sanity_1h_gns_local_gpu.json` | older GNS local final eval dir |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/final_gns_local_s42_88dc163f_1791760366/sanity_final_gns_local.json` | older GNS local final eval dir |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/final_gns_local_s42_88dc163f_1791760366/tab_accuracy.csv` | older GNS local final eval dir |
| `gns_table1b_or_eval_pre_0cb9a0e` | `Tema 1/results/ml/final_gns_local_s42_88dc163f_1791760366/tab_cost.csv` | older GNS local final eval dir |
| `ckpt_select_pre_0cb9a0e_era` | `Tema 1/results/ml/logs/ckpt_select_v2_gns_local_s42_88dc163f_1791760366.json` | select uses full_rollout_batched (already re-centered); kept for audit; superseded by *_after_fix.json |
| `ckpt_select_pre_0cb9a0e_era` | `Tema 1/results/ml/logs/ckpt_select_v2_gns_baseline_s42_5716ee5b_1791828699.json` | select uses full_rollout_batched (already re-centered); kept for audit; superseded by *_after_fix.json |

## W&B runs (training itself is not invalidated by the rollout bug; GNS *eval* tables above are stale)

| model | W&B run | URL |
|---|---|---|
| gns_baseline (k=1, `5716ee5b`) | `w7ghs67l` (`devout-night-5`) | https://wandb.ai/octavian-radu-upb/poliphysics-ballooning-ml/runs/w7ghs67l |
| gns_local (`88dc163f`) | `3k23eo7m` (`chocolate-aardvark-3`) | https://wandb.ai/octavian-radu-upb/poliphysics-ballooning-ml/runs/3k23eo7m |

## GPU wall-time estimate (Steps 2+3) — **STOP: exceeds 48 h**

Sources: `Tema 1/results/ml/logs/sanity_watch/train_vm_tail_at_stop.log` (baseline train `elapsed_s≈109555` at epoch 10 → **~3.0 h/epoch** wall incl. validation); GPU select walls in `ckpt_select_v2_gns_baseline_s42_5716ee5b_1791828699.json` (~**250 s/ckpt** for 32 val runs); prior full test eval `paper_report_pull/*/eval/metrics.json` (baseline **~1.3 h**, local **~0.4 h** for 2890 test runs).

| Work | Estimate |
|---|---|
| Step 2: full test eval ×2 (local + baseline k=1) | ~1.7 h |
| Step 3 train: 3×`noise_k`∈{0.1,0.3,3}, same recipe as k=1 | **3 × ~11 epochs × ~3.0 h/epoch ≈ 99 h** (k=1 ran to epoch 10 under cost cap; config `max_epochs=15`, early-stop patience 2) |
| Step 3 select: 3 new k × ~13 ckpts × ~250 s | ~2.7 h |
| Step 2 again if winning k≠1 | ~1.3 h |
| **Total** | **≳ 100 h GPU ≫ 48 h** |

Optimistic early-stop (e.g. 5 epochs ×3): still ~45 h train + select/eval ≈ **~50 h**, still over the cap. **VM not started.**

