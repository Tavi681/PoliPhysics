# gns_baseline failure diagnostics

- checkpoint: `/home/octav/poliphysics/Tema 1/results/ml/runs/gns_baseline_s42_5716ee5b_1791828699/epoch_06.pt`
- n_runs: 20
- physical_edge_tol: 0.1

## Outcomes (simulator vs surrogate)

| run_id | N | sim | surr | ttf_s | first_fail_t | edge (i,j) |
| --- | ---: | --- | --- | ---: | ---: | --- |
| N1_sw0.15_x+4_i0099 | 1 | rise | fall | 0.01 | 0.01 | (0,1) |
| N1_sw0.30_x+1_i0114 | 1 | rise | fall | 0.02 | 0.02 | (0,1) |
| N2_sw0.15_x-1_i0114 | 2 | fall | fall | 0.01 | 0.01 | (0,1) |
| N2_sw0.30_x+1_i0047 | 2 | fall | fall | 0.02 | 0.02 | (0,1) |
| N4_sw0.15_x+1_i0029 | 4 | fall | fall | 0.01 | 0.01 | (0,1) |
| N4_sw0.30_x+1_i0196 | 4 | rise | fall | 0.01 | 0.01 | (0,51) |
| N1_sw0.15_x+1_i0035 | 1 | rise | fall | 0.01 | 0.01 | (0,1) |
| N1_sw0.30_x+1_i0032 | 1 | rise | fall | 0.02 | 0.02 | (0,1) |
| N2_sw0.15_x-1_i0057 | 2 | fall | fall | 0.01 | 0.01 | (0,1) |
| N2_sw0.30_x+0_i0037 | 2 | fall | fall | 0.01 | 0.01 | (0,1) |
| N4_sw0.15_x+0_i0188 | 4 | fall | fall | 0.01 | 0.01 | (0,1) |
| N4_sw0.30_x+1_i0030 | 4 | fall | fall | 0.03 | 0.03 | (0,51) |
| N1_sw0.15_x+4_i0075 | 1 | rise | fall | 0.01 | 0.01 | (0,1) |
| N1_sw0.30_x+2_i0058 | 1 | rise | fall | 0.01 | 0.01 | (0,1) |
| N2_sw0.15_x-1_i0007 | 2 | fall | fall | 0.02 | 0.02 | (0,1) |
| N2_sw0.30_x+2_i0097 | 2 | rise | fall | 0.01 | 0.01 | (0,1) |
| N4_sw0.15_x+0_i0075 | 4 | rise | fall | 0.01 | 0.01 | (0,51) |
| N4_sw0.30_x+1_i0076 | 4 | rise | fall | 0.01 | 0.01 | (0,1) |
| N1_sw0.15_x+0_i0190 | 1 | rise | fall | 0.01 | 0.01 | (0,1) |
| N1_sw0.30_x+2_i0086 | 1 | rise | fall | 0.01 | 0.01 | (0,1) |

## Plots

- `/home/octav/poliphysics/Tema 1/results/ml/diagnostics/gns_baseline_epoch06_val20/edge_dev_vs_time_N1.png`
- `/home/octav/poliphysics/Tema 1/results/ml/diagnostics/gns_baseline_epoch06_val20/edge_dev_vs_time_N2.png`
- `/home/octav/poliphysics/Tema 1/results/ml/diagnostics/gns_baseline_epoch06_val20/edge_dev_vs_time_N4.png`
- `/home/octav/poliphysics/Tema 1/results/ml/diagnostics/gns_baseline_epoch06_val20/spider_z_N1_sw0.15_x+4_i0099.png`
- `/home/octav/poliphysics/Tema 1/results/ml/diagnostics/gns_baseline_epoch06_val20/spider_z_N1_sw0.30_x+1_i0114.png`
- `/home/octav/poliphysics/Tema 1/results/ml/diagnostics/gns_baseline_epoch06_val20/spider_z_N2_sw0.15_x-1_i0114.png`
- `/home/octav/poliphysics/Tema 1/results/ml/diagnostics/gns_baseline_epoch06_val20/spider_z_N2_sw0.30_x+1_i0047.png`
- `/home/octav/poliphysics/Tema 1/results/ml/diagnostics/gns_baseline_epoch06_val20/spider_z_N4_sw0.15_x+1_i0029.png`
- `/home/octav/poliphysics/Tema 1/results/ml/diagnostics/gns_baseline_epoch06_val20/spider_z_N4_sw0.30_x+1_i0196.png`
- `/home/octav/poliphysics/Tema 1/results/ml/diagnostics/gns_baseline_epoch06_val20/one_step_pre_fail_by_type.png`
