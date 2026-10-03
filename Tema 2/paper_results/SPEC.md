# paper_results/SPEC.md: runs and CSV formats needed for the paper

Put this file at `paper_results/SPEC.md`. Each item below is one CSV in `paper_results/`, written by `validation/export_paper.py`.

## General rules

- **CSV header.** Every CSV begins with a comment line `# commit=<hash> date=<ISO> config=<file>`, then one header row, then the data.
- **Units and number format.** SI units. Store ratios as plain numbers. Use at least 4 significant digits.
- **Run parameters.** Unless stated otherwise:
  - star net, \(R=1\) m;
  - \(\epsilon_p=0.1\,\epsilon_b\);
  - `gripped` contact (baseline; `frictionless` is a low-priority sensitivity study, not needed now);
  - \(n_s=40\), \(C=0.5\);
  - the working values of \(r_d\), \(k_c\), \(R_{max}\), \(k_{max}\) from the config.
- **Parameter file.** Also export `params_used.csv`, with columns `symbol,value,unit,comment`, holding every value that goes into the appendix parameter table.
- **Raw data.** Do not send HDF5 files, except the 2–4 runs listed for Fig. `phase`.

## Validation (Sec. "Validation of the simulator")

| File | Table/figure | Runs | Columns |
|---|---|---|---|
| `tab_smith.csv` | Tab. smith | S, D × \(v_0\) ∈ {100, 500, 1000} m/s, finest resolution | `material,v0,eps_an,eps_num,eps_err_pct,cL_an,cL_num,cL_err_pct,cT_an,cT_num,cT_err_pct,n_seg` |
| `tab_smith_conv.csv` | supplement | S, \(v_0\)=500, three resolutions | `n_seg,eps_err_pct,cL_err_pct,cT_err_pct` |
| `fig_smith.csv` | Fig. smith | S, \(v_0\)=500, three times | `t,X,eps,x,y` (long format) |
| `tab_etaA.csv` | Tab. etaA | S, D × \(\epsilon_p/\epsilon_b\) ∈ {0, 0.1, 0.3}, quasi-static hub | `material,ep_frac,eta_an,eta_num,wb_an,wb_num` |
| `fig_Fw.csv` | Fig. Fw | S, D, N=8, \(\epsilon_p=0.1\epsilon_b\), quasi-static | `material,w_over_R,F_an,F_num,failed` |
| `tab_junction.csv` | Tab. junction | N ∈ {2,3,4,6,8,12,16}, S and D, \(\epsilon_p/\epsilon_b\)=0.1 | `material,N,SN,SN2d,T0_1d,T0_2d,T0_num,dTmax_1d,dTmax_2d,dTmax_num,dTmin_1d,dTmin_2d,dTmin_num` |
| `junction_scaling.csv` | supplement | D, N=8, \(\epsilon_p/\epsilon_b\) ∈ {0.02, 0.1, 0.3} | `material,ep_frac,ZT_over_ZL,T0_1d,T0_2d,T0_num` |
| `tab_conv.csv` | Tab. conv | S, N=8, M=1, \(v_0\)=15, \(a=R/2\), \(n_s\) ∈ {10,20,40,80}, gripped contact | `contact,n_s,wmax_over_R,d_wmax_pct,n_failed,eta,d_eta_pct,R_d,cpu_s,drone_x_arrest,drone_y_arrest` |
| `energy_conv.csv` | supplement | reference case without failure, C ∈ {0.5, 0.25, 0.125}, gripped (frictionless optional) | `contact,C,dt,energy_error` |

## Results (Sec. "Analytical versus numerical comparison")

| File | Table/figure | Runs | Columns |
|---|---|---|---|
| `fig_etaa.csv` | Fig. etaa (a) | star N=8, S and D, \(a/R\) ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}, quasi-static, \(\epsilon_p\) ∈ {0, 0.1\(\epsilon_b\)} | `net,material,ep_frac,a_over_R,etaA_num,etaB_num,etaA_ref,etaB_ref,first_failure,etaA_free_num,etaA_free_ref,px_fail` (`ref` = `ref/offc.py`; free = `ref/offc_free.py`; `first_failure` = inner/outer) |
| `fig_etaa_ring.csv` | Fig. etaa (b) | star N=8 + ring at \(R/2\), \(q_{ring}/q_{rad}\) ∈ {0.5, 1}, same \(a/R\) | same columns as `fig_etaa.csv`, with `ref` left empty, plus `q_ratio` |
| `tab_daf.csv` | Tab. daf | item 5a of the round-2 prompt, N ∈ {4,8,16}, linear and cubic regime | `material,N,n_eff_num,eps_s_over_eps0,eps_m_over_eps0,m_hub` |
| `fig_daf_ramp.csv` | new figure | N=8, D, \(t_f/T_n\) ∈ {0,0.25,0.5,1,2,4} | `N,material,tf_over_Tn,eps_m_over_eps0,daf_an` (`daf_an` = \(1+|\sin x|/x\), \(x=\pi t_f/T_n\)) |
| `dyn_runs.csv` | Fig. phase (c), Sec. localization | S and D, star+ring, M ∈ {0.25, 1, 2}, \(v_0\) ∈ {10, 15, 20}, \(a/R\) ∈ {0, 0.25, 0.5}, \(s\) on a grid around \(m^B_{min}\) | `material,net,M,v0,a_over_R,s,m_net,PiE_M_over_m,arrested,wmax_over_R,eta,n_failed,R_d,cascade,energy_error,drone_x_arrest,drone_y_arrest` |
| `fig_phase_maps/` | Fig. phase (a,b) | S and D, same net, drone and impact, criterion B at \(m^B_{min}\) | 2–4 HDF5 files, plus `segments_<run>.csv` with `seg,parent,x1,y1,x2,y2,broken,t_break` |
| `tab_mmin.csv` | Tab. mmin, Fig. mmin | Algorithm 2, star N=8 + ring at \(R/2\), S and D, M ∈ {0.25, 2}, \(v_0\)=20, and the full \((M,v_0)\) grid of Tab. regime | `material,M,v0,Ekin,m_lower_g,mA_min_g,mB_min_g,ratio,n_broken_B,worst_p_x,worst_p_y` |
| `mmin_monotone.csv` | supplement | coarse scan of item 5b | `material,criterion,s_over_shi,pass` |

## What to send me

1. **The `paper_results/` folder, as a zip archive,** with all the CSVs, `params_used.csv`, and the `fig_phase_maps/` folder.
2. **The summary text from Cursor.** This is the pass/fail list with numbers and the implementation decisions not fixed by the paper.
3. **The pytest output.**

The ML runs (dataset, splits, surrogate) are not part of this delivery.
