Round-8b CSVs stamped `code_dirty=true` differ from commit `a6c4705` only by the
opt-in `shock_visc` / `fail_avg_n_seg` knobs (both default off), cache file
locking in `netsim/mmin.py`, and the GCP launcher/supervisor. Production
defaults are bit-identical to `a6c4705`.

`nesting_ok` is 1 iff, at **every** impact point of that row, the three masses
satisfy `m^A >= min(m^{B_loc}) >= m^{B_any}` with a `1e-12` gram absolute
tolerance (not the 1% bisection tolerance). The flag is copied onto all 30
lines of the row, so 8 failing cases × 30 lines = 240 of 360. Those 8 cases
fail only at a few hub (and two ring-D) points, where B sits ~0.2% above A
because the A and B bisections stop independently at 1%. Within 1% the masses
are nested on every row and point.
