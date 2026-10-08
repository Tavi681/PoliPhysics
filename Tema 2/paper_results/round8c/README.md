Round-8c CSVs stamped `code_dirty=true` differ from commit `a351fff` only by
the opt-in `NumericsConfig.continue_after_arrest` flag (default off) and the
diagnostic `t_arrest` / `t_frame_contact` fields in the integrator. Production
defaults still stop at the first vertical rebound and are bit-identical to
`a6c4705`.
