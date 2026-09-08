# Validation

- 13 new diagnostics tests passed; 40 existing D-series/checkpoint diagnostics tests passed.
- Native T=1 and alpha=.5 predictions are bitwise identical over the complete VAL and TEST loaders.
- All saved prediction arrays reproduce every reported MAE/RMSE/Hit exactly.
- Routing-only interventions preserve fixed p exactly; alpha interventions use q=p exactly.
- Fixed alpha=.5 full Z trajectory max difference = 0.
- Model state_dict and checkpoint SHA256 are unchanged; final native prediction difference = 0.
- No optimizer/scheduler step or training epoch.
- Temperature argmax occupancy unchanged.
- Temporal-prefix perturbation checks passed. Unrestricted shuffle is explicitly user-authorized NON-calendar-causal stress testing.

See `validation_initial.log`, `validation_new.log`, `results.json` and `run.log`.
