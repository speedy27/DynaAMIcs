# EXP1 generalization across gLV instances (MEASURED)

HYBRID mc=0.3 world model per instance. `mppi_latent` = LEARNED raw-latent-distance MPPI (the clean closure test). ORACLE = true-dynamics MPPI (controllability ref). Same episodes (3 seeds x 12) + tol per instance.

| instance | guilds/S/K | tol | oracle succ/final | mppi_latent succ/final | crosses tol | near-oracle |
|---|---|---|---|---|---|---|
| g4_s24 | 4/24/24 | 0.861 | 100% / 0.65 | 89%±7 / 0.78 | yes | no |
| g3_s18 | 3/18/18 | 0.862 | 100% / 0.61 | 100%±0 / 0.74 | yes | yes |
| g5_s30 | 5/30/30 | 0.861 | 100% / 0.67 | 100%±0 / 0.65 | yes | yes |
| g3_s24_strongcomp | 3/24/24 | 0.988 | 100% / 0.79 | 89%±7 / 0.95 | yes | no |