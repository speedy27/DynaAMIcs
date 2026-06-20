# EXP2 IDM-reweight self-supervised closure (MEASURED)

metric_coeff=0 (pure JEPA). IDM uses only (z_t,z_{t+1})->action — self-supervised, no true-state distance. IDM induces a CONTROL metric, not the Euclidean state metric tol is defined on.

| variant | d | mppi_latent succ/final | latent↔true Spearman | free-run 6-step | recog guild | recog basin |
|---|---|---|---|---|---|---|
| pure-JEPA idm=1.0 (ref, fails) | 128 | 0% / 4.53 | +0.085 | 0.084 | 0.899 | 0.690 |
| HYBRID mc=0.3 (upper bar) | 128 | 100% / 0.80 | +0.990 | 0.283 | 0.971 | 0.812 |
| idm=2 | 128 | 0% / 4.33 | -0.017 | 0.084 | 0.896 | 0.731 |
| idm=5 | 128 | 0% / 4.26 | -0.196 | 0.096 | 0.901 | 0.725 |
| idm=10 | 128 | 0% / 4.38 | -0.313 | 0.086 | 0.887 | 0.696 |