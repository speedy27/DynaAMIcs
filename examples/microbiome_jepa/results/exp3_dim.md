# EXP3 bottleneck shrink (latent dim toward true-state dim=24) (MEASURED)

metric_coeff=0 (pure JEPA). pure JEPA; latent dim swept toward the true gLV state dim S=24.

| variant | d | mppi_latent succ/final | latent↔true Spearman | free-run 6-step | recog guild | recog basin |
|---|---|---|---|---|---|---|
| pure-JEPA idm=1.0 (ref, fails) | 128 | 0% / 4.53 | +0.085 | 0.084 | 0.899 | 0.690 |
| HYBRID mc=0.3 (upper bar) | 128 | 100% / 0.80 | +0.990 | 0.283 | 0.971 | 0.812 |
| d=16 | 16 | 0% / 4.82 | +0.254 | 0.041 | 0.672 | 0.575 |
| d=24 | 24 | 0% / 5.09 | +0.197 | 0.050 | 0.676 | 0.533 |
| d=32 | 32 | 0% / 4.23 | +0.237 | 0.057 | 0.747 | 0.581 |