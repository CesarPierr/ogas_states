# Comprehensive Statistical Benchmark Matrix: 1D Burgers Profile B (10 Seeds)

**Regime:** High-Reynolds ($Re \in [2000, 10000]$, $\nu \in [10^{-4}, 5 \times 10^{-4}]$), Razor-sharp persistent shocks ($T=40$ steps rollout).

## 1. Multi-Step Autoregressive Rollout Benchmark ($T=40$ Horizons)

| Strategy | Completed | Rollout NRMSE Mean | Rollout NRMSE Median [IQR] | Rollout Final Step ($T=40$) | Rollout Tail Error (p95) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Uniform Baseline (On-Attractor)** | 10/10 | `0.0335 +/- 0.0422` | `0.0175 [0.0123]` | `0.0713 +/- 0.0926` | `0.0773 +/- 0.0935` |
| **Heuristic Tube Perturbations** | 10/10 | `0.0339 +/- 0.0360` | `0.0158 [0.0049]` | `0.0676 +/- 0.0676` | `0.0800 +/- 0.0817` |
| **Classical AL (Top-K Uncertainty)** | 10/10 | `0.0579 +/- 0.0526` | `0.0288 [0.0372]` | `0.1225 +/- 0.1116` | `0.1452 +/- 0.1391` |
| **Classical AL (SBAL Sampling)** | 10/10 | `0.0567 +/- 0.0487` | `0.0303 [0.0672]` | `0.1211 +/- 0.0966` | `0.1240 +/- 0.0998` |
| **OGAS (Generative Active Sampling)** | 10/10 | `0.0308 +/- 0.0061` | `0.0311 [0.0104]` | `0.0677 +/- 0.0146` | `0.0666 +/- 0.0122` |

## 2. 1-Step Prediction & Out-of-Distribution Tail Errors

| Strategy | 1-Step NRMSE Mean | 1-Step NRMSE Median [IQR] | 1-Step Tail p90 | 1-Step Tail p95 | 1-Step Tail p99 (Worst Case) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Uniform Baseline (On-Attractor)** | `0.0031 +/- 0.0038` | `0.0018 [0.0004]` | `0.0070 +/- 0.0089` | `0.0085 +/- 0.0099` | `0.0138 +/- 0.0117` |
| **Heuristic Tube Perturbations** | `0.0043 +/- 0.0047` | `0.0021 [0.0002]` | `0.0105 +/- 0.0120` | `0.0130 +/- 0.0138` | `0.0199 +/- 0.0162` |
| **Classical AL (Top-K Uncertainty)** | `0.0054 +/- 0.0048` | `0.0023 [0.0057]` | `0.0121 +/- 0.0106` | `0.0149 +/- 0.0119` | `0.0229 +/- 0.0137` |
| **Classical AL (SBAL Sampling)** | `0.0054 +/- 0.0049` | `0.0027 [0.0059]` | `0.0116 +/- 0.0101` | `0.0144 +/- 0.0115` | `0.0231 +/- 0.0160` |
| **OGAS (Generative Active Sampling)** | `0.0028 +/- 0.0003` | `0.0029 [0.0004]` | `0.0064 +/- 0.0006` | `0.0086 +/- 0.0010` | `0.0150 +/- 0.0020` |