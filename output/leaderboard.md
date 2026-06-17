# Leaderboard

## Detectors — ranked by overall AUC

| Rank | Detector | AUC | Toughest adversary (AUC) | # benign | # malign |
| ---: | :--- | ---: | :--- | ---: | ---: |
| 1 | laplacian_spectrum | 1.0000 | tetwild (1.000) | 1000 | 4191 |
| 2 | image_classifier | 0.9989 | tetwild (0.998) | 1000 | 4191 |
| 3 | pointnet | 0.9938 | tetwild (0.992) | 1000 | 4191 |
| 4 | file_hash | 0.5001 | degenerate (0.500) | 1000 | 4191 |
| 5 | always_benign | 0.5000 | degenerate (0.500) | 1000 | 4191 |
| 6 | always_malign | 0.5000 | degenerate (0.500) | 1000 | 4191 |
| 7 | always_random | 0.4840 | degenerate (0.469) | 1000 | 4191 |

## Detectors — timing (most recent run)

| Detector | Install (s) | Train (s) | Load (s) | Eval (ms/shape) | # eval |
| :--- | ---: | ---: | ---: | ---: | ---: |
| laplacian_spectrum | — | 781.0 | — | 107.2 | 5191 |
| image_classifier | 0.7 | 5078.4 | — | 118.8 | 5191 |
| pointnet | — | 7130.1 | — | 45.9 | 5191 |
| file_hash | — | 2.6 | — | 0.7 | 5191 |
| always_benign | — | 0.0 | — | 0.0 | 5191 |
| always_malign | — | 0.0 | — | 0.0 | 5191 |
| always_random | — | 0.0 | — | 0.0 | 5191 |

## Adversaries — ranked by evasion (mean AUC across detectors, lowest first)

| Rank | Adversary | Mean AUC | Best detector (AUC) | # malign |
| ---: | :--- | ---: | :--- | ---: |
| 1 | degenerate | 0.7089 | laplacian_spectrum (1.000) | 700 |
| 2 | rigid | 0.7093 | laplacian_spectrum (1.000) | 700 |
| 3 | tetwild | 0.7105 | laplacian_spectrum (1.000) | 691 |
| 4 | faceswap | 0.7105 | laplacian_spectrum (1.000) | 700 |
| 5 | disconnected | 0.7123 | laplacian_spectrum (1.000) | 700 |
| 6 | jitter | 0.7144 | laplacian_spectrum (1.000) | 700 |

## Detector × adversary AUC matrix

| Adversary \ Detector | always_benign | always_malign | always_random | file_hash | image_classifier | laplacian_spectrum | pointnet | Mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| degenerate | 0.500 | 0.500 | 0.469 | 0.500 | 0.999 | 1.000 | 0.994 | 0.709 |
| rigid | 0.500 | 0.500 | 0.473 | 0.500 | 0.998 | 1.000 | 0.994 | 0.709 |
| tetwild | 0.500 | 0.500 | 0.483 | 0.500 | 0.998 | 1.000 | 0.992 | 0.711 |
| faceswap | 0.500 | 0.500 | 0.480 | 0.501 | 1.000 | 1.000 | 0.993 | 0.711 |
| disconnected | 0.500 | 0.500 | 0.493 | 0.500 | 0.998 | 1.000 | 0.994 | 0.712 |
| jitter | 0.500 | 0.500 | 0.506 | 0.500 | 1.000 | 1.000 | 0.995 | 0.714 |
| **Mean** | 0.500 | 0.500 | 0.484 | 0.500 | 0.999 | 1.000 | 0.994 | 0.711 |
