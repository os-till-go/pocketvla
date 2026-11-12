# Archived result statistics

Generated with `python tools/compare_results.py` (5000 paired bootstrap samples; seed 0).

| Run | Split | Success | Wilson 95% CI | Grasp |
| --- | --- | --- | --- | --- |
| SFT | ID | 395/500 (79.0%) | 75.2-82.3% | 82.4% |
| SFT | OOD | 93/300 (31.0%) | 26.0-36.4% | 58.0% |
| GRPO20 | ID | 405/500 (81.0%) | 77.3-84.2% | 82.6% |
| GRPO20 | OOD | 92/300 (30.7%) | 25.7-36.1% | 58.7% |
| G1 | ID | 393/500 (78.6%) | 74.8-82.0% | 81.2% |
| G1 | OOD | 97/300 (32.3%) | 27.3-37.8% | 60.7% |
| G3 | ID | 390/500 (78.0%) | 74.2-81.4% | 81.6% |
| G3 | OOD | 103/300 (34.3%) | 29.2-39.9% | 60.7% |

| Comparison (B - A) | Split | Delta (pp) | Gained / lost | McNemar p | Bootstrap 95% CI (pp) |
| --- | --- | --- | --- | --- | --- |
| GRPO20 - SFT | ID | +2.0 | 29 / 19 | 0.1934 | [-0.6, +4.6] |
| G1 - SFT | ID | -0.4 | 42 / 44 | 0.9142 | [-4.0, +3.4] |
| G3 - SFT | ID | -1.0 | 30 / 35 | 0.6201 | [-4.0, +2.2] |
| G3 - G1 | ID | -0.6 | 37 / 40 | 0.8199 | [-4.2, +2.8] |
| GRPO20 - SFT | OOD | -0.3 | 9 / 10 | 1.0000 | [-3.0, +2.3] |
| G1 - SFT | OOD | +1.3 | 17 / 13 | 0.5847 | [-2.0, +5.0] |
| G3 - SFT | OOD | +3.3 | 23 / 13 | 0.1325 | [-0.3, +7.3] |
| G3 - G1 | OOD | +2.0 | 22 / 16 | 0.4177 | [-2.0, +6.0] |

Pairing assumes the saved reports replayed the same scenarios. Export-time hashes are not historical provenance. CIs cover scenario sampling, not training-seed variation; p-values are unadjusted exploratory comparisons.
