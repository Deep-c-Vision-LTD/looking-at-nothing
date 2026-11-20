# Full-data readouts: results (2026-11-19 21:41:37)

Cluster bootstrap 10,000 (cells, positions within cells), 95% CI; seeds averaged per configuration; size avg = mean of the 336 and 448 px effects. Validity: MIT1003 CC >= CC(R4) - 0.03.

## MIT1003 held-out CC (mean of seeds)

| features | R1n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| dinov2reg_L12 | 0.708 (n 3) | 0.706 (n 3) | 0.722 (n 3) | 0.726 (n 3) |  |
| dinov2reg_L6 | 0.700 (n 3) | 0.690 (n 3) | 0.723 (n 3) | 0.723 (n 3) |  |

## Kanizsa illusory - control180, inside AOI (size avg)

| features | R1n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| dinov2reg_L12 | +0.003 [-0.040, +0.043] p = .863 (seeds 3) | -0.035 [-0.064, -0.008] p = .008 (seeds 3) | +0.025 [-0.008, +0.059] p = .135 (seeds 3) | +0.003 [-0.033, +0.040] p = .856 (seeds 3) |  |
| dinov2reg_L6 | +0.067 [+0.055, +0.082] p < .001 (seeds 3) | +0.038 [+0.024, +0.053] p < .001 (seeds 3) | +0.106 [+0.090, +0.123] p < .001 (seeds 3) | +0.098 [+0.072, +0.124] p < .001 (seeds 3) |  |

## Kanizsa illusory - control180, core AOI (r_core30) (size avg)

| features | R1n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| dinov2reg_L12 | +0.001 [-0.043, +0.043] p = .948 (seeds 3) | -0.031 [-0.058, -0.005] p = .017 (seeds 3) | +0.020 [-0.010, +0.051] p = .202 (seeds 3) | +0.000 [-0.040, +0.040] p = .979 (seeds 3) |  |
| dinov2reg_L6 | +0.057 [+0.039, +0.077] p < .001 (seeds 3) | +0.032 [+0.016, +0.050] p < .001 (seeds 3) | +0.097 [+0.086, +0.108] p < .001 (seeds 3) | +0.076 [+0.054, +0.100] p < .001 (seeds 3) |  |

## Kanizsa illusory - control90, inside AOI (size avg)

| features | R1n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| dinov2reg_L12 | +0.044 [+0.006, +0.081] p = .023 (seeds 3) | -0.014 [-0.042, +0.015] p = .347 (seeds 3) | +0.052 [+0.023, +0.081] p < .001 (seeds 3) | +0.046 [+0.006, +0.083] p = .028 (seeds 3) |  |
| dinov2reg_L6 | +0.085 [+0.068, +0.103] p < .001 (seeds 3) | +0.054 [+0.039, +0.070] p < .001 (seeds 3) | +0.119 [+0.098, +0.141] p < .001 (seeds 3) | +0.102 [+0.078, +0.127] p < .001 (seeds 3) |  |

## Kanizsa real_filled - control180, inside AOI (size avg)

| features | R1n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| dinov2reg_L12 | +0.017 [-0.036, +0.070] p = .536 (seeds 3) | -0.019 [-0.055, +0.016] p = .298 (seeds 3) | +0.035 [-0.010, +0.080] p = .135 (seeds 3) | +0.023 [-0.022, +0.070] p = .329 (seeds 3) |  |
| dinov2reg_L6 | +0.152 [+0.128, +0.174] p < .001 (seeds 3) | +0.121 [+0.099, +0.146] p < .001 (seeds 3) | +0.185 [+0.173, +0.197] p < .001 (seeds 3) | +0.114 [+0.097, +0.132] p < .001 (seeds 3) |  |

## Ehrenstein illusory - control_end, inside AOI (size avg)

| features | R1n | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| dinov2reg_L12 | +0.013 [+0.004, +0.021] p = .002 (seeds 3) | +0.034 [+0.025, +0.044] p < .001 (seeds 3) | -0.031 [-0.045, -0.019] p < .001 (seeds 3) | -0.031 [-0.044, -0.019] p < .001 (seeds 3) |  |
| dinov2reg_L6 | +0.084 [+0.070, +0.099] p < .001 (seeds 3) | +0.055 [+0.045, +0.066] p < .001 (seeds 3) | +0.107 [+0.090, +0.127] p < .001 (seeds 3) | +0.091 [+0.073, +0.110] p < .001 (seeds 3) |  |

## Paired readout contrasts, Kanizsa illusory - control180 (size avg)

- dinov2reg_L12 R1n-R1: inside +0.038 [+0.022, +0.054]; core30 +0.032 [+0.012, +0.051]; Ehrenstein -0.022 [-0.030, -0.015]
- dinov2reg_L12 R1n-R2: inside -0.022 [-0.046, +0.003]; core30 -0.019 [-0.041, +0.005]; Ehrenstein +0.044 [+0.031, +0.057]
- dinov2reg_L12 R1n-R3: inside -0.000 [-0.021, +0.022]; core30 +0.001 [-0.018, +0.020]; Ehrenstein +0.044 [+0.028, +0.059]
- dinov2reg_L6 R1n-R1: inside +0.029 [+0.021, +0.036]; core30 +0.025 [+0.018, +0.032]; Ehrenstein +0.029 [+0.021, +0.038]
- dinov2reg_L6 R1n-R2: inside -0.039 [-0.059, -0.020]; core30 -0.040 [-0.054, -0.025]; Ehrenstein -0.023 [-0.029, -0.017]
- dinov2reg_L6 R1n-R3: inside -0.031 [-0.051, -0.011]; core30 -0.019 [-0.033, -0.004]; Ehrenstein -0.007 [-0.016, +0.002]

## Section B ratio(20), size avg (eligible readouts only)

- dinov2reg_L6 R1n inside: 0.53 [0.25, 0.72] intermediate
- dinov2reg_L6 R1n inside_eroded: 0.53 [0.25, 0.72] intermediate
- dinov2reg_L6 R1n r_core30: 0.50 [0.18, 0.69] intermediate
- dinov2reg_L6 R1 inside: 0.60 [0.21, 0.77] intermediate
- dinov2reg_L6 R1 inside_eroded: 0.59 [0.19, 0.77] intermediate
- dinov2reg_L6 R1 r_core30: 0.58 [0.14, 0.75] intermediate
- dinov2reg_L6 R2 inside: 0.61 [0.49, 0.71] intermediate
- dinov2reg_L6 R2 inside_eroded: 0.60 [0.47, 0.70] intermediate
- dinov2reg_L6 R2 r_core30: 0.53 [0.40, 0.64] intermediate
- dinov2reg_L6 R3 inside: 0.61 [0.42, 0.73] intermediate
- dinov2reg_L6 R3 inside_eroded: 0.60 [0.42, 0.72] intermediate
- dinov2reg_L6 R3 r_core30: 0.47 [0.21, 0.64] relatability-tuned

## Alternating rotation (stim_alt), theta 20, size avg

| features | readout | measure | ratio_uni20 | ratio_alt20 | diff20 | drop_ratio20 |
|---|---|---|---|---|---|---|
| dinov2reg_L12 | R1n | r_inside | 4.31 [-7.74, 9.34] (den. CI includes 0) | 4.31 [-7.38, 8.95] (den. CI includes 0) | -0.01 [-3.18, 3.01] (den. CI includes 0) | 1.00 [-2.78, 6.00] (den. CI includes 0) |
| dinov2reg_L12 | R1n | r_core30 | 7.84 [-5.45, 6.95] (den. CI includes 0) | 8.24 [-5.33, 6.68] (den. CI includes 0) | 0.40 [-2.86, 2.49] (den. CI includes 0) | 1.06 [-4.27, 6.79] (den. CI includes 0) |
| dinov2reg_L12 | R1n | f_core30 | 0.05 [-9.53, 10.96] (den. CI includes 0) | 0.14 [-6.95, 8.32] (den. CI includes 0) | 0.09 [-3.07, 2.97] (den. CI includes 0) | 0.90 [0.51, 2.06] |
| dinov2reg_L12 | R1 | r_inside | 0.15 [-1.88, 0.59] | 0.49 [-0.41, 0.74] | 0.34 [0.05, 1.48] | 0.60 [0.39, 0.88] |
| dinov2reg_L12 | R1 | r_core30 | 0.20 [-2.38, 0.68] | 0.53 [-0.68, 0.81] | 0.32 [-0.01, 1.76] | 0.59 [0.32, 0.98] |
| dinov2reg_L12 | R1 | f_core30 | 0.29 [-0.79, 0.63] | 0.53 [0.10, 0.69] | 0.24 [-0.01, 0.96] | 0.66 [0.42, 1.03] |
| dinov2reg_L12 | R2 | r_inside | 1.30 [-1.48, 4.41] (den. CI includes 0) | 1.05 [0.36, 1.88] (den. CI includes 0) | -0.25 [-2.62, 1.89] (den. CI includes 0) | 0.16 [-0.92, 1.18] (den. CI includes 0) |
| dinov2reg_L12 | R2 | r_core30 | 1.45 [-3.66, 6.57] (den. CI includes 0) | 1.01 [0.05, 1.91] (den. CI includes 0) | -0.44 [-5.14, 4.09] (den. CI includes 0) | 0.01 [-1.19, 0.57] |
| dinov2reg_L12 | R2 | f_core30 | -8.64 [-24.27, 28.58] (den. CI includes 0) | -5.62 [-15.80, 19.42] (den. CI includes 0) | 3.02 [-8.73, 7.95] (den. CI includes 0) | 0.69 [0.52, 1.02] |
| dinov2reg_L12 | R3 | r_inside | 2.32 [-5.23, 7.09] (den. CI includes 0) | 1.76 [-2.87, 4.51] (den. CI includes 0) | -0.56 [-5.36, 4.85] (den. CI includes 0) | 0.57 [-6.84, 8.27] (den. CI includes 0) |
| dinov2reg_L12 | R3 | r_core30 | 13.25 [-4.74, 7.28] (den. CI includes 0) | 3.70 [-2.65, 4.19] (den. CI includes 0) | -9.55 [-5.78, 5.08] (den. CI includes 0) | 0.22 [-6.70, 7.80] (den. CI includes 0) |
| dinov2reg_L12 | R3 | f_core30 | -0.51 [-11.94, 14.06] (den. CI includes 0) | -0.25 [-8.29, 9.92] (den. CI includes 0) | 0.26 [-4.77, 4.25] (den. CI includes 0) | 0.83 [0.41, 1.84] |
| dinov2reg_L6 | R1n | r_inside | 0.53 [0.25, 0.72] | 0.93 [0.86, 0.99] | 0.40 [0.23, 0.64] | 0.15 [0.02, 0.26] |
| dinov2reg_L6 | R1n | r_core30 | 0.50 [0.18, 0.69] | 0.95 [0.89, 1.02] | 0.45 [0.26, 0.76] | 0.10 [-0.04, 0.22] |
| dinov2reg_L6 | R1n | f_core30 | 0.63 [0.42, 0.77] | 0.95 [0.90, 1.00] | 0.32 [0.18, 0.52] | 0.14 [0.00, 0.29] |
| dinov2reg_L6 | R1 | r_inside | 0.60 [0.21, 0.77] | 1.13 [1.02, 1.35] | 0.53 [0.26, 1.13] | -0.32 [-0.54, -0.08] |
| dinov2reg_L6 | R1 | r_core30 | 0.58 [0.14, 0.75] | 1.19 [1.04, 1.60] | 0.61 [0.29, 1.45] | -0.46 [-0.78, -0.13] |
| dinov2reg_L6 | R1 | f_core30 | 0.66 [0.34, 0.79] | 1.15 [1.02, 1.47] | 0.49 [0.24, 1.12] | -0.44 [-0.81, -0.10] |
| dinov2reg_L6 | R2 | r_inside | 0.61 [0.49, 0.71] | 0.83 [0.72, 0.91] | 0.22 [0.13, 0.31] | 0.45 [0.27, 0.63] |
| dinov2reg_L6 | R2 | r_core30 | 0.53 [0.40, 0.64] | 0.77 [0.67, 0.86] | 0.24 [0.14, 0.35] | 0.50 [0.33, 0.66] |
| dinov2reg_L6 | R2 | f_core30 | 0.62 [0.54, 0.69] | 0.85 [0.78, 0.92] | 0.23 [0.14, 0.33] | 0.40 [0.22, 0.58] |
| dinov2reg_L6 | R3 | r_inside | 0.61 [0.42, 0.73] | 0.89 [0.79, 0.98] | 0.29 [0.18, 0.45] | 0.27 [0.07, 0.47] |
| dinov2reg_L6 | R3 | r_core30 | 0.47 [0.21, 0.64] | 0.84 [0.72, 0.94] | 0.36 [0.22, 0.60] | 0.31 [0.11, 0.48] |
| dinov2reg_L6 | R3 | f_core30 | 0.66 [0.53, 0.76] | 0.99 [0.90, 1.11] | 0.33 [0.19, 0.53] | 0.03 [-0.31, 0.31] |
