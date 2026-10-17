# stim_alt: stimulus measures (stimuli/verify_stim_alt.py)

Hard checks: all passed. H1 renderer identity on 360 stim_v3 images; H2 360 alt images re-rendered identically; H3 max differing-pixel counts vs illusory: {'hi_outside_disc': 0, 'lo_outside_band': 0, 'lo_in_core30': 0, 'lo_in_core45': 0, 'lo_in_inside_eroded_all': 0, 'lo_in_inside': 266}; H4 max per-inducer ink-area difference (mis, alt) 0.0027; H5 max notch-direction error 0.521 deg.

## square

| condition | notch-sector pixels inked | by inducer | share of inducer ink in notch sectors | exact mirrors | S analytic vertical | S analytic horizontal | S analytic diag_TL_BR | S analytic diag_TR_BL | S image vertical | S image horizontal | S image diag_TL_BR | S image diag_TR_BL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| illusory | 0.037 | 0.016, 0.036, 0.058, 0.037 | 0.013 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | 0.997 | 1.000 | 0.994 |
| control180 | 0.992 | 0.980, 0.994, 0.999, 0.994 | 0.342 | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | 0.997 | 1.000 | 0.994 |
| control90 | 0.980 | 0.978, 0.992, 0.978, 0.972 | 0.338 | 0 | 0.660 | 0.660 | 0.660 | 0.660 | 0.668 | 0.668 | 0.670 | 0.667 |
| mis10 | 0.125 | 0.113, 0.139, 0.134, 0.114 | 0.043 | 0 | 0.926 | 0.926 | 0.926 | 0.926 | 0.924 | 0.924 | 0.926 | 0.922 |
| mis20 | 0.232 | 0.222, 0.246, 0.239, 0.219 | 0.080 | 0 | 0.852 | 0.852 | 0.852 | 0.852 | 0.850 | 0.850 | 0.852 | 0.848 |
| mis30 | 0.338 | 0.331, 0.354, 0.343, 0.325 | 0.117 | 0 | 0.778 | 0.778 | 0.778 | 0.778 | 0.776 | 0.776 | 0.778 | 0.775 |
| alt10a | 0.124 | 0.113, 0.112, 0.134, 0.135 | 0.043 | 2 | 1.000 | 1.000 | 0.926 | 0.926 | 0.996 | 0.997 | 0.927 | 0.923 |
| alt10b | 0.126 | 0.114, 0.139, 0.139, 0.114 | 0.044 | 2 | 1.000 | 1.000 | 0.926 | 0.926 | 0.997 | 0.996 | 0.925 | 0.921 |
| alt20a | 0.231 | 0.222, 0.218, 0.239, 0.244 | 0.080 | 2 | 1.000 | 1.000 | 0.852 | 0.852 | 0.996 | 0.997 | 0.852 | 0.849 |
| alt20b | 0.233 | 0.222, 0.246, 0.243, 0.219 | 0.080 | 2 | 1.000 | 1.000 | 0.852 | 0.852 | 0.997 | 0.996 | 0.851 | 0.847 |
| alt30a | 0.338 | 0.331, 0.324, 0.343, 0.352 | 0.117 | 2 | 1.000 | 1.000 | 0.778 | 0.778 | 0.996 | 0.997 | 0.778 | 0.775 |
| alt30b | 0.339 | 0.330, 0.354, 0.347, 0.325 | 0.117 | 2 | 1.000 | 1.000 | 0.778 | 0.778 | 0.997 | 0.996 | 0.777 | 0.774 |

Edge geometry per side at sr 0.5 (turn angle deg / far tip offset from the near edge's line px / kind):

| condition | 0-1 | 1-2 | 2-3 | 3-0 |
|---|---|---|---|---|
| illusory | 0 / 0.0 / collinear | 0 / 0.0 / collinear | 0 / 0.0 / collinear | 0 / 0.0 / collinear |
| control180 | 0 / 0.0 / collinear, edges point away (not relatable) | 0 / 0.0 / collinear, edges point away (not relatable) | 0 / 0.0 / collinear, edges point away (not relatable) | 0 / 0.0 / collinear, edges point away (not relatable) |
| control90 | 0 / 190.0 / parallel offset (not relatable) | 0 / 190.0 / parallel offset (not relatable) | 0 / 190.0 / parallel offset (not relatable) | 0 / 190.0 / parallel offset (not relatable) |
| mis10 | 0 / 33.0 / parallel offset (not relatable) | 0 / 33.0 / parallel offset (not relatable) | 0 / 33.0 / parallel offset (not relatable) | 0 / 33.0 / parallel offset (not relatable) |
| mis20 | 0 / 65.0 / parallel offset (not relatable) | 0 / 65.0 / parallel offset (not relatable) | 0 / 65.0 / parallel offset (not relatable) | 0 / 65.0 / parallel offset (not relatable) |
| mis30 | 0 / 95.0 / parallel offset (not relatable) | 0 / 95.0 / parallel offset (not relatable) | 0 / 95.0 / parallel offset (not relatable) | 0 / 95.0 / parallel offset (not relatable) |
| alt10a | 20 / 16.8 / relatable bend | 20 / 16.8 / relatable bend | 20 / 16.8 / relatable bend | 20 / 16.8 / relatable bend |
| alt10b | 20 / 16.8 / relatable bend | 20 / 16.8 / relatable bend | 20 / 16.8 / relatable bend | 20 / 16.8 / relatable bend |
| alt20a | 40 / 34.5 / relatable bend | 40 / 34.5 / relatable bend | 40 / 34.5 / relatable bend | 40 / 34.5 / relatable bend |
| alt20b | 40 / 34.5 / relatable bend | 40 / 34.5 / relatable bend | 40 / 34.5 / relatable bend | 40 / 34.5 / relatable bend |
| alt30a | 60 / 53.9 / relatable bend | 60 / 53.9 / relatable bend | 60 / 53.9 / relatable bend | 60 / 53.9 / relatable bend |
| alt30b | 60 / 53.9 / relatable bend | 60 / 53.9 / relatable bend | 60 / 53.9 / relatable bend | 60 / 53.9 / relatable bend |

## triangle

| condition | notch-sector pixels inked | by inducer | share of inducer ink in notch sectors | exact mirrors | S analytic apex(vertical) | S analytic BR | S analytic BL | S image apex(vertical) | S image BR | S image BL |
|---|---|---|---|---|---|---|---|---|---|---|
| illusory | 0.025 | 0.010, 0.038, 0.028 | 0.005 | 3 | 1.000 | 1.000 | 1.000 | 0.997 | 0.994 | 0.995 |
| control180 | 0.992 | 0.983, 1.000, 0.994 | 0.200 | 3 | 1.000 | 1.000 | 1.000 | 0.997 | 0.994 | 0.995 |
| control90 | 0.992 | 0.983, 1.000, 0.994 | 0.200 | 0 | 0.798 | 0.800 | 0.800 | 0.800 | 0.798 | 0.799 |
| mis10 | 0.169 | 0.176, 0.188, 0.145 | 0.034 | 0 | 0.933 | 0.933 | 0.933 | 0.931 | 0.930 | 0.930 |
| mis20 | 0.334 | 0.343, 0.350, 0.309 | 0.067 | 0 | 0.867 | 0.867 | 0.867 | 0.865 | 0.864 | 0.864 |
| mis30 | 0.498 | 0.508, 0.513, 0.474 | 0.101 | 0 | 0.800 | 0.800 | 0.800 | 0.801 | 0.800 | 0.800 |
| alt10a | 0.130 | 0.010, 0.188, 0.191 | 0.026 | 1 | 1.000 | 0.955 | 0.955 | 0.997 | 0.952 | 0.952 |
| alt10b | 0.109 | 0.010, 0.171, 0.145 | 0.022 | 1 | 1.000 | 0.955 | 0.955 | 0.997 | 0.951 | 0.952 |
| alt20a | 0.240 | 0.010, 0.350, 0.358 | 0.048 | 1 | 1.000 | 0.911 | 0.911 | 0.997 | 0.908 | 0.908 |
| alt20b | 0.217 | 0.010, 0.331, 0.309 | 0.044 | 1 | 1.000 | 0.911 | 0.911 | 0.997 | 0.907 | 0.908 |
| alt30a | 0.349 | 0.010, 0.513, 0.525 | 0.071 | 1 | 1.000 | 0.866 | 0.866 | 0.996 | 0.865 | 0.865 |
| alt30b | 0.325 | 0.010, 0.492, 0.474 | 0.066 | 1 | 1.000 | 0.866 | 0.866 | 0.997 | 0.865 | 0.865 |

Edge geometry per side at sr 0.5 (turn angle deg / far tip offset from the near edge's line px / kind):

| condition | 0-1 | 1-2 | 2-0 |
|---|---|---|---|
| illusory | 0 / 0.0 / collinear | 0 / 0.0 / collinear | 0 / 0.0 / collinear |
| control180 | 0 / 0.0 / collinear, edges point away (not relatable) | 0 / 0.0 / collinear, edges point away (not relatable) | 0 / 0.0 / collinear, edges point away (not relatable) |
| control90 | 0 / 230.0 / parallel offset (not relatable) | 0 / 230.0 / parallel offset (not relatable) | 0 / 230.0 / parallel offset (not relatable) |
| mis10 | 0 / 39.9 / parallel offset (not relatable) | 0 / 39.9 / parallel offset (not relatable) | 0 / 39.9 / parallel offset (not relatable) |
| mis20 | 0 / 78.7 / parallel offset (not relatable) | 0 / 78.7 / parallel offset (not relatable) | 0 / 78.7 / parallel offset (not relatable) |
| mis30 | 0 / 115.0 / parallel offset (not relatable) | 0 / 115.0 / parallel offset (not relatable) | 0 / 115.0 / parallel offset (not relatable) |
| alt10a | 10 / 10.0 / not relatable | 20 / 20.3 / relatable bend | 10 / 29.9 / not relatable |
| alt10b | 10 / 10.0 / not relatable | 20 / 20.3 / relatable bend | 10 / 29.9 / not relatable |
| alt20a | 20 / 19.7 / not relatable | 40 / 41.7 / relatable bend | 20 / 59.0 / not relatable |
| alt20b | 20 / 19.7 / not relatable | 40 / 41.7 / relatable bend | 20 / 59.0 / not relatable |
| alt30a | 30 / 28.8 / not relatable | 60 / 65.2 / relatable bend | 30 / 86.2 / not relatable |
| alt30b | 30 / 28.8 / not relatable | 60 / 65.2 / relatable bend | 30 / 86.2 / not relatable |

