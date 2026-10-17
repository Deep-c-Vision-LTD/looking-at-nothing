# Readout swap: design

Four frozen feature sets, five readouts, three seeds (60 readouts), trained with identical data and recipe.

Feature sets (`pilot/common.py`, `FSETS`):

| key | features | channels |
|---|---|---:|
| `dinov2_L12` | DINOv2-B (facebook/dinov2-base) `hidden_states[12]`, before the final layer norm | 768 |
| `dinov2_L6` | DINOv2-B `hidden_states[6]` (same forward pass as block 12) | 768 |
| `siglip_L24` | SigLIP-L/16 (google/siglip-large-patch16-256) `hidden_states[24]`, before the post layer norm | 1,024 |
| `iie_shapenetc` | ShapeNet-C ResNet-50 of DeepGaze IIE at its six readout taps, weights from the IIE checkpoint | 2,048 |

Readouts (`fullcommon.make_head`): R1n (pointwise, fixed per-channel normalisation), R1 (pointwise, DeepGaze image-wide
layer normalisation), R2 (atrous spatial pyramid pooling without the global-pooling branch), R3 (one dilation-1 3 x 3
branch), R4 (full atrous spatial pyramid pooling with global pooling). Seeds 0, 1 and 2.

Recipe: the 12,803 training images of `models/ab_objective.py` (SALICON 10,000, CAT2000 2,000, MIT1003 803), 336 px,
6 epochs, batch 32, AdamW (learning rate 5e-4, weight decay 1e-4, cosine schedule), loss KL + 2(1 - CC), no augmentation,
final epoch. Validity: held-out MIT1003 CC (200 images, centre bias on) at least that of R4 on the same features minus
0.03. Illusion displays: 336 and 448 px, centre bias off.

Run order: `stats_full.py` (R1n input statistics), `train_full.py`, `eval_full.py`, `analyze_full.py`; `run_full.sh` runs
the whole queue. `pilot/` holds the shared definitions and the earlier 3,000-image pilot.
