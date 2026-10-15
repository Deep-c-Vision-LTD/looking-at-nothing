# Looking at Nothing, version 2: analysis plan and decision rules

Written 2026-10-15, after the Perception desk rejection (PEC-26-0231) and before any of the analyses below were run.
Question: what computation in a saliency model turns an arrangement of inducers into attentional priority for an
illusory figure? Existing results (six models, 18 layer readouts) are kept; the analyses below are new.

## Stimuli (stim_v3)
Kanizsa: square and triangle x support ratio 0.3, 0.5, 0.7 x dark and light x 5 positions = 60 configurations, each in
10 conditions: illusory (aligned), control180 (primary control), control90, real outline, real filled (grey surface, as
in Kavsek 2024), realC (the real outline drawn over the control180 inducers), and misaligned inducers rotated in the
same direction by 10, 20 and 30 deg (notches still facing inward). Ehrenstein: 2 line counts x 2 radii x 2 polarities
x 5 positions = 40 configurations in illusory, control_end (primary control), real outline, real filled and realC.
Measure: r = map mass inside the figure / (inside + inducer regions), as before; also with the inside region eroded by
8 px from the contour. Effect = r(test) - r(primary control).

## Statistics
Cluster bootstrap (10,000 resamples) over the shape x support ratio x polarity cells (positions resampled within cells);
95% confidence intervals; p-values reported no smaller than "p < .001". Primary layers per backbone, fixed now: first
sweep layer, best-fixation layer, last layer (DINOv2-B L2, L8, L12; CLIP-L L4, L12, L24; SigLIP-L L4, L8, L24).

## A. Readout architecture (decisive test)
Features: DINOv2-B L12, DINOv2-B L6, SigLIP-L L24, and one DeepGaze IIE CNN feature set. Readouts on identical features:
R1 pointwise (1x1 layers, then Gaussian blur, as in DeepGaze), R2 ASPP without the global-pooling branch, R3 ASPP with
dilation 1 only and no global pooling, R4 full ASPP with global pooling (the sweep readout). Three seeds each.
Validity: a readout is interpreted only if its held-out MIT1003 CC is at least that of R4 on the same features minus 0.03.
Gate for the CPU pilot (3,000 cached training images): R4 on DINOv2 L12 must reproduce the reversal (Kanizsa effect CI
below 0); if it does not, the pilot is reported as non-diagnostic and only the full-data runs count.
Decision on DINOv2 L12: "readout decides" if R1's effect CI is above 0 and R4's below 0, with all seeds of each sharing
the sign; "global pooling is the critical component" if additionally R2 and R3 are not below 0; "features decide" if
R1 to R4 share the sign of the effect on every feature set; otherwise "mixed", reported as such.

## B. Configural completion or inward-notch cue
For every model or readout whose aligned Kanizsa effect CI is above 0: ratio(theta) = effect(misaligned theta) /
effect(aligned). "Relatability-tuned" if ratio(20 deg) <= 0.5 with CI upper bound below 0.75; "inward-notch cue" if
the CI lower bound of ratio(20 deg) is at least 0.5; otherwise "intermediate".

## C. Encoding (readout-free)
Index E = <f(test) - f(control180), f(realC) - f(control180)> / ||f(realC) - f(control180)||^2 at tokens whose input
patch is pixel-identical across test, control180, control90 and realC, inside the figure; also cosine. Nulls: the same
index with control90 as the test image, and the index at background tokens more than 80 px from the figure. All layers
of DINOv2-B, CLIP-L and SigLIP-L at 336 and 448 px; DeepGaze IIE CNN units whose theoretical receptive field excludes
every inducer. "Figure-specific encoding" at a layer if the interior CI lower bound exceeds the upper bounds of both
nulls. "Encoded but not prioritised" is claimed for a readout only if its layer shows figure-specific encoding, the
readout's effect CI is at or below 0, and the encoding also holds for misaligned 20 deg at under half its aligned value
(otherwise it is called an inward-notch code, not a figure code).

## D. What carries the effect in DeepGaze IIE and DeepGaze MSDB
Shapley decomposition of the Kanizsa effect over components (IIE: its backbones; MSDB: its feature groups and scales),
replacing a component's features for the illusory image with those for the control180 image, averaged over all
coalitions. A component "carries" the effect if its Shapley share is at least 0.5.

## E. Human comparison
Illusion index = effect / (r(real filled) - r(control180)), computed only where the denominator CI is above 0, and
compared with Kavsek's pooled indices (0.86 Kanizsa, 0.82 Ehrenstein), stated as dominated by children (97 of 127).

## Title rule
A "readout decides" or "global pooling" outcome leads the title with the readout; "features decide" leads with feature
depth and pretraining; an encoding claim enters the title only if C shows figure-specific, relatability-tuned encoding
at middle layers of all three networks.
