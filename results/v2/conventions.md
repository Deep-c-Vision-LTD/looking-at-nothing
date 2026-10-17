# Conventions and human reference

Read by `paper_numbers/make_numbers.py` (the configuration counts and the size of the human sample are taken from the
two sentences below).

Conventions used throughout. r = map mass inside the figure / (inside + inducer AOI). Kanizsa effect =
r(illusory) - r(control180); Ehrenstein effect = r(illusory) - r(control_end). Mean over configurations (60 Kanizsa, 40
Ehrenstein) with a 95% cluster-bootstrap CI (10,000 resamples over shape x support ratio x polarity cells, positions
resampled within cells). Layer readouts: per-configuration effect averaged over 336 and 448 px. Primary AOI = inside;
the 8 px eroded AOIs are sensitivity checks.

Human reference. Illusion index = effect / (r(real_filled) - r(control)), only where the denominator CI is above 0;
compared with the pooled values computed from Kavsek (2024), 0.857 (Kanizsa) and 0.824 (Ehrenstein), a sample
dominated by children (97 of 127).
