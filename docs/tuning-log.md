# Live tuning log

Decisions behind the live detection and tracking defaults, with the bench
numbers that justified them. Newest details are also in the commit messages
(`git log -- swift/Sources/BsafeCore bsafe/live.py bsafe/fastdetect.py`).
Setup and scoring: [bench.md](bench.md).

## Tracker defaults (`BsafeCore/DisplayTracker.swift`)

Matched detections blend per edge: an edge moved outward snaps, an edge moved
inward shrinks by `TrackerConfig.shrinkAlpha` (0.1) per 0.1 s of
detection-frame time (`1-(1-a)^(dt/0.1)`; per pass when the same frame is
re-applied). A track is removed only after `persistPasses` missed passes
**and** `TrackerConfig.minPersistSeconds` (3.0) since its last match, so
persistence does not shrink as detection gets faster.

Both defaults (`shrinkAlpha`, `minPersistSeconds`) (was 0.25 / 0.6) were chosen with `bsafe bench sweep` on scrolling-feed recordings: lower leak across all classes for a small over-censor cost. `minPersistSeconds` was then raised 1.0 -> 3.0 on five simulated 3-min X sessions (320n + 0.5, padding 0.4, min-padding 24, 640m GT): mean leak 4.8% -> 3.9%, over-censor 2.4% -> 3.6%. Most live leak on still video is the 320n detector losing an object the oracle still sees, so longer persistence helps; a still-only persistence rule was rejected (about 2/3 of the extra over-censor is in still frames anyway, and moving-frame leak also improves). Extra scale 0.25 gave a similar gain to 2 s -> 3 s persistence at +26% detect time, so it stays off by default.

## Menu bar Strict preset

The menu bar Strict preset is `--extra-scales 0.5,0.25` (detect-scale 1.0): vs a 640m@1x+1.5x union GT with live-calibrated timing, mean leak/over/missed 5.0/4.1/10.5% at ~23 ms, vs 5.2/3.8/12.7% for `--detect-scale 1.5 --extra-scales 1.0,0.5` at ~47 ms (1.5x wins feet/tits, 0.25 wins armpits/ass). 1.5 + 0.5 without the 1.0 pass leaked more than Standard (armpit close-ups need the native-size pass).

## Still-screen 640m pass (`--still-model`)

`--still-model 640m` (off by default; the menu bar enables it for Standard/Strict when the model file exists) runs a second `FullFrameNudeDetector` synchronously on the same frame once `StillnessTracker` (`bsafe/live.py`: gray, 1/4 INTER_AREA, `cv2.phaseCorrelate`, still < 0.5 full-res px, same as bench `motion.jsonl`) has seen the display still for `--still-after` (0.5 s), at most every `--still-period` (2.0 s); detections are appended before `build_censor_boxes`. Chosen with `~/bsafe-bench/heavy.py` (injects 640m dets + 80 ms into replay dets): leak 5.0 -> 3.9% Strict, 5.6 -> 4.4% Standard, over-censor unchanged; periods 0.5-3 s were equal, 4 s worse.

## Motion estimation region

Each track also keeps `core` (the latest matched detection, unblended); motion is
estimated on `estimationRegion(core)`, the central 55% (matching the suggested
0.4 per-side padding; the CLI default is 0.0), both per frame and when catching a detection up from its
`seq`, so a static margin cannot out-vote an object moving over a static
background. A core that crosses a frame edge falls back to the full core (the
central crop breaks edge coasting). Known gaps: near frame edges and with
`--full-censor` (3x boxes) the full or cropped region can still be mostly static.
