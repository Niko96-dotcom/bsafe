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

### Still pass scales (`--still-scales`)

The menu bar's Strict preset runs the still pass as `--still-scales 1.0,1.5` (period 1 s, so each scale every 2 s). Simulated with `~/bsafe-bench/heavy.py` / `heavy_alt.py` on the Strict base (`sim-cq47`, ps3, 640m 73 ms at 1x, 170 ms at 1.5x), mean leak/over-censor over five sessions vs a 640m@1x + 640m@1.5x union GT: no still pass 5.0/4.1, 1x every 2 s 3.9/4.1, 1.5x every 2 s 3.7/4.2, 1x and 1.5x together every 2 s 3.4/4.2, alternating every 1 s **3.2/4.2** (same ~120 ms per still second as together, no 240 ms stall). 1.5x alone loses tits (1.5 -> 3.9%) because large objects are found at 1x; feet 5.7 -> 4.6%, armpits 6.6 -> 3.5%. Live captures 1728 px (points), so 1.5x is an in-Python `cv2.INTER_LINEAR` upscale; 640m on upscaled live frames scored the same as on true 1.5x recordings (3.2/4.2, both GTs). Standard keeps the single native pass (alternation not simulated there).

## Motion estimation region

Each track also keeps `core` (the latest matched detection, unblended); motion is
estimated on `estimationRegion(core)`, the central 55% (matching the suggested
0.4 per-side padding; the CLI default is 0.0), both per frame and when catching a detection up from its
`seq`, so a static margin cannot out-vote an object moving over a static
background. A core that crosses a frame edge falls back to the full core (the
central crop breaks edge coasting). Known gaps: near frame edges and with
`--full-censor` (3x boxes) the full or cropped region can still be mostly static.

## Fail-closed and static-screen fixes (2026-09-29)

Correctness fixes from the architecture review; none of them changes detection
on the bench recordings.

- **Static screens are re-detected.** ScreenCaptureKit delivers no complete
  frames while the screen is unchanged, and a credit was only answered with a
  new frame, so a fully static screen was detected once and never got the
  640m still pass. `DisplayPipeline.idleRefeed` re-feeds the latest frame as a
  new frame (new seq, current host time) once a credit has waited 0.5 s. The
  bench recordings deliver frames continuously, so this gap never showed there.
- **Still pass off the light loop.** `--still-model` now runs on its own thread
  (`StillWorker` in `bsafe/live.py`). Its boxes are re-sent for the display's
  last replied seq (merged with that frame's light detections), but only while
  that frame is the one the still pass ran on or the screen is still still.
  Before, the 73–170 ms pass delayed the light reply for its frame.
- **Detection failures fail closed.** A failed light pass no longer replies with
  an empty box list (which counted as a miss and let tracks expire). Swift
  pixelates the whole display when frames keep changing but no detection has
  been applied for 2 s (`DisplayPipeline.staleCoverSeconds`), armed after the
  first detection.
- **No box size cap.** The overlay clamped boxes to 80% of the display width and
  height (inherited upstream code), leaving the edge of large close-ups uncovered.
- Checked, not a bug: `NSWindow.sharingType = .none` still keeps the overlay out
  of ScreenCaptureKit captures on macOS 26 (a `.readOnly` control window shows up).

## Keep-only detections (`--keep-conf`, default 0.1)

Detections from `--keep-conf` up to 0.25 (`fastdetect.STRONG_CONFIDENCE`) are
sent as keep-only boxes (optional trailing list in `CMD_CENSOR_SEQ`). The tracker
matches them only against tracks left unmatched by the strong detections
(ByteTrack's second stage): a match refreshes the track like a normal match, and
an unmatched keep box never starts a track. NMS runs in score order, so the
strong set is unchanged by the lower floor. The old `_DETECT_THRESHOLD` 0.2 was
dead: NMS's 0.25 score threshold dropped everything below it.

Measured on the five simulated X sessions (`~/bsafe-bench/keep.sh`): Strict
light passes (320n at 1x + 0.5,0.25) recorded down to 0.1 (`bench detect
--score-floor 0.1`), detect time normalized to the live-calibrated 22.7 ms
(`normtime.py`; the machine was loaded during detection), padding 0.4, min
padding 24, 3 s persistence, GT = 640m@1x + 640m@1.5x union, no still pass.
Mean leak / over-censor / objects missed:

| keep | leak | over | missed |
|---|---|---|---|
| off | 5.43 | 4.20 | 10.7 |
| 0.2 | 4.94 | 4.37 | 9.7 |
| 0.15 | 4.51 | 4.52 | 9.1 |
| 0.1 | 3.86 | 4.86 | 8.1 |

Every session improved (leak feet 5.6 -> 2.9, armpits 9.2 -> 8.2, ass 2.6 -> 1.8,
ass2 3.3 -> 2.3, tits 6.5 -> 4.1). For comparison, raising persistence from 1 s to
3 s cut leak 4.8 -> 3.9 for +1.2 points of over-censor. The still pass is not
included in this run; its boxes are strong-only.
