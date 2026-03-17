# Bsafe — Project Roadmap

## Phase 1 — Proof of Concept

- Build Swift helper for screen capture, send frames to Python over IPC.
- Run NudeNet inference on received frames in Python.
- Log detections to console (no overlay yet).
- Validate end-to-end data flow on a single monitor.

## Phase 2 — MVP

- Swift helper renders always-on-top blackout overlays.
- End-to-end pipeline: capture → detect → censor.
- `bsafe start` works for a single monitor in foreground mode.
- Basic class filtering and box padding.

## Phase 3 — Refinement

- Face-aware padding exclusion (avoid censoring heads near nudity).
- Temporal smoothing and box tracking to reduce flicker.
- Box merging for overlapping detections.
- Multi-monitor and Retina scaling support.

## Phase 4 — Hardening

- Performance optimization (throttling, downscaling, latency targets).
- `bsafe doctor` with real checks (permissions, model presence, Swift helper).
- Error handling and permissions guidance.
- Packaging and distribution.
