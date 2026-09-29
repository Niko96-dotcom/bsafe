# Benchmark

Files: `bsafe/bench.py`, `swift/Sources/BsafeReplay`, `BsafeCore/ReplayScheduler.swift`.
`bsafe bench detect` writes per-frame detections (`dets.jsonl`); `replay` turns
them into censor boxes (same class filter + padding + merge as live) and hands
them to the `bsafe-replay` binary, which replays them through the live tracker
on the live frame-credit schedule, then scores against ground truth. Ground
truth = per-frame oracle detections, linked into objects and gap-filled.
Build with `cd swift && swift build -c release` (produces `bsafe-replay`).
Bench recordings must be scaled to the live capture size (even height) or
detection resolution/latency will not match live.
`bench replay --min-padding N` floors box padding and is recorded in
`score.json` (run dir gets a `-mpxN` suffix when nonzero); `bench sweep
--min-padding N` applies it to every sweep config. Bench writes only metadata
(never pixels). Scoring: GT matches the on-screen line (latest `display_pts`
<= pts, including `event: apply` lines rendered at completion time via
`ReplayScheduler.beginFrameTimed`); over-censor averages over every displayed
frame vs the dilated GT of the nearest dets frame (per-class null);
never-covered objects add span + 1 frame to `first_cover_ms` (+ p90); detect
pts use `CAP_PROP_POS_MSEC` with index/fps fallback, header fps
`(n-1)/last_pts`. `bench replay/sweep --gt-dets PATH` (repeatable, `--gt-conf`)
scores recall against the union of stronger GT dets (scaled to live size,
same-frame IoU>=0.5 merged, frame-count tolerance <=2); new metrics
`objects_missed_pct` / `objects_missed_by_live_pct` (live IoU>=0.1).
`replay`/`sweep` auto-compute `DIR/motion.jsonl` (per-frame phaseCorrelate shift magnitudes, metadata only; reused only while its header matches the video's resolved path, size, mtime_ns, frame count and downscale, else recomputed) and report `by_phase` still/moving leak splits (still: shift < 0.5 px); `--covered` adds `*_COVERED` classes and a `-cov` run-dir suffix.
