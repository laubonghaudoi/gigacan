# Transcription Optimization Log

Last updated: 2026-02-17

## Purpose

This document records the optimization work done on `uv run transcribe`, including:

- what was implemented,
- what was benchmarked,
- what failed,
- and what settings are currently safest for long runs.

Use this as the source of truth when revisiting performance tuning later.

## Environment Used For Benchmarking

- CPU: AMD Ryzen 9 9950X3D (16 cores)
- RAM: 58 GiB
- GPU: NVIDIA GeForce RTX 5090 (32 GiB VRAM)
- Main benchmark dataset: `download/2013`
  - 14 files
  - total audio: 55,454.808 seconds (~15.40 hours)

## Implemented Optimizations

1. Batch transcription over entire `download/` tree.
2. `--year` filter for per-year runs.
3. Resumable behavior (skip existing `.srt` unless `--overwrite`).
4. Atomic SRT writing (temp file + replace) to avoid broken output on interruption.
5. Backend support for both `vllm` and `transformers`.
6. Persistent worker mode (socket-based runtime reuse).
7. Cross-file super-batching (global segment queue).
8. Frame-aware batch selection to reduce padding waste.
9. ASR payload prefetch (`--asr-prefetch-batches`) for overlap.
10. CPU parallelism knobs (`--prep-workers`, `--vad-workers`).
11. VAD cache (`.cache/qwen_srt_vad`) and segment merge controls.
12. Binary-split fallback in batch error paths.
13. Memory pressure controls:
   - decode backlog capping,
   - decoded audio RAM budget (`--super-batch-max-decoded-gib`),
   - long/short interleaved job ordering.
14. `legco.csv` `transcribed` column + scan integration in `2_scan_progress.py`.

## Benchmark Attempts (2013)

| Run | Config (key knobs) | Elapsed (s) | Realtime Factor (x) | Result |
|---|---|---:|---:|---|
| vLLM baseline | `prep/vad=4/4`, `active/preload=8/10`, `max_decoded=6GiB` | 112.87 | 491.32 | Best so far |
| vLLM push #1 | `prep/vad=6/6`, `active/preload=12/18`, `prefetch=3`, `queue_mult=6`, `max_decoded=14GiB` | 126.70 | 437.69 | Slower, higher RAM |
| vLLM push #2 | `prep/vad=4/4`, `active/preload=10/14`, `queue_mult=6`, `max_decoded=12GiB` | 120.28 | 461.05 | Still slower than baseline |
| transformers compare | same shape as baseline (`prep/vad=4/4`, `active/preload=8/10`, `max_decoded=6GiB`) | 342.48 | 161.92 | Much slower |

Speed comparison from the measured runs:

- vLLM vs transformers: `342.48 / 112.87 = 3.03x` faster for vLLM.

## Key Findings

1. Higher instantaneous GPU usage does not guarantee better throughput.
2. Increasing worker/preload/queue aggressively increased RAM use but reduced end-to-end speed.
3. Current best throughput came from a moderate, stable profile (vLLM baseline above).
4. Pipeline is naturally bursty; short GPU dips can be normal if file completions continue.

## Pitfalls Encountered

### 1) RAM exhaustion crashes

Observed during earlier aggressive tuning attempts (high worker counts + large active/preload windows), causing:

- RAM and swap to fill,
- system instability/crashes.

Mitigation used:

- conservative defaults,
- explicit caps on active/preload settings,
- decoded audio budget control.

### 2) Long-run stall with low utilization

During full-folder run (`transcribe_all_20260217_193901`, with `--super-batch-max-decoded-gib 6`), symptoms were:

- GPU near 0% for extended periods,
- process alive but sleeping (`futex_do_wait`),
- no log growth,
- no `.srt` count increase.

Workaround used:

- restart run with larger decoded budget (`--super-batch-max-decoded-gib 48`).
- New session (`transcribe_all_restart_20260217_194739`) resumed normal progress.

Note:

- This points to a possible deadlock/starvation edge case in decoded-budget backpressure logic under long full-run conditions.

### 3) Orphan runtime processes after interruption

Killing tmux did not always stop all child processes (worker/engine lingering).

Mitigation used:

- explicitly verify with `pgrep -af ...`,
- stop worker/engine before re-launch.

### 4) Driver/kernel operational issue

At one point GPU driver availability disappeared after kernel/driver changes.

Mitigation used:

- reboot + driver/kernel alignment before resuming transcription.

## Current Recommended Full-Run Command

Prefer this as the default stable launch profile:

```bash
uv run transcribe \
  --audio-dir download \
  --output-dir transcriptions \
  --backend vllm \
  --prep-workers 4 \
  --vad-workers 4 \
  --super-batch-active-files 8 \
  --super-batch-preload-files 10 \
  --super-batch-max-decoded-gib 48
```

Why `48` now:

- `6` GiB was fast on small benchmark but showed a full-run stall once.
- `48` GiB has been stable in the current resumed full run.

## Health Check Playbook

When checking if run is healthy, use all three signals together:

1. GPU: `nvidia-smi` should show periodic spikes (not flat zero for long periods).
2. Progress: `find transcriptions -name '*.srt' | wc -l` should increase over time.
3. Log/pane: tmux/log should show advancing `Transcribing: ...` file counters.

Useful commands:

```bash
tmux ls
tmux attach -t <session>
tail -f logs/<session>.log
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
find transcriptions -type f -name '*.srt' | wc -l
```

## Open Follow-ups

1. Investigate and fix the decoded-budget stall edge case so low budget values remain safe on full runs.
2. Add built-in watchdog logging for queue depth / worker state / budget counters.
3. Add automated benchmark script to run reproducible A/B sweeps and output summary table.
