# Qwen3-ASR Transcription Optimization Log

This file records optimization strategies we tried, how we measured them, and the resulting speed impact.

## Dataset and metric

- Benchmark dataset: `download/2013`
- Audio duration: `55,454.81s` (`~15.4041h`)
- Primary metric: end-to-end wall time (seconds)
- Secondary metric: realtime factor (`audio_seconds / wall_seconds`)

## Optimizations implemented in pipeline

These are already in code and affect both backends unless noted:

1. Cross-file super-batching (`global segment queue`)
2. Frame-aware batch selection (reduce padding waste)
3. ASR payload prefetch (`--asr-prefetch-batches`)
4. Parallel decode prep + VAD workers (`--prep-workers`, `--vad-workers`)
5. VAD segment cache (`--vad-cache-dir`, enabled by default)
6. Segment merge controls (`merge-target/max/gap`)
7. Decoded-audio RAM budget backpressure (`--super-batch-max-decoded-gib`)
8. Duration-aware file interleaving (`sort_jobs_by_duration`)
9. Binary-split fallback on ASR batch errors
10. Optional persistent worker (warm-run/steady-state benchmark mode)

Backend-specific:

1. `vllm`: `Qwen3ASRModel.LLM(...)`, `--vllm-gpu-memory-utilization`, `--vllm-tensor-parallel-size`
2. `transformers`: `--qwen-dtype auto` (`bfloat16` on CUDA)

## Experiment A: Warm-run throughput (persistent worker)

Method:

1. Warmup pass (not timed)
2. Timed pass with same persistent worker

Common flags:

- `--prep-workers 4`
- `--vad-workers 4`
- `--super-batch-active-files 8`
- `--super-batch-preload-files 10`
- `--super-batch-max-decoded-gib 6`

Results (`benchmarks/time_2013_*_20260218_072910.txt`):

| Backend | Wall time (s) | Realtime (x) |
|---|---:|---:|
| `vllm` | 96.35 | 575.56 |
| `transformers` | 337.42 | 164.35 |

- Speedup (`transformers / vllm`): **3.50x**
- Report: `benchmarks/benchmark_2013_20260218_072910.md`

## Experiment B: Cold single-pass tuning for total wall-clock (vLLM)

Goal: optimize *first full run* time (no warmup flow).

Method:

1. No warmup pass
2. Fresh output dir + fresh VAD cache per case
3. Same `download/2013` dataset

Results (`benchmarks/cold_tune_20260218_075512/time_*.txt`):

| Case | Key settings | Wall time (s) | Realtime (x) |
|---|---|---:|---:|
| `gpu_vad_w1` | `--vad-workers 1`, `--prep-workers 4`, active/preload `8/10`, RAM `6GiB` | 182.04 | 304.63 |
| `cpu_vad_w4` | `--vad-workers 4`, `--prep-workers 4`, active/preload `8/10`, RAM `6GiB` | 200.47 | 276.62 |
| `cpu_vad_w4_push` | `--vad-workers 4`, `--prep-workers 6`, active/preload `10/12`, RAM `8GiB` | 213.57 | 259.66 |

Observed effect:

- `gpu_vad_w1` was best in this cold-run matrix.
- Improvement vs `cpu_vad_w4`: **1.10x**
- Improvement vs `cpu_vad_w4_push`: **1.17x**

## Experiment C: Cold no-warmup backend baseline (apples-to-apples)

Goal: measure true first-run backend speed gap.

Method:

1. No warmup
2. No persistent worker
3. Same flags for both backends:
   - `--prep-workers 4`
   - `--vad-workers 1`
   - `--super-batch-active-files 8`
   - `--super-batch-preload-files 10`
   - `--super-batch-max-decoded-gib 6`

Results:

| Backend | Artifact | Wall time (s) | Realtime (x) |
|---|---|---:|---:|
| `transformers` | `benchmarks/baseline_2013_transformers_cold_20260218_081323.time` | 397.34 | 139.57 |
| `vllm` | `benchmarks/baseline_2013_vllm_cold_nopw_20260218_082014.time` | 179.85 | 308.34 |

- Speedup (`transformers / vllm`): **2.21x**

Notes:

- `benchmarks/baseline_2013_transformers_cold_20260218_081133.time` is invalid (interrupted run; `0s`) and excluded.

## What warmup actually does (and does not do)

Warmup run is a full pass, not a lightweight precompute step:

1. Loads ASR runtime/model
2. Performs VAD + decode + ASR + SRT writing
3. Populates VAD cache for subsequent runs

Clarification:

- It is **not** “CPU+GPU VAD precompute together.”
- VAD device is chosen by runtime logic:
  - with `--vad-workers > 1` on CUDA, VAD usually runs on CPU
  - with `--vad-workers 1`, VAD runs on ASR device (CUDA here)

## Current best config choice by objective

If objective is **steady-state throughput** (long-running service / repeated runs):

- Use persistent worker warm-run workflow (`vllm`), which gave `96.35s` on 2013 set.

If objective is **total wall-clock for first full run**:

- Use no-warmup cold-run strategy.
- Current best tested setting:
  - `--asr-backend vllm`
  - `--prep-workers 4`
  - `--vad-workers 1`
  - `--super-batch-active-files 8`
  - `--super-batch-preload-files 10`
  - `--super-batch-max-decoded-gib 6`
