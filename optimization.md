# Transcription Pipeline Optimization Log

This file records optimization strategies, benchmarks, and resulting performance for the Qwen3-ASR + vLLM transcription pipeline.

## Hardware

- GPU: NVIDIA GeForce RTX 5090 (32 GB VRAM)
- CPU: AMD Ryzen 9 9950X3D (16 cores / 32 threads)
- RAM: 58 GiB

## Production dataset

- 14,033 audio files (LegCo proceedings), 948 GB total
- File size: 0.1–494 MB (median 71 MB, mean 68 MB)
- Audio duration: ~24,700 hours total (median ~1.76 h per file)

## Current optimal defaults

```
DEFAULT_SEGMENT_BATCH_SIZE       = 1536
DEFAULT_SUPER_BATCH_ACTIVE_FILES = 48
DEFAULT_SUPER_BATCH_QUEUE_MULTIPLIER = 48
DEFAULT_SUPER_BATCH_PRELOAD_FILES = 96
DEFAULT_SUPER_BATCH_MAX_DECODED_GIB = 25.0
DEFAULT_PREP_WORKERS             = 24
DEFAULT_VAD_WORKERS              = 24
DEFAULT_ASR_PREFETCH_BATCHES     = 6
DEFAULT_VAD_MAX_SEGMENT_MS       = 20000
DEFAULT_VAD_MAX_END_SILENCE_MS   = 300
VAD pre-computation workers      = 8   (multiprocessing, auto-capped)
ffmpeg decode threads             = 1   (per subprocess, -threads 1)
use_vad_cache                     = True
```

### Qwen3 vLLM-specific defaults

```
DEFAULT_VLLM_MAX_MODEL_LEN      = 4096
DEFAULT_VLLM_MAX_NUM_SEQS       = 256
vllm_gpu_memory_utilization      = 0.9
disable_log_stats                = True
enable_prefix_caching            = True   (vLLM v0.14 default)
```

## Two-phase pipeline architecture (2026-02-20)

The pipeline runs in two sequential phases to maximize both CPU and GPU utilization:

### Phase 1 — VAD pre-computation (CPU-bound, multiprocessing)

- Uses `multiprocessing.Pool` with 8 workers (spawn context, each with own GIL)
- Each worker: ffmpeg decode (1-thread subprocess) → streaming chunked VAD → save to cache
- Memory-efficient: ~38 MB peak per worker regardless of file length (no audio accumulation)
- Shortest-first file ordering maximizes cache fill rate (6–14 files/s on small files, ~0.3 files/s on large files)
- CPU utilization: **75–98%** (all cores active)
- GPU: idle (expected, VAD is CPU-only)

### Phase 2 — ASR transcription (GPU-bound, threaded)

- Producer threads load cached VAD (instant) + ffmpeg decode (subprocess, GIL-free)
- Consumer feeds GPU with pre-computed segment batches via cross-file super-batching
- GPU utilization: **35–96%** (avg ~71%) during ASR inference
- CPU: 56–92% (parallel ffmpeg decode + batch preparation)

### E2E benchmark (2026-02-20)

9 representative files spanning P05–P99 (833 MB total, ~20 h audio):

| Phase | Duration | CPU | GPU |
| ----- | -------: | --: | --: |
| VAD pre-computation | 56 s | 21–80% | 0% |
| ASR transcription | 63 s | 56–92% | 35–96% (avg 71%) |
| **Total** | **119 s** | — | — |

### Estimated production run (14,033 files, ~24,700 h audio)

- Phase 1 (VAD): ~4–5 hours
- Phase 2 (ASR): ~22 hours
- **Total: ~27 hours** (vs ~14 days with the original threaded pipeline = **~13× speedup**)

## Optimizations implemented

### Architectural (in code)

1. **Cross-file super-batching** — global segment queue mixes segments from multiple active files into each ASR batch.
2. **Frame-aware batch selection** — sliding-window algorithm picks segments with similar frame counts to reduce padding waste.
3. **ASR payload prefetch** — background thread pool pre-builds batches (including fbank extraction) while the GPU processes the current batch.
4. **Multiprocessing VAD pre-computation** — `multiprocessing.Pool` (8 workers, spawn context) runs FSMN-VAD in separate processes to bypass the GIL. Each worker streams ffmpeg output through chunked VAD without accumulating audio in memory. Results are saved to the VAD cache so the main pipeline only needs ffmpeg decode (subprocess, GIL-free).
5. **VAD segment cache** — caches VAD results per file (`.cache/qwen_srt_vad/`) to skip recomputation on subsequent or interrupted runs.
6. **Segment merge controls** — adjacent short VAD segments are merged up to configurable target/max/gap thresholds.
7. **Decoded-audio RAM budget backpressure** — caps host RAM for decoded audio at 25 GiB, preventing OOM on large batches of files.
8. **Duration-aware file interleaving** — `sort_jobs_by_duration` alternates long and short files to smooth CPU/RAM pressure during the ASR phase.
9. **Binary-split fallback on ASR batch errors** — if a batch fails, binary split isolates the problematic segment(s).

### Qwen3 vLLM fast wrapper (2026-02-20)

10. **Reduced `max_model_len` (65536 → 4096)** — default vLLM context window was 65k tokens, but ASR sequences are only ~300–500 tokens. This limited KV cache concurrency to 2.95×. With `max_model_len=4096`, concurrency increases to 47.2×.
11. **Increased `max_num_seqs` (default → 256)** — allows up to 256 concurrent sequences in the vLLM scheduler, matching the many-short-request pattern of ASR transcription.
12. **`Qwen3VLLMFastWrapper` bypass** — wraps upstream `Qwen3ASRModel` to skip redundant `normalize_audios()`, `split_audio_into_chunks()`, and cache the prompt template. Directly constructs vLLM input dicts and calls `model.generate()`.
13. **`disable_log_stats=True`** — removes per-batch stats logging overhead in the vLLM engine.

### OOM fixes (2026-02-20)

14. **Switched audio loading from torchaudio to ffmpeg** — `torchaudio.load()` created ~14 GB intermediate buffers for a 10-hour file before downsampling. `ffmpeg` decodes directly to 16 kHz mono float32, reducing peak per-file memory from ~25 GB to ~3.5 GB.
15. **Combined decode+VAD into a single worker** — eliminated redundant audio loading by FunASR's VAD model (which internally re-loaded each file at the original sample rate).
16. **Fixed decoded-budget deadlock** — replaced blocking `reserve_decoded_budget()` with non-blocking `try_reserve_decoded_budget()` that alternates with `drain_jobs(block=True)`.

### GIL bottleneck fix (2026-02-20)

The critical production bottleneck: `ThreadPoolExecutor` workers share a single GIL. FunASR's `model.generate()` holds the GIL during Python-level preprocessing, and FSMN-VAD is too lightweight for PyTorch tensor ops to dominate. Result: only 1 VAD ran at a time despite 24 workers, leaving GPU at 0% for 95%+ of the time and only 1–2 CPU cores active.

17. **Multiprocessing VAD pre-computation** — `precompute_vad_multiprocessing()` uses `multiprocessing.Pool` (spawn context) so each worker has its own GIL. True parallelism achieved.
18. **Memory-efficient `compute_vad_streaming()`** — streams ffmpeg output through chunked VAD without accumulating decoded audio. Peak RAM per worker ≈ 38 MB regardless of file length.
19. **Shortest-first file ordering** — VAD pre-computation sorts files by size ascending. Small files complete immediately, filling the cache fast and keeping all workers busy.
20. **`ffmpeg -threads 1`** — limits each ffmpeg subprocess to 1 codec thread. Without this, each ffmpeg spawned ~17 internal threads; with 24 workers that meant 408 threads on 32 logical CPUs.
21. **`torch.set_num_threads(1)` in workers** — FSMN-VAD does not benefit from multi-threading (measured: identical speed at 1 vs 4 threads). Setting to 1 eliminates cross-worker contention.

## Benchmark results

### Qwen3 vLLM tuning (2026-02-20)

Benchmark on `download/2013/` (14 files, 15.4 h audio), RTX 5090, cold start:

| Config | Wall time (s) | Speedup | Concurrency | GPU util |
| ------ | ------------: | ------: | ----------: | -------: |
| Baseline (mml=65536, no bypass) | 155.8 | 1.00× | 2.95× | 26.7% |
| max_model_len=4096 | 113.7 | 1.37× | 47.2× | 33.4% |
| bypass + mml=4096 (optimal) | **89.0** | **1.75×** | 47.2× | 43.4% |

### VAD multiprocessing worker count (2026-02-20)

Benchmark on 30 median-sized files (~70 MB each), with `ffmpeg -threads 1`:

| Workers | Rate (files/s) | Est 14K (h) |
| ------: | -------------: | ----------: |
|       1 |           0.14 |        26.9 |
|       4 |           0.27 |        14.4 |
|       8 |           0.27 |        14.5 |
|      12 |           0.28 |        14.0 |
|      16 |           0.27 |        14.5 |

4–16 workers plateau at ~0.27 files/s on median files. 8 is the default (safe middle ground).

### GPU VAD vs CPU multiprocessing VAD (2026-02-20)

Benchmark on 9 representative production files (P05–P99, 833 MB, ~20 h audio):

| Approach | VAD time | Total time | Throughput |
| -------- | -------: | ---------: | ---------: |
| GPU VAD (cuda:0, 1 process) | 47.5 s | 70.6 s | 0.13 files/s |
| CPU VAD (1 process) | 68.4 s | 91.4 s | 0.10 files/s |
| **CPU VAD (8 multiprocessing workers)** | parallel | ~110 s | **0.27 files/s** |

GPU is 1.44× faster per-file, but CPU wins on **total throughput** (2.1×) because it scales across 8 parallel processes. With a single GPU also needed for ASR inference (28 GB VRAM occupied by vLLM), multiprocessing CPU VAD is the better choice.

### Per-file decode+VAD profiling (2026-02-20)

Single-process streaming decode+VAD times by file size percentile:

| Percentile | File size | Decode | VAD | Streaming | Segments | Audio |
| ---------: | --------: | -----: | --: | --------: | -------: | ----: |
|        P05 |    1.8 MB |  0.1 s | 0.2 s |     0.2 s |       95 | 0.05 h |
|        P25 |   23.0 MB |  1.5 s | 1.5 s |     2.0 s |      332 | 0.61 h |
|        P50 |   70.6 MB |  4.3 s | 4.3 s |     5.3 s |    1,862 | 1.76 h |
|        P75 |   91.6 MB |  5.6 s | 5.0 s |     6.9 s |    2,671 | 2.20 h |
|        P90 |  113.7 MB |  7.1 s | 6.4 s |     8.4 s |    3,844 | 2.83 h |
|        P99 |  332.1 MB | 20.6 s | 18.6 s |    23.8 s |    8,450 | 8.15 h |

### Qwen3 vLLM vs SenseVoice (2026-02-20)

Benchmark on `download/2013/` (14 files, 15.4 h audio), RTX 5090, cold start, VAD cached:

| Engine | Wall time | Realtime factor | Peak RAM |
| ------ | --------: | --------------: | -------: |
| SenseVoice (direct CTC) | **78.0 s** | **711×** | 10.2 GB |
| Qwen3 vLLM (1.7B, bypass) | 100.3 s | 553× | 6.3 GB |

SenseVoice is **1.29× faster** on pure ASR inference. However, for production (14K+ files where VAD dominates wall time), the engine choice matters less.

## Recommended production command

```bash
tmux new -s transcribe ./run_production.sh
```

All defaults are tuned for RTX 5090 + 9950X3D + 58 GB RAM. No flags needed.
