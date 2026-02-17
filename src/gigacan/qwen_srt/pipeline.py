from __future__ import annotations

import tempfile
from pathlib import Path

from .config import TranscribeConfig
from .postprocess import CantonesePostProcessor, clean_asr_text
from .runtime import (
    build_asr_model,
    build_vad_model,
    resolve_device,
    resolve_qwen_dtype,
    resolve_segment_batch_size,
)
from .srt import extract_segment_to_wav, write_srt


def resolve_context_prompt(config: TranscribeConfig) -> tuple[str, bool]:
    use_prompt = config.use_prompt and bool(config.qwen_context.strip())
    context_prompt = config.qwen_context if use_prompt else ""
    return context_prompt, use_prompt


def transcribe_to_srt(config: TranscribeConfig) -> Path:
    if not config.audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {config.audio}")

    resolved_device = resolve_device(config.device)
    qwen_dtype = resolve_qwen_dtype(config.qwen_dtype, resolved_device)
    segment_batch_size = resolve_segment_batch_size(
        resolved_device, config.segment_batch_size
    )
    context_prompt, use_prompt = resolve_context_prompt(config)

    print(f"Using device: {resolved_device}")
    print(f"Qwen dtype: {qwen_dtype}")
    print(f"Segment batch size: {segment_batch_size}")
    print(f"Qwen context prompt: {'enabled' if use_prompt else 'disabled'}")

    asr_model = build_asr_model(config, resolved_device, qwen_dtype, segment_batch_size)
    vad_model = build_vad_model(resolved_device, config.vad_max_segment_ms)
    postprocessor = CantonesePostProcessor()

    vad_res = vad_model.generate(input=str(config.audio))
    if not vad_res or "value" not in vad_res[0]:
        raise RuntimeError(f"Unexpected VAD output for {config.audio}: {vad_res}")
    raw_segments = vad_res[0]["value"]
    segments = [
        (int(start), int(end))
        for start, end in raw_segments
        if int(end) - int(start) >= config.min_segment_ms
    ]
    print(f"VAD segments: {len(raw_segments)}; used: {len(segments)}")

    entries: list[tuple[int, int, str]] = []
    with tempfile.TemporaryDirectory(prefix="qwen3_segments_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for batch_start in range(0, len(segments), segment_batch_size):
            batch_segments = segments[batch_start : batch_start + segment_batch_size]
            batch_wavs: list[str] = []
            for seg_idx, (start_ms, end_ms) in enumerate(
                batch_segments, start=batch_start + 1
            ):
                seg_wav = tmpdir_path / f"seg_{seg_idx:05d}.wav"
                extract_segment_to_wav(config.audio, start_ms, end_ms, seg_wav)
                batch_wavs.append(str(seg_wav))

            results = asr_model.transcribe(
                audio=batch_wavs,
                context=context_prompt,
                language=config.qwen_language,
            )
            if len(results) != len(batch_segments):
                raise RuntimeError(
                    f"ASR result size mismatch: got {len(results)}, expected {len(batch_segments)}"
                )

            for item, (start_ms, end_ms) in zip(results, batch_segments):
                raw_text = str(getattr(item, "text", ""))
                text = clean_asr_text(raw_text)
                text = postprocessor.apply(text)
                if text:
                    entries.append((start_ms, end_ms, text))

            done = min(batch_start + len(batch_segments), len(segments))
            print(f"Transcribed {done}/{len(segments)} segments")

    config.output_srt.parent.mkdir(parents=True, exist_ok=True)
    write_srt(config.output_srt, entries)
    print(f"Wrote SRT: {config.output_srt}")
    return config.output_srt
