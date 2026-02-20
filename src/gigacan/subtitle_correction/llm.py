from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True, frozen=True)
class CorrectionRequest:
    asr_text: str
    evidence_text: str
    terminology: list[str]


@dataclass(slots=True, frozen=True)
class CorrectionCandidate:
    corrected_text: str
    change_type: str
    confidence: float
    reason: str
    valid: bool = True
    error: str = ""


class CorrectionModel(Protocol):
    def correct_batch(self, requests: list[CorrectionRequest]) -> list[CorrectionCandidate]:
        ...


JSON_CANDIDATE_RE = re.compile(r"\{.*\}", re.DOTALL)
CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_first_json_object(text: str) -> str | None:
    fenced = CODE_BLOCK_RE.search(text)
    if fenced is not None:
        text = fenced.group(1)

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = JSON_CANDIDATE_RE.search(text)
    if match is None:
        return None

    candidate = match.group(0)
    depth = 0
    start = None
    for index, char in enumerate(candidate):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return candidate[start : index + 1]
    return None


def _validate_candidate_text(original: str, corrected: str) -> bool:
    cleaned = corrected.strip()
    if not cleaned:
        return False

    max_allowed = max(int(len(original) * 2.5), len(original) + 20)
    if len(cleaned) > max_allowed:
        return False

    return True


def parse_candidate_response(original: str, response_text: str) -> CorrectionCandidate:
    blob = _extract_first_json_object(response_text)
    if blob is None:
        return CorrectionCandidate(
            corrected_text=original,
            change_type="none",
            confidence=0.0,
            reason="failed_to_parse_json",
            valid=False,
            error="failed_to_parse_json",
        )

    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        return CorrectionCandidate(
            corrected_text=original,
            change_type="none",
            confidence=0.0,
            reason="invalid_json",
            valid=False,
            error="invalid_json",
        )

    corrected_text = str(payload.get("corrected_text", "")).strip()
    if not _validate_candidate_text(original, corrected_text):
        return CorrectionCandidate(
            corrected_text=original,
            change_type="none",
            confidence=0.0,
            reason="candidate_rejected",
            valid=False,
            error="candidate_rejected",
        )

    raw_confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    change_type = str(payload.get("change_type", "minor") or "minor")
    reason = str(payload.get("reason", ""))

    return CorrectionCandidate(
        corrected_text=corrected_text,
        change_type=change_type,
        confidence=confidence,
        reason=reason,
        valid=True,
    )


def build_prompt(request: CorrectionRequest) -> str:
    terms = ", ".join(request.terminology[:30]) if request.terminology else "(none)"
    return (
        "You are a Cantonese subtitle correction assistant. "
        "Use the ASR text as the base and only apply conservative edits when supported by the reference evidence.\n\n"
        "Rules:\n"
        "1. Keep Cantonese style and keep the sentence concise.\n"
        "2. Do not hallucinate facts or names that are not supported by ASR/evidence.\n"
        "3. Prefer minimal changes; preserve meaning and tone.\n"
        "4. If evidence is weak, keep the original text.\n"
        "5. Return ONLY JSON with keys: corrected_text, change_type, confidence, reason.\n\n"
        f"ASR_TEXT: {request.asr_text}\n"
        f"REFERENCE_EVIDENCE: {request.evidence_text}\n"
        f"TERMINOLOGY_HINTS: {terms}\n"
    )


class VllmCorrectionModel:
    """Local vLLM-backed correction model."""

    def __init__(
        self,
        *,
        model: str,
        gpu_memory_utilization: float,
        tensor_parallel_size: int,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        from vllm import LLM, SamplingParams

        self._llm = LLM(
            model=model,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self._sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_new_tokens,
        )

    def correct_batch(self, requests: list[CorrectionRequest]) -> list[CorrectionCandidate]:
        if not requests:
            return []

        prompts = [build_prompt(request) for request in requests]
        outputs = self._llm.generate(prompts, self._sampling_params, use_tqdm=False)

        candidates: list[CorrectionCandidate] = []
        for request, output in zip(requests, outputs):
            try:
                text = output.outputs[0].text if output.outputs else ""
            except Exception:
                text = ""
            candidates.append(parse_candidate_response(request.asr_text, text))

        if len(candidates) < len(requests):
            missing = len(requests) - len(candidates)
            start_index = len(candidates)
            candidates.extend(
                CorrectionCandidate(
                    corrected_text=requests[start_index + index].asr_text,
                    change_type="none",
                    confidence=0.0,
                    reason="missing_model_output",
                    valid=False,
                    error="missing_model_output",
                )
                for index in range(missing)
            )

        return candidates


class OllamaCorrectionModel:
    """Local Ollama-backed correction model."""

    def __init__(
        self,
        *,
        model: str,
        host: str,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        from ollama import Client

        self._client = Client(host=host)
        self._model = model
        self._options = {
            "temperature": temperature,
            "num_predict": max_new_tokens,
        }

    def correct_batch(self, requests: list[CorrectionRequest]) -> list[CorrectionCandidate]:
        if not requests:
            return []

        candidates: list[CorrectionCandidate] = []
        for request in requests:
            prompt = build_prompt(request)
            try:
                response = self._client.generate(
                    model=self._model,
                    prompt=prompt,
                    stream=False,
                    options=self._options,
                )
                text = str(response.get("response", ""))
                candidates.append(parse_candidate_response(request.asr_text, text))
            except Exception:
                candidates.append(
                    CorrectionCandidate(
                        corrected_text=request.asr_text,
                        change_type="none",
                        confidence=0.0,
                        reason="ollama_runtime_error",
                        valid=False,
                        error="ollama_runtime_error",
                    )
                )

        return candidates
