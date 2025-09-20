import os
import sys
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional


# --- DashScope (Qwen-ASR) setup ---
try:
    import dashscope
    from dashscope import MultiModalConversation
except Exception:
    dashscope = None
    MultiModalConversation = None


# Default system prompt used for Cantonese TV dialogue transcription
DEFAULT_SYSTEM_PROMPT = (
    "請將呢段語音轉寫成標準粵文，唔好寫普通話。要求區分「咁噉」「係系喺」，除非固定譯名用「俾」之外規定都用「畀」。"
)


def require_dashscope():
    if dashscope is None or MultiModalConversation is None:
        print("dashscope is required. Install with: pip install dashscope", file=sys.stderr)
        raise SystemExit(1)


def load_api_key_from_env_or_file() -> str:
    """Return API key from env or .env; exits on failure.

    Priority:
    1) `API_KEY` environment variable
    2) `.env` file (CURRENT WORKING DIRECTORY)
    """
    api_key = os.environ.get("API_KEY")
    if api_key:
        return api_key

    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("\"\'")
                    if k == "API_KEY" and v:
                        os.environ["API_KEY"] = v
                        return v
        except Exception:
            pass

    print("Missing API_KEY. Set env var or add to .env", file=sys.stderr)
    raise SystemExit(1)


class RateLimiter:
    """Rolling-window limiter: <= max_rpm starts in any 60s window."""

    def __init__(self, max_rpm: int = 60):
        self.max_rpm = max(1, int(max_rpm))
        self.window = 60.0
        self._lock = threading.Lock()
        self._starts = deque()  # timestamps (monotonic)

    def acquire(self):
        while True:
            now = time.monotonic()
            with self._lock:
                while self._starts and now - self._starts[0] >= self.window:
                    self._starts.popleft()
                if len(self._starts) < self.max_rpm:
                    self._starts.append(now)
                    return
                wait_s = self.window - (now - self._starts[0])
            time.sleep(min(max(wait_s, 0.01), 0.5))


class TranscriptionRateLimitError(RuntimeError):
    """Raised when the ASR backend reports a rate-limit after retries."""


def transcribe_segment(
    segment_path: str,
    language: str = "zh",
    enable_itn: bool = True,
    limiter: "RateLimiter | None" = None,
    max_retries: int = 3,
    backoff_base: float = 0.8,
    system_prompt: Optional[str] = None,
) -> str:
    """Transcribe a single audio segment and return text.

    system_prompt:
        - None: use DEFAULT_SYSTEM_PROMPT
        - "": no system message (omit system role)
        - any other string: use as the system message
    """
    require_dashscope()

    # Build messages with configurable system prompt
    sys_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
    messages: List[dict] = []
    if sys_prompt != "":
        messages.append({"role": "system", "content": [{"text": sys_prompt}]})
    messages.append({"role": "user", "content": [{"audio": segment_path}]})

    attempt = 0
    while True:
        attempt += 1
        if limiter is not None:
            limiter.acquire()
        try:
            # Build asr_options following official API semantics:
            # - If language == 'auto', omit the 'language' field entirely.
            asr_opts = {
                "enable_lid": True,
                "enable_itn": enable_itn,
            }
            if language != "auto":
                asr_opts["language"] = language

            response = MultiModalConversation.call(
                model="qwen3-asr-flash",
                messages=messages,
                result_format="message",
                asr_options=asr_opts,
            )
        except Exception as e:
            # Treat exceptions as transient unless we've exhausted retries
            if attempt <= max_retries:
                time.sleep(min(8.0, backoff_base * (2 ** (attempt - 1))))
                continue
            print(f"ASR request exception: {e}", file=sys.stderr)
            return ""

        if hasattr(response, "status_code") and response.status_code == 200:
            try:
                choice = response.output.choices[0]
                content = choice.message.content[0]
                if isinstance(content, dict) and "text" in content:
                    return content["text"].strip()
            except Exception:
                pass
        status = getattr(response, "status_code", None)
        msg = getattr(response, "message", "unknown error")
        if status in (429, 500, 502, 503, 504) and attempt <= max_retries:
            backoff = min(8.0, backoff_base * (2 ** (attempt - 1)))
            time.sleep(backoff)
            continue
        if status == 429:
            raise TranscriptionRateLimitError(msg)
        print(f"ASR request failed (status={status}): {msg}", file=sys.stderr)
        return ""


def transcribe_segments(
    prepared: List[Tuple[int, float, float, str]],
    *,
    language: str = "zh",
    enable_itn: bool = True,
    concurrency: int = 4,
    max_rpm: int = 60,
    max_retries: int = 3,
    backoff_base: float = 0.8,
    system_prompt: Optional[str] = None,
    max_rounds: int = 3,
) -> List[Tuple[float, float, str]]:
    """Transcribe prepared segments concurrently and return (start, end, text).

    Pass a custom `system_prompt` to control ASR style/behavior per task.
    See `transcribe_segment` for semantics (None/""/custom).
    """
    api_key = load_api_key_from_env_or_file()
    try:
        dashscope.api_key = api_key
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    except Exception:
        pass

    limiter = RateLimiter(max_rpm)
    prepared_lookup: Dict[int, Tuple[int, float, float, str]] = {
        idx: (idx, start, end, path) for idx, start, end, path in prepared
    }

    total = len(prepared_lookup)
    completed = 0
    results_text: Dict[int, str] = {}
    pending_indices = list(prepared_lookup.keys())
    round_no = 1

    while pending_indices and round_no <= max(1, int(max_rounds)):
        rate_limited: List[int] = []
        max_workers = min(max(1, concurrency), len(pending_indices))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(
                    transcribe_segment,
                    prepared_lookup[idx][3],
                    language,
                    enable_itn,
                    limiter,
                    max_retries,
                    backoff_base,
                    system_prompt,
                ): idx
                for idx in pending_indices
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                _, start_s, end_s, _path = prepared_lookup[idx]
                try:
                    text = fut.result()
                except TranscriptionRateLimitError as exc:
                    rate_limited.append(idx)
                    print(
                        f"[retry] Rate limit for segment {idx:04d}: {exc} (round {round_no})",
                        file=sys.stderr,
                    )
                    continue
                except Exception as e:
                    print(f"[Error] Segment {idx} failed: {e}", file=sys.stderr)
                    text = ""
                completed += 1
                print(
                    f"[{completed}/{total}] ASR {fmt_time(start_s)} - {fmt_time(end_s)} (seg {idx:04d})"
                )
                results_text[idx] = text

        if not rate_limited:
            break

        pending_indices = rate_limited
        round_no += 1
        sleep_s = min(15.0, backoff_base * (2 ** (round_no - 1)))
        time.sleep(sleep_s)

    # Any segments still pending after exhausting rounds get empty transcripts
    for idx in pending_indices:
        if idx not in results_text:
            _, start_s, end_s, _path = prepared_lookup[idx]
            print(
                f"[warn] Giving up on segment {idx:04d} "
                f"({fmt_time(start_s)} - {fmt_time(end_s)}) after rate limits.",
                file=sys.stderr,
            )
            completed += 1
            results_text[idx] = ""
            print(
                f"[{completed}/{total}] ASR {fmt_time(start_s)} - {fmt_time(end_s)} (seg {idx:04d})"
            )

    entries: List[Tuple[float, float, str]] = []
    for idx, start_s, end_s, _path in sorted(prepared_lookup.values(), key=lambda x: x[0]):
        entries.append((start_s, end_s, results_text.get(idx, "")))
    return entries


def fmt_time(seconds: float) -> str:
    # HH:MM:SS.mmm for logs
    ms = int(round((seconds - int(seconds)) * 1000.0))
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if ms == 1000:
        ms = 0
        sec += 1
        if sec == 60:
            sec = 0
            m += 1
            if m == 60:
                m = 0
                h += 1
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"


def write_webvtt(out_path: str, entries: List[Tuple[float, float, str]]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for idx, (start_s, end_s, text) in enumerate(entries, start=1):
            f.write(f"{idx}\n")
            f.write(f"{fmt_time(start_s)} --> {fmt_time(end_s)}\n")
            f.write((text or "").strip() + "\n\n")
