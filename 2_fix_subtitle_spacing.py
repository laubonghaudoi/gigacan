from __future__ import annotations

import argparse
from pathlib import Path


# --- CONFIGURATION ---
DEFAULT_SUBTITLE_DIR = Path("subtitle")
DEFAULT_EXTENSIONS = {".srt", ".vtt"}
# --- END CONFIGURATION ---


# CJK ideographs + CJK punctuation blocks.
_CJK_TOKEN_CLASS = r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u3000-\u303F\uFF00-\uFFEF]"


def _build_spacing_pattern():
    import re

    return re.compile(rf"(?<={_CJK_TOKEN_CLASS})[ \t]+(?={_CJK_TOKEN_CLASS})")


SPACE_BETWEEN_CJK_PATTERN = _build_spacing_pattern()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove spaces between CJK characters/punctuation in subtitle files."
    )
    parser.add_argument(
        "--subtitle-dir",
        type=Path,
        default=DEFAULT_SUBTITLE_DIR,
        help=f"Subtitle root directory (default: {DEFAULT_SUBTITLE_DIR})",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=sorted(DEFAULT_EXTENSIONS),
        help=(
            "Subtitle extension to include (repeatable). "
            + f"Default: {sorted(DEFAULT_EXTENSIONS)}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each changed file path.",
    )
    return parser.parse_args()


def _normalise_extensions(values: list[str]) -> set[str]:
    normalised: set[str] = set()
    for value in values:
        ext = value.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        normalised.add(ext)
    return normalised


def _iter_subtitle_files(root: Path, extensions: set[str]) -> list[Path]:
    if not root.is_dir():
        return []
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files)


def _clean_text(text: str) -> tuple[str, int]:
    return SPACE_BETWEEN_CJK_PATTERN.subn("", text)


def main() -> None:
    args = parse_args()

    subtitle_dir: Path = args.subtitle_dir
    extensions = _normalise_extensions(args.ext)
    dry_run: bool = bool(args.dry_run)
    verbose: bool = bool(args.verbose)

    if not subtitle_dir.is_dir():
        print(f"Error: subtitle directory not found: {subtitle_dir}")
        return

    files = _iter_subtitle_files(subtitle_dir, extensions)
    if not files:
        print(
            "No subtitle files found under "
            + f"{subtitle_dir} with extensions {sorted(extensions)}."
        )
        return

    changed_files = 0
    touched_files = 0
    total_replacements = 0
    failed_files = 0

    for path in files:
        touched_files += 1
        try:
            original_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            failed_files += 1
            print(f"[error] Failed to read {path}: {exc}")
            continue

        cleaned_text, replacements = _clean_text(original_text)
        if replacements <= 0:
            continue

        changed_files += 1
        total_replacements += replacements
        if verbose:
            print(f"[change] {path} ({replacements} replacements)")

        if dry_run:
            continue

        try:
            path.write_text(cleaned_text, encoding="utf-8")
        except OSError as exc:
            failed_files += 1
            print(f"[error] Failed to write {path}: {exc}")

    mode = "Dry-run" if dry_run else "Done"
    print(
        f"{mode}. Scanned {touched_files} files; "
        + f"changed {changed_files}; replacements {total_replacements}; "
        + f"read/write errors {failed_files}."
    )


if __name__ == "__main__":
    main()
