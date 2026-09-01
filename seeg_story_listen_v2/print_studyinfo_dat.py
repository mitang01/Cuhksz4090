#!/usr/bin/env python3
"""Decode Chinese StudyInfo .dat files and summarize them into one text report."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_INPUT = Path("/share/workspace3/ieeg/seeg/story_listen_v2")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "studyinfo_dat_summary.txt"
CANDIDATE_ENCODINGS = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "cp936",
    "big5",
    "shift_jis",
)
FALLBACK_ENCODINGS = CANDIDATE_ENCODINGS + ("latin1",)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
KEY_VALUE_RE = re.compile(r"^([^:=\r\n]{1,80})\s*[:=]\s*(.+)$")
SECTION_RE = re.compile(r"^\[(.+)\]\s*$")


@dataclass(frozen=True)
class DecodedText:
    encoding: str
    text: str
    score: float
    mode: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find StudyInfo .dat files under the story-listening tree, decode "
            "Chinese text that looks scrambled in UTF-8 editors, and write one "
            "readable summary report."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Directory to search recursively (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Summary text file destination (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--encoding",
        help=(
            "Force one text encoding. By default the script auto-detects among "
            "utf-8, gb18030/gbk/cp936, big5, and shift_jis."
        ),
    )
    parser.add_argument(
        "--all-dat",
        action="store_true",
        help=(
            "Include every .dat file, not only StudyInfo.dat. Temporary "
            "mmap preload files named *_preload.dat are still skipped."
        ),
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=8_000_000,
        help="Skip files larger than this many bytes (default: 8000000)",
    )
    return parser.parse_args(argv)


def find_dat_files(root: Path, *, all_dat: bool) -> list[Path]:
    """Return StudyInfo.dat paths, or all .dat paths when requested."""
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() != ".dat":
            continue
        name = path.name.casefold()
        if name.endswith("_preload.dat") or name.startswith("."):
            continue
        if all_dat or name == "studyinfo.dat":
            paths.append(path)
    return sorted(paths, key=lambda path: str(path).casefold())


def infer_subject(path: Path, root: Path) -> str:
    """Best-effort subject label from path components such as sub002."""
    try:
        parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        parts = path.parts
    for part in parts:
        if re.fullmatch(r"sub\d+", part, flags=re.IGNORECASE):
            return part
    return "unknown"


def _score_text(text: str, *, high_byte_fraction: float, encoding: str) -> float:
    if not text:
        return -1e9
    length = len(text)
    replacement = text.count("\ufffd")
    control = len(CONTROL_RE.findall(text))
    cjk = len(CJK_RE.findall(text))
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in text)
    newlines = text.count("\n")
    score = (
        printable / length
        + 2.0 * min(cjk, 200) / 200
        + 0.1 * min(newlines, 50) / 50
        - 8.0 * replacement / length
        - 3.0 * control / length
    )
    # Scrambled Chinese files contain high bytes; prefer real CJK decodes.
    if high_byte_fraction > 0.02 and cjk == 0:
        score -= 1.5
    if encoding == "latin1" and high_byte_fraction > 0.02:
        score -= 1.0
    return score


def decode_bytes(data: bytes, encoding: str | None = None) -> DecodedText:
    """Decode whole-file text, falling back to printable-string extraction."""
    encodings = (encoding,) if encoding else FALLBACK_ENCODINGS
    high_byte_fraction = (
        sum(byte >= 0x80 for byte in data) / len(data) if data else 0.0
    )
    best: DecodedText | None = None
    for candidate in encodings:
        try:
            text = data.decode(candidate)
            used_replace = False
        except UnicodeDecodeError:
            text = data.decode(candidate, errors="replace")
            used_replace = True
        # Null bytes usually mean the payload is mixed/binary; prefer string
        # extraction over a whole-file latin1/mojibake decode.
        if b"\x00" in data and candidate == "latin1":
            continue
        score = _score_text(
            text, high_byte_fraction=high_byte_fraction, encoding=candidate
        )
        if used_replace:
            score -= 0.25
        scored = DecodedText(
            encoding=candidate,
            text=_normalize_newlines(text),
            score=score,
            mode="text",
        )
        if best is None or scored.score > best.score:
            best = scored

    extracted = _extract_strings(data, encodings, high_byte_fraction)
    if best is None:
        return extracted
    # Prefer whole-file text for ordinary StudyInfo files; use string
    # extraction when null bytes or a clearly better decode appears.
    if b"\x00" in data and extracted.score >= best.score - 0.05:
        return extracted
    if extracted.score > best.score + 0.15:
        return extracted
    return best


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip("\n") + "\n"


def _extract_strings(
    data: bytes, encodings: Iterable[str], high_byte_fraction: float
) -> DecodedText:
    """Pull readable runs out of binary or mixed files."""
    runs: list[bytes] = []
    current = bytearray()
    for byte in data:
        if 32 <= byte <= 126 or byte in (9, 10, 13) or byte >= 0x80:
            current.append(byte)
        else:
            if len(current) >= 4:
                runs.append(bytes(current))
            current.clear()
    if len(current) >= 4:
        runs.append(bytes(current))

    best: DecodedText | None = None
    for candidate in encodings:
        chunks: list[str] = []
        for run in runs:
            try:
                chunks.append(run.decode(candidate))
            except UnicodeDecodeError:
                chunks.append(run.decode(candidate, errors="replace"))
        text = _normalize_newlines("\n".join(chunks))
        scored = DecodedText(
            encoding=candidate,
            text=text,
            score=_score_text(
                text, high_byte_fraction=high_byte_fraction, encoding=candidate
            ),
            mode="strings",
        )
        if best is None or scored.score > best.score:
            best = scored
    if best is None:
        return DecodedText("latin1", "\n", -1e9, "strings")
    return best


def extract_fields(text: str) -> dict[str, str]:
    """Collect simple section/key-value pairs for the summary table."""
    fields: dict[str, str] = {}
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        section_match = SECTION_RE.match(stripped)
        if section_match:
            section = section_match.group(1).strip()
            continue
        key_match = KEY_VALUE_RE.match(stripped)
        if not key_match:
            continue
        key = key_match.group(1).strip()
        value = key_match.group(2).strip()
        label = f"{section}.{key}" if section else key
        if label not in fields:
            fields[label] = value
    return fields


def summarize_file(
    path: Path,
    root: Path,
    *,
    encoding: str | None,
    max_bytes: int,
) -> tuple[str, dict[str, str], DecodedText | None, str | None]:
    """Return report block, fields, decode result, and optional error."""
    subject = infer_subject(path, root)
    size = path.stat().st_size
    if size > max_bytes:
        error = f"skipped: file larger than --max-bytes ({size} > {max_bytes})"
        block = (
            f"{'=' * 80}\n"
            f"file: {path}\n"
            f"subject: {subject}\n"
            f"bytes: {size}\n"
            f"ERROR: {error}\n\n"
        )
        return block, {}, None, error

    data = path.read_bytes()
    decoded = decode_bytes(data, encoding=encoding)
    fields = extract_fields(decoded.text)
    preview_fields = "\n".join(
        f"  {key}: {value}" for key, value in list(fields.items())[:40]
    )
    if not preview_fields:
        preview_fields = "  (no key=value or section fields detected)"
    block = (
        f"{'=' * 80}\n"
        f"file: {path}\n"
        f"subject: {subject}\n"
        f"bytes: {size}\n"
        f"decoded_as: {decoded.encoding} ({decoded.mode})\n"
        f"detected_fields:\n{preview_fields}\n"
        f"{'-' * 80}\n"
        f"decoded_text:\n"
        f"{decoded.text}"
        f"\n"
    )
    return block, fields, decoded, None


def write_summary(
    paths: Sequence[Path],
    root: Path,
    output: Path,
    *,
    encoding: str | None,
    max_bytes: int,
) -> int:
    """Decode every file and write one combined UTF-8 report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    overview_rows: list[str] = []
    blocks: list[str] = []

    for path in paths:
        block, fields, decoded, error = summarize_file(
            path, root, encoding=encoding, max_bytes=max_bytes
        )
        blocks.append(block)
        subject = infer_subject(path, root)
        if error:
            failures += 1
            overview_rows.append(f"{subject}\tERROR\t{path}\t{error}")
            print(f"ERROR: {path}: {error}", file=sys.stderr)
            continue
        assert decoded is not None
        overview_rows.append(
            f"{subject}\t{decoded.encoding}/{decoded.mode}\t{path}\t"
            f"{len(fields)} fields"
        )
        print(
            f"OK   {path} ({decoded.encoding}, {decoded.mode}, "
            f"{len(fields)} fields)",
            file=sys.stderr,
        )

    report = [
        f"StudyInfo .dat summary for {root.resolve()}",
        f"files discovered: {len(paths)}",
        f"succeeded: {len(paths) - failures}",
        f"failed: {failures}",
        "",
        "overview:",
        *overview_rows,
        "",
        "per-file decoded content:",
        "",
        *blocks,
    ]
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote summary to {output}", file=sys.stderr)
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_bytes < 1:
        print("ERROR: --max-bytes must be positive", file=sys.stderr)
        return 2

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    paths = find_dat_files(input_dir, all_dat=args.all_dat)
    if not paths:
        target = "any .dat files" if args.all_dat else "StudyInfo.dat files"
        print(f"ERROR: no {target} found below {input_dir}", file=sys.stderr)
        return 2

    output = args.output.expanduser().resolve()
    failures = write_summary(
        paths,
        input_dir,
        output,
        encoding=args.encoding,
        max_bytes=args.max_bytes,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
