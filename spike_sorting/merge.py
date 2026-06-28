"""
Merge Intan MAT chunks into one binary stream per session.

Output channel layout per sample:
  [amplifier_data channels..., board_dig_in_data channels...]

For the current datasets that means:
  - 128 amplifier channels
  - 8 board digital input channels
  - total = 136 channels

Notes:
  - For subjects in LABEL_AWARE_PADDING_SUBJECTS (currently sub5), some MAT chunks
    drop a single amplifier channel (e.g. A-008 missing -> 127 channels). For those
    chunks the missing channel is zero-padded at its correct label position (detected
    from the per-file channel labels, expected to run A-001 .. A-128), NOT at the end.
  - sub6 chunks always carry the full 128 amplifier channels, so label-aware padding
    is not enabled for sub6.
  - Existing output files are overwritten on each run (truncated before writing).
"""

import errno
from pathlib import Path
import re
import time
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat


MAT_FILE_GLOB = "Temp_26012*.mat"
MERGE_JOBS = [
    {
        "subject": "sub5",
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5/Temp_260121_095012"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub5_session1"),
    },
    {
        "subject": "sub5",
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5/Temp_260121_103639"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub5_session2"),
    },
    {
        "subject": "sub5",
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5/Temp_260121_104824"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub5_session3"),
    },
    {
        "subject": "sub5",
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5/Temp_260121_105933"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub5_session4"),
    },
    {
        "subject": "sub5",
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5/Temp_260121_125018"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub5_session5"),
    },
    {
        "subject": "sub5",
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5/Temp_260121_130023"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub5_session6"),
    },
    {
        "subject": "sub6",
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0123/bistable_sub6_1/Temp_260123_115236"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub6_session1"),
    },
    {
        "subject": "sub6",
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0123/bistable_sub6_2/Temp_260123_120327"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub6_session3"),
    },
    {
        "subject": "sub6",
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0123/bistable_sub6_3/Temp_260123_121427"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub6_session6"),
    },
]
GAIN_TO_UV = 0.195  # Intan RHD2000 conversion factor
AMPLIFIER_CHANNELS_EXPECTED = 128
DIGITAL_IN_CHANNELS_EXPECTED = 8
# Full amplifier channel label set for 128-channel Intan exports (A-001 .. A-128).
EXPECTED_AMPLIFIER_LABELS = [f"A-{i:03d}" for i in range(1, AMPLIFIER_CHANNELS_EXPECTED + 1)]
# Subjects whose MAT chunks may drop a single amplifier channel (e.g. A-008 missing,
# leaving 127 channels). For these, pad the missing channel at its correct label
# position (identified from the per-file channel labels) instead of at the end.
LABEL_AWARE_PADDING_SUBJECTS = {"sub5"}
MAX_IO_RETRIES = 6
BASE_RETRY_SLEEP_S = 1.0


def sort_key(path: Path):
    """
    Sort by all digit groups in filename first, then by name.
    This keeps files in a stable temporal-like order for common naming patterns.
    """
    digits = re.findall(r"\d+", path.stem)
    if digits:
        return tuple(int(d) for d in digits), path.name
    return (path.name,)


def is_retryable_io_error(exc: OSError) -> bool:
    return exc.errno in {errno.ESTALE, errno.EIO, errno.ETIMEDOUT}


def retry_delay_s(attempt: int) -> float:
    return BASE_RETRY_SLEEP_S * (2**attempt)


def open_with_retry(path: Path, mode: str):
    last_exc: OSError | None = None
    for attempt in range(MAX_IO_RETRIES):
        try:
            return path.open(mode)
        except OSError as exc:
            last_exc = exc
            if not is_retryable_io_error(exc) or attempt == MAX_IO_RETRIES - 1:
                raise
            wait_s = retry_delay_s(attempt)
            print(
                f"  [WARN] open failed ({exc}). retry {attempt + 1}/{MAX_IO_RETRIES} "
                f"after {wait_s:.1f}s: {path}"
            )
            time.sleep(wait_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Unexpected open retry failure for {path}")


def safe_stat_size(path: Path) -> int:
    last_exc: OSError | None = None
    for attempt in range(MAX_IO_RETRIES):
        try:
            return path.stat().st_size
        except OSError as exc:
            last_exc = exc
            if not is_retryable_io_error(exc) or attempt == MAX_IO_RETRIES - 1:
                raise
            wait_s = retry_delay_s(attempt)
            print(
                f"  [WARN] stat failed ({exc}). retry {attempt + 1}/{MAX_IO_RETRIES} "
                f"after {wait_s:.1f}s: {path}"
            )
            time.sleep(wait_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Unexpected stat retry failure for {path}")


def touch_truncate_with_retry(path: Path) -> None:
    with open_with_retry(path, "wb"):
        return


def append_array_with_retry(path: Path, arr: np.ndarray, chunk_name: str) -> None:
    last_exc: OSError | None = None
    for attempt in range(MAX_IO_RETRIES):
        try:
            with open_with_retry(path, "ab") as out_f:
                arr.tofile(out_f)
            return
        except OSError as exc:
            last_exc = exc
            if not is_retryable_io_error(exc) or attempt == MAX_IO_RETRIES - 1:
                raise
            wait_s = retry_delay_s(attempt)
            print(
                f"  [WARN] append failed for {chunk_name} ({exc}). "
                f"retry {attempt + 1}/{MAX_IO_RETRIES} after {wait_s:.1f}s."
            )
            time.sleep(wait_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Unexpected append retry failure for {chunk_name}")


def orient_samples_channels(data: np.ndarray, expected_channels: int, name: str, mat_path: Path) -> np.ndarray:
    """Return array in (samples, channels) orientation."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected {name} shape {arr.shape} in {mat_path}")

    if arr.shape[1] == expected_channels:
        return arr
    if arr.shape[0] == expected_channels and arr.shape[1] != expected_channels:
        return arr.T

    # Fallback heuristic: usually samples >> channels.
    if arr.shape[0] > arr.shape[1]:
        return arr
    return arr.T


def normalize_channel_count(
    arr: np.ndarray,
    expected_channels: int,
    stream_name: str,
    mat_path: Path,
    pad_value: int = 0,
) -> np.ndarray:
    n_samples, n_ch = arr.shape
    if n_ch == expected_channels:
        return arr
    if n_ch < expected_channels:
        missing = expected_channels - n_ch
        print(
            f"  [WARN] {mat_path.name}: {stream_name} has {n_ch} channels, "
            f"expected {expected_channels}. Padding {missing} missing channel(s) with {pad_value}."
        )
        pad = np.full((n_samples, missing), pad_value, dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=1)

    extra = n_ch - expected_channels
    print(
        f"  [WARN] {mat_path.name}: {stream_name} has {n_ch} channels, "
        f"expected {expected_channels}. Truncating last {extra} extra channel(s)."
    )
    return arr[:, :expected_channels]


def _decode_hdf5_name_dataset(f: "h5py.File", ref: Any) -> str:
    try:
        ds = f[ref]
        arr = np.array(ds)
    except Exception:
        return ""
    if arr.size == 0:
        return ""
    if arr.dtype.kind in {"U", "S"}:
        return "".join(str(x) for x in arr.ravel()).strip()
    flat = np.asarray(arr).ravel()
    if np.issubdtype(flat.dtype, np.integer):
        return "".join(chr(int(c)) for c in flat if 0 < int(c) < 256).strip()
    return ""


def _decode_hdf5_channel_names(f: "h5py.File", group_key: str) -> list[str]:
    if group_key not in f:
        return []
    g = f[group_key]
    for key in ("native_channel_name", "custom_channel_name"):
        if key not in g:
            continue
        refs = np.array(g[key], dtype=object).ravel()
        out: list[str] = []
        for i, ref in enumerate(refs, start=1):
            name = _decode_hdf5_name_dataset(f, ref)
            out.append(name if name else f"{group_key}_{i}")
        return out
    return []


def _extract_legacy_name(one_channel_obj: Any, idx: int, group_key: str) -> str:
    for field in ("native_channel_name", "custom_channel_name"):
        if hasattr(one_channel_obj, field):
            value = getattr(one_channel_obj, field)
            if isinstance(value, np.ndarray):
                if value.dtype.kind in {"U", "S"}:
                    s = "".join(str(x) for x in value.ravel())
                    return s.strip() or f"{group_key}_{idx}"
                if np.issubdtype(value.dtype, np.integer):
                    s = "".join(chr(int(x)) for x in value.ravel() if int(x) != 0)
                    return s.strip() or f"{group_key}_{idx}"
            if isinstance(value, str):
                return value.strip() or f"{group_key}_{idx}"
    if isinstance(one_channel_obj, np.void) and one_channel_obj.dtype.names:
        for field in ("native_channel_name", "custom_channel_name"):
            if field in one_channel_obj.dtype.names:
                v = one_channel_obj[field]
                arr = np.asarray(v).ravel()
                if arr.dtype.kind in {"U", "S"}:
                    s = "".join(str(x) for x in arr)
                    return s.strip() or f"{group_key}_{idx}"
                if np.issubdtype(arr.dtype, np.integer):
                    s = "".join(chr(int(x)) for x in arr if int(x) != 0)
                    return s.strip() or f"{group_key}_{idx}"
    return f"{group_key}_{idx}"


def _decode_legacy_channel_names(mat_data: dict[str, Any], group_key: str) -> list[str]:
    if group_key not in mat_data:
        return []
    arr = np.asarray(mat_data[group_key]).ravel()
    return [_extract_legacy_name(obj, i, group_key) for i, obj in enumerate(arr, start=1)]


def read_amplifier_channel_labels(mat_path: Path) -> list[str]:
    """Read amplifier channel labels (e.g. A-001 .. A-128) from a MAT file."""
    if h5py.is_hdf5(str(mat_path)):
        try:
            with h5py.File(mat_path, "r") as f:
                return _decode_hdf5_channel_names(f, "amplifier_channels")
        except OSError:
            return []
    try:
        mat_data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception:
        return []
    return _decode_legacy_channel_names(mat_data, "amplifier_channels")


def normalize_amplifier_by_labels(
    arr: np.ndarray,
    expected_labels: list[str],
    actual_labels: list[str],
    stream_name: str,
    mat_path: Path,
    pad_value: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Rebuild amplifier data so columns line up with expected_labels.

    Missing labels (e.g. A-008 absent from a 127-channel chunk) become a
    zero-valued column at the correct position, instead of being padded at the
    end. Extra/unknown labels are dropped.

    Returns (out, missing_labels) where missing_labels lists the labels that
    were zero-padded, in expected-column order.
    """
    n_samples, n_ch = arr.shape
    expected_n = len(expected_labels)
    label_to_col: dict[str, int] = {}
    for col, label in enumerate(actual_labels):
        label_to_col.setdefault(label, col)

    missing = [label for label in expected_labels if label not in label_to_col]
    extra = [label for label in actual_labels if label not in expected_labels]
    missing_set = set(missing)
    detail = ", ".join(
        f"{label} (column {i + 1} of {expected_n})"
        for i, label in enumerate(expected_labels)
        if label in missing_set
    )
    print(
        f"  [PATCH] {mat_path.name}: {stream_name} {n_ch}->{expected_n} channels; "
        f"zero-padded missing channel(s): {detail or 'none'}; "
        f"extra labels dropped: {extra}."
    )

    out = np.full((n_samples, expected_n), pad_value, dtype=arr.dtype)
    for out_col, label in enumerate(expected_labels):
        src_col = label_to_col.get(label)
        if src_col is not None:
            out[:, out_col] = arr[:, src_col]
    return out, missing


def flatten_time_vector(t: np.ndarray | None) -> np.ndarray | None:
    if t is None:
        return None
    arr = np.asarray(t).squeeze()
    if arr.ndim != 1 or arr.size == 0:
        return None
    return arr.astype(np.float64, copy=False)


def align_digital_to_amplifier(
    amplifier: np.ndarray,
    dig: np.ndarray,
    t_amplifier: np.ndarray | None,
    t_dig: np.ndarray | None,
    mat_path: Path,
) -> np.ndarray:
    """
    Align digital samples to amplifier samples.

    Priority:
      1) if same sample count, keep as-is;
      2) if both t vectors exist, map each amplifier timestamp to nearest t_dig;
      3) fallback trim to min length.
    """
    n_amp = amplifier.shape[0]
    n_dig = dig.shape[0]
    if n_amp == n_dig:
        return dig

    if t_amplifier is not None and t_dig is not None and t_amplifier.size == n_amp and t_dig.size == n_dig:
        idx = np.searchsorted(t_dig, t_amplifier, side="left")
        idx = np.clip(idx, 0, n_dig - 1)
        # Choose nearest between left candidate and immediate predecessor.
        prev = np.maximum(idx - 1, 0)
        choose_prev = np.abs(t_dig[prev] - t_amplifier) < np.abs(t_dig[idx] - t_amplifier)
        idx[choose_prev] = prev[choose_prev]
        return dig[idx, :]

    n = min(n_amp, n_dig)
    print(
        f"  [WARN] Sample mismatch without usable time vectors in {mat_path.name}: "
        f"amplifier={n_amp}, dig={n_dig}. Trimming both to {n}."
    )
    return dig[:n, :]


def load_streams(mat_path: Path, label_aware_padding: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load amplifier and board digital inputs from one MAT file.

    Returns:
      amplifier_uV: float array, shape (samples, 128)
      board_dig: int16 array, shape (samples, 8), values in {0,1}
      patch_info: dict with keys
        - "missing_labels": list[str] of amplifier labels zero-padded at their
          correct position (only populated when label_aware_padding triggers the
          label-based path); empty otherwise.
        - "fallback_end_padding": int count of channels end-padded when labels
          could not be read/matched (safety net only).

    When label_aware_padding is True (used for subjects whose chunks may drop a
    single amplifier channel, e.g. sub5), a short-channel amplifier block is
    rebuilt so the missing channel lands at its correct label position (detected
    from the per-file channel labels) rather than being padded at the end.
    """
    amp_data: Any
    dig_data: Any
    t_amp: Any = None
    t_dig: Any = None

    if h5py.is_hdf5(str(mat_path)):
        try:
            with h5py.File(mat_path, "r") as h5f:
                if "amplifier_data" not in h5f:
                    raise KeyError("'amplifier_data' not found in file")
                if "board_dig_in_data" not in h5f:
                    raise KeyError("'board_dig_in_data' not found in file")
                amp_data = np.array(h5f["amplifier_data"])
                dig_data = np.array(h5f["board_dig_in_data"])
                if "t_amplifier" in h5f:
                    t_amp = np.array(h5f["t_amplifier"])
                if "t_dig" in h5f:
                    t_dig = np.array(h5f["t_dig"])
        except OSError as exc:
            raise RuntimeError(f"Failed to read HDF5 MAT file: {mat_path}") from exc
    else:
        try:
            mat_data = loadmat(mat_path)
        except NotImplementedError:
            # Safety fallback in case a v7.3 file is misdetected.
            with h5py.File(mat_path, "r") as h5f:
                if "amplifier_data" not in h5f:
                    raise KeyError("'amplifier_data' not found in file")
                if "board_dig_in_data" not in h5f:
                    raise KeyError("'board_dig_in_data' not found in file")
                amp_data = np.array(h5f["amplifier_data"])
                dig_data = np.array(h5f["board_dig_in_data"])
                if "t_amplifier" in h5f:
                    t_amp = np.array(h5f["t_amplifier"])
                if "t_dig" in h5f:
                    t_dig = np.array(h5f["t_dig"])
        else:
            if "amplifier_data" not in mat_data:
                raise KeyError(f"'amplifier_data' not found in {mat_path}")
            if "board_dig_in_data" not in mat_data:
                raise KeyError(f"'board_dig_in_data' not found in {mat_path}")
            amp_data = np.array(mat_data["amplifier_data"])
            dig_data = np.array(mat_data["board_dig_in_data"])
            t_amp = np.array(mat_data["t_amplifier"]) if "t_amplifier" in mat_data else None
            t_dig = np.array(mat_data["t_dig"]) if "t_dig" in mat_data else None

    amplifier = orient_samples_channels(
        amp_data,
        expected_channels=AMPLIFIER_CHANNELS_EXPECTED,
        name="amplifier_data",
        mat_path=mat_path,
    )
    patch_info: dict[str, Any] = {"missing_labels": [], "fallback_end_padding": 0}
    if label_aware_padding and amplifier.shape[1] != AMPLIFIER_CHANNELS_EXPECTED:
        actual_labels = read_amplifier_channel_labels(mat_path)
        if actual_labels and len(actual_labels) == amplifier.shape[1]:
            amplifier, missing_labels = normalize_amplifier_by_labels(
                amplifier,
                expected_labels=EXPECTED_AMPLIFIER_LABELS,
                actual_labels=actual_labels,
                stream_name="amplifier_data",
                mat_path=mat_path,
                pad_value=0,
            )
            patch_info["missing_labels"] = missing_labels
        else:
            print(
                f"  [WARN] {mat_path.name}: could not read matching amplifier "
                f"channel labels (got {len(actual_labels)} labels for "
                f"{amplifier.shape[1]} columns); falling back to end-padding."
            )
            patch_info["fallback_end_padding"] = AMPLIFIER_CHANNELS_EXPECTED - amplifier.shape[1]
            amplifier = normalize_channel_count(
                amplifier,
                expected_channels=AMPLIFIER_CHANNELS_EXPECTED,
                stream_name="amplifier_data",
                mat_path=mat_path,
                pad_value=0,
            )
    else:
        amplifier = normalize_channel_count(
            amplifier,
            expected_channels=AMPLIFIER_CHANNELS_EXPECTED,
            stream_name="amplifier_data",
            mat_path=mat_path,
            pad_value=0,
        )
    dig = orient_samples_channels(
        dig_data,
        expected_channels=DIGITAL_IN_CHANNELS_EXPECTED,
        name="board_dig_in_data",
        mat_path=mat_path,
    )
    dig = normalize_channel_count(
        dig,
        expected_channels=DIGITAL_IN_CHANNELS_EXPECTED,
        stream_name="board_dig_in_data",
        mat_path=mat_path,
        pad_value=0,
    )

    t_amp_1d = flatten_time_vector(t_amp)
    t_dig_1d = flatten_time_vector(t_dig)
    dig_aligned = align_digital_to_amplifier(
        amplifier=amplifier,
        dig=dig,
        t_amplifier=t_amp_1d,
        t_dig=t_dig_1d,
        mat_path=mat_path,
    )

    if amplifier.shape[0] != dig_aligned.shape[0]:
        n = min(amplifier.shape[0], dig_aligned.shape[0])
        amplifier = amplifier[:n, :]
        dig_aligned = dig_aligned[:n, :]

    if amplifier.shape[1] != AMPLIFIER_CHANNELS_EXPECTED:
        raise RuntimeError(
            f"Internal error: amplifier normalization failed for {mat_path}: "
            f"{amplifier.shape[1]} != {AMPLIFIER_CHANNELS_EXPECTED}"
        )
    if dig_aligned.shape[1] != DIGITAL_IN_CHANNELS_EXPECTED:
        raise RuntimeError(
            f"Internal error: board_dig_in normalization failed for {mat_path}: "
            f"{dig_aligned.shape[1]} != {DIGITAL_IN_CHANNELS_EXPECTED}"
        )

    # Keep digital stream as strict binary states.
    dig_bin = (np.asarray(dig_aligned) > 0).astype(np.int16, copy=False)
    return amplifier.astype(np.float32, copy=False), dig_bin, patch_info


def write_channel_layout_metadata(output_path: Path, total_channels: int) -> None:
    meta_path = Path(f"{output_path}.meta.txt")
    lines = [
        f"output_file: {output_path}",
        "sample_layout: [amplifier_data channels..., board_dig_in_data channels...]",
        f"amplifier_channels: {AMPLIFIER_CHANNELS_EXPECTED}",
        f"board_dig_in_channels: {DIGITAL_IN_CHANNELS_EXPECTED}",
        f"total_channels: {total_channels}",
        "channel_indexing: 1-based",
        (
            "board_dig_in_channel_range: "
            f"{AMPLIFIER_CHANNELS_EXPECTED + 1}-{AMPLIFIER_CHANNELS_EXPECTED + DIGITAL_IN_CHANNELS_EXPECTED}"
        ),
    ]
    with open_with_retry(meta_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def merge_folder(input_dir: Path, output_path: Path, label_aware_padding: bool = False):
    mat_files = sorted(input_dir.rglob(MAT_FILE_GLOB), key=sort_key)
    if not mat_files:
        print(f"[SKIP] No '{MAT_FILE_GLOB}' files found in {input_dir}")
        return

    print(f"[START] Merging {len(mat_files)} files from {input_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_total_channels = AMPLIFIER_CHANNELS_EXPECTED + DIGITAL_IN_CHANNELS_EXPECTED
    written_chunks = 0
    patched_files: list[tuple[str, list[str]]] = []
    if output_path.exists():
        print(f"[INFO] Overwriting existing output: {output_path}")
    touch_truncate_with_retry(output_path)
    for idx, mat_path in enumerate(mat_files, start=1):
        try:
            amplifier_uv, board_dig, patch_info = load_streams(mat_path, label_aware_padding=label_aware_padding)
        except Exception as exc:
            print(
                f"  [WARN] Skipping unreadable file {mat_path.name}: {type(exc).__name__}: {exc}"
            )
            continue
        if patch_info.get("missing_labels"):
            patched_files.append((mat_path.name, list(patch_info["missing_labels"])))
        if patch_info.get("fallback_end_padding"):
            patched_files.append(
                (mat_path.name, [f"<unknown:{patch_info['fallback_end_padding']} end-padded>"])
            )
        amplifier_int16 = np.round(amplifier_uv / GAIN_TO_UV).astype(np.int16, copy=False)
        merged = np.concatenate([amplifier_int16, board_dig], axis=1)
        if merged.shape[1] != expected_total_channels:
            print(
                f"  [WARN] Skipping {mat_path.name}: merged channel count "
                f"{merged.shape[1]} != expected {expected_total_channels}"
            )
            continue
        if merged.shape[0] == 0:
            print(f"  [WARN] Skipping {mat_path.name}: zero samples after alignment")
            continue

        try:
            append_array_with_retry(output_path, merged.astype(np.int16, copy=False), mat_path.name)
        except Exception as exc:
            print(
                f"  [WARN] Failed appending {mat_path.name}: {type(exc).__name__}: {exc}"
            )
            continue

        written_chunks += 1
        print(
            f"  [{idx:03d}/{len(mat_files):03d}] {mat_path.name} "
            f"shape_amp={amplifier_uv.shape} shape_dig={board_dig.shape} "
            f"shape_merged={merged.shape}"
        )

    size_gb = safe_stat_size(output_path) / 1e9
    write_channel_layout_metadata(output_path=output_path, total_channels=expected_total_channels)
    if patched_files:
        print(
            f"[SUMMARY] {len(patched_files)} file(s) had missing amplifier "
            f"channel(s) patched (label_aware_padding={label_aware_padding}):"
        )
        for fname, labels in patched_files:
            print(f"  - {fname}: patched {labels}")
    else:
        print(
            f"[SUMMARY] No missing-channel patches applied "
            f"(label_aware_padding={label_aware_padding})."
        )
    print(
        f"[DONE] {output_path} ({size_gb:.2f} GB), chunks_written={written_chunks}, "
        f"channels_per_sample={expected_total_channels}"
    )


def main():
    for job in MERGE_JOBS:
        input_dir = Path(job["input_dir"])
        output_path = Path(job["output_path"])
        if not input_dir.exists():
            print(f"[SKIP] Input folder does not exist: {input_dir}")
            continue
        label_aware_padding = job.get("subject") in LABEL_AWARE_PADDING_SUBJECTS
        try:
            merge_folder(
                input_dir=input_dir,
                output_path=output_path,
                label_aware_padding=label_aware_padding,
            )
        except Exception as exc:
            print(
                f"[ERROR] Merge failed for {input_dir} -> {output_path}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue


if __name__ == "__main__":
    main()