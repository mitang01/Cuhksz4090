#!/usr/bin/env python3
"""
Inspect MAT files and report channel names/counts.

Default root:
  /share/workspace3/ieeg/micro/word_boun_perce_v1/bistable_sub4

The default root contains three subfolders (1, 2, 3), each holding two .mat files.

For each .mat file found (recursive), this script reports:
  - file name
  - amplifier_data shape and inferred sample/channel counts
  - board_dig_in_data shape and inferred sample/channel counts
  - amplifier channel names (if available)
  - board_dig_in channel names (if available)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

try:
    from scipy.io import loadmat
except ModuleNotFoundError:
    loadmat = None


DEFAULT_ROOT = Path("/share/workspace3/ieeg/micro/word_boun_perce_v1/bistable_sub4")


def infer_samples_channels(shape: tuple[int, ...]) -> tuple[int | None, int | None, str]:
    if len(shape) != 2:
        return None, None, "non-2D"
    a, b = int(shape[0]), int(shape[1])
    # Typical iEEG exports: samples >> channels.
    if a >= b:
        return a, b, "(samples, channels)"
    return b, a, "(channels, samples) -> interpreted as transposed"


def decode_hdf5_name_dataset(f: "h5py.File", ref: Any) -> str:
    try:
        ds = f[ref]
        arr = np.array(ds)
    except Exception:
        return ""

    if arr.size == 0:
        return ""
    if arr.dtype.kind in {"U", "S"}:
        s = "".join(str(x) for x in arr.ravel())
        return s.strip()

    flat = np.asarray(arr).ravel()
    if np.issubdtype(flat.dtype, np.integer):
        chars: list[str] = []
        for c in flat:
            ci = int(c)
            if ci == 0:
                continue
            if 0 < ci < 256:
                chars.append(chr(ci))
        return "".join(chars).strip()
    return ""


def decode_hdf5_channel_names(f: "h5py.File", group_key: str) -> list[str]:
    if group_key not in f:
        return []
    g = f[group_key]
    for key in ("native_channel_name", "custom_channel_name"):
        if key not in g:
            continue
        refs = np.array(g[key], dtype=object).ravel()
        out: list[str] = []
        for i, ref in enumerate(refs, start=1):
            name = decode_hdf5_name_dataset(f, ref)
            out.append(name if name else f"{group_key}_{i}")
        return out
    return []


def _extract_legacy_name(one_channel_obj: Any, idx: int, group_key: str) -> str:
    # scipy.io.loadmat can return mat_struct-like objects or ndarray records.
    for field in ("native_channel_name", "custom_channel_name"):
        if hasattr(one_channel_obj, field):
            value = getattr(one_channel_obj, field)
            if isinstance(value, np.ndarray):
                if value.dtype.kind in {"U", "S"}:
                    s = "".join(str(x) for x in value.ravel())
                    return s.strip() or f"{group_key}_{idx}"
                if np.issubdtype(value.dtype, np.integer):
                    return "".join(chr(int(x)) for x in value.ravel() if int(x) != 0).strip() or f"{group_key}_{idx}"
            if isinstance(value, str):
                return value.strip() or f"{group_key}_{idx}"
    if isinstance(one_channel_obj, np.void) and one_channel_obj.dtype.names:
        for field in ("native_channel_name", "custom_channel_name"):
            if field in one_channel_obj.dtype.names:
                v = one_channel_obj[field]
                arr = np.asarray(v).ravel()
                if arr.dtype.kind in {"U", "S"}:
                    return "".join(str(x) for x in arr).strip() or f"{group_key}_{idx}"
                if np.issubdtype(arr.dtype, np.integer):
                    return "".join(chr(int(x)) for x in arr if int(x) != 0).strip() or f"{group_key}_{idx}"
    return f"{group_key}_{idx}"


def decode_legacy_channel_names(mat_data: dict[str, Any], group_key: str) -> list[str]:
    if group_key not in mat_data:
        return []
    arr = np.asarray(mat_data[group_key]).ravel()
    out: list[str] = []
    for i, obj in enumerate(arr, start=1):
        out.append(_extract_legacy_name(obj, i, group_key))
    return out


def read_hdf5_info(path: Path) -> dict[str, Any]:
    if h5py is None:
        raise ModuleNotFoundError("h5py is required for HDF5 MAT files")
    info: dict[str, Any] = {"kind": "hdf5"}
    with h5py.File(path, "r") as f:
        amp_shape = tuple(int(x) for x in f["amplifier_data"].shape) if "amplifier_data" in f else None
        dig_shape = tuple(int(x) for x in f["board_dig_in_data"].shape) if "board_dig_in_data" in f else None
        info["amp_shape"] = amp_shape
        info["dig_shape"] = dig_shape
        info["amp_names"] = decode_hdf5_channel_names(f, "amplifier_channels")
        info["dig_names"] = decode_hdf5_channel_names(f, "board_dig_in_channels")
    return info


def read_legacy_info(path: Path) -> dict[str, Any]:
    if loadmat is None:
        raise ModuleNotFoundError("scipy is required for non-HDF5 MAT files")
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    info: dict[str, Any] = {"kind": "legacy-mat"}
    amp_shape = tuple(int(x) for x in np.asarray(mat["amplifier_data"]).shape) if "amplifier_data" in mat else None
    dig_shape = tuple(int(x) for x in np.asarray(mat["board_dig_in_data"]).shape) if "board_dig_in_data" in mat else None
    info["amp_shape"] = amp_shape
    info["dig_shape"] = dig_shape
    info["amp_names"] = decode_legacy_channel_names(mat, "amplifier_channels")
    info["dig_names"] = decode_legacy_channel_names(mat, "board_dig_in_channels")
    return info


def print_names(title: str, names: list[str]) -> None:
    print(f"  {title}: {len(names)}")
    if not names:
        print("    (none found)")
        return
    for i, name in enumerate(names, start=1):
        print(f"    {i:>3}: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="List MAT file channel names and counts.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root folder containing raw MAT files (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.mat",
        help="Recursive glob pattern under root (default: *.mat)",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"Error: root folder not found or not a directory: {root}", file=sys.stderr)
        return 2

    mat_files = sorted(root.rglob(args.glob))
    if not mat_files:
        print(f"No MAT files found under: {root}")
        return 0

    print(f"Root: {root}")
    print(f"Found {len(mat_files)} MAT file(s)")
    print()

    for p in mat_files:
        print("=" * 100)
        print(f"File: {p.name}")
        print(f"Path: {p}")
        try:
            is_hdf5 = h5py is not None and h5py.is_hdf5(str(p))
            info = read_hdf5_info(p) if is_hdf5 else read_legacy_info(p)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] Could not read: {type(exc).__name__}: {exc}")
            continue

        amp_shape = info.get("amp_shape")
        dig_shape = info.get("dig_shape")

        if amp_shape is not None:
            amp_samples, amp_channels, amp_note = infer_samples_channels(amp_shape)
            print(
                f"  amplifier_data shape: {amp_shape} -> "
                f"samples={amp_samples}, channels={amp_channels} {amp_note}"
            )
        else:
            print("  amplifier_data: not found")

        if dig_shape is not None:
            dig_samples, dig_channels, dig_note = infer_samples_channels(dig_shape)
            print(
                f"  board_dig_in_data shape: {dig_shape} -> "
                f"samples={dig_samples}, channels={dig_channels} {dig_note}"
            )
        else:
            print("  board_dig_in_data: not found")

        print_names("amplifier channel names", info.get("amp_names", []))
        print_names("board_dig_in channel names", info.get("dig_names", []))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
