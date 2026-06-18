# This script is for merging the raw mat files into binary files. the output file doesn't end with .MAT 


from pathlib import Path
import re

import h5py
import numpy as np
from scipy.io import loadmat


MAT_FILE_GLOB = "Temp_26012*.mat"
MERGE_JOBS = [
    {
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5/Temp_260121_095012"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub5_session1"),
    },
    {
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5/Temp_260121_103639"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub5_session2"),
    },
    {
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5/Temp_260121_104824"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub5_session3"),
    },
    {
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5/Temp_260121_105933"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub5_session4"),
    },
    {
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5/Temp_260121_125018"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub5_session5"),
    },
    {
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0121/bistable_sub5/Temp_260121_130023"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub5_session6"),
    },
    {
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0123/bistable_sub6_1/Temp_260123_115236"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub6_session1"),
    },
    {
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0123/bistable_sub6_2/Temp_260123_120327"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub6_session3"),
    },
    {
        "input_dir": Path("/share/workspace2/tangmi/20260120-20260123/0123/bistable_sub6_3/Temp_260123_121427"),
        "output_path": Path("/share/workspace2/tangmi/bistable_sub6_session6"),
    },
]
GAIN_TO_UV = 0.195  # Intan RHD2000 conversion factor


def sort_key(path: Path):
    """
    Sort by all digit groups in filename first, then by name.
    This keeps files in a stable temporal-like order for common naming patterns.
    """
    digits = re.findall(r"\d+", path.stem)
    if digits:
        return tuple(int(d) for d in digits), path.name
    return (path.name,)


def load_amplifier_data(mat_path: Path) -> np.ndarray:
    """
    Load 'amplifier_data' from either HDF5-style .mat (v7.3) or old MATLAB format.
    Returns float array with shape (samples, channels).
    """
    # v7.3 MAT files are HDF5; older MAT files are not.
    # Detect format first so we do not incorrectly fall back between readers.
    if h5py.is_hdf5(str(mat_path)):
        try:
            with h5py.File(mat_path, "r") as h5f:
                if "amplifier_data" not in h5f:
                    raise KeyError("'amplifier_data' not found in file")
                data = np.array(h5f["amplifier_data"])
        except OSError as exc:
            raise RuntimeError(f"Failed to read HDF5 MAT file: {mat_path}") from exc
    else:
        try:
            mat_data = loadmat(mat_path)
        except NotImplementedError as exc:
            # Safety fallback in case a v7.3 file is misdetected.
            with h5py.File(mat_path, "r") as h5f:
                if "amplifier_data" not in h5f:
                    raise KeyError("'amplifier_data' not found in file")
                data = np.array(h5f["amplifier_data"])
        else:
            if "amplifier_data" not in mat_data:
                raise KeyError(f"'amplifier_data' not found in {mat_path}")
            data = np.array(mat_data["amplifier_data"])

    # Ensure data layout is always (samples, channels)
    if data.ndim != 2:
        raise ValueError(f"Unexpected amplifier_data shape {data.shape} in {mat_path}")
    if data.shape[0] == 128 and data.shape[1] != 128:
        data = data.T
    return data


def merge_folder(input_dir: Path, output_path: Path):
    mat_files = sorted(input_dir.rglob(MAT_FILE_GLOB), key=sort_key)
    if not mat_files:
        print(f"[SKIP] No '{MAT_FILE_GLOB}' files found in {input_dir}")
        return

    print(f"[START] Merging {len(mat_files)} files from {input_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as out_f:
        for idx, mat_path in enumerate(mat_files, start=1):
            try:
                data_uv = load_amplifier_data(mat_path)
            except Exception as exc:
                print(
                    f"  [WARN] Skipping unreadable file {mat_path.name}: {type(exc).__name__}: {exc}"
                )
                continue
            data_int16 = np.round(data_uv / GAIN_TO_UV).astype(np.int16, copy=False)
            data_int16.tofile(out_f)
            print(
                f"  [{idx:03d}/{len(mat_files):03d}] {mat_path.name} "
                f"shape={data_uv.shape}"
            )

    size_gb = output_path.stat().st_size / 1e9
    print(f"[DONE] {output_path} ({size_gb:.2f} GB)")


def main():
    for job in MERGE_JOBS:
        input_dir = Path(job["input_dir"])
        output_path = Path(job["output_path"])
        if not input_dir.exists():
            print(f"[SKIP] Input folder does not exist: {input_dir}")
            continue
        merge_folder(input_dir=input_dir, output_path=output_path)


if __name__ == "__main__":
    main()