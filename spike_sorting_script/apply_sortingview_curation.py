#!/usr/bin/env python
"""
Apply sortingview curation snapshots (sha1://...) to spike sorting outputs.

Supports:
1) Single target: --session --region --uri
2) Batch targets: --curation-file JSON
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import spikeinterface.curation as scur
import spikeinterface.full as si


DEFAULT_OUTPUT_ROOT = Path("/share/home/mitan/spike_sorting")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply sortingview curation snapshot(s) to sorting outputs."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory containing sorting_results_session* folders.",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session name, e.g. session2 (used with --region and --uri).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region name, e.g. HG (used with --session and --uri).",
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=None,
        help="sortingview snapshot URI, e.g. sha1://...",
    )
    parser.add_argument(
        "--curation-file",
        type=Path,
        default=None,
        help=(
            "JSON file for batch mode. Format: "
            '[{"session":"session2","region":"HG","uri":"sha1://..."}, ...]'
        ),
    )
    parser.add_argument(
        "--save-curated-sorting",
        action="store_true",
        help="Try to save curated sorting extractor to disk (best effort).",
    )
    return parser.parse_args()


def build_targets(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.curation_file is not None:
        payload = json.loads(args.curation_file.read_text())
        if not isinstance(payload, list):
            raise ValueError("--curation-file must contain a JSON list.")
        targets = []
        for i, item in enumerate(payload):
            if not all(k in item for k in ("session", "region", "uri")):
                raise ValueError(f"Missing keys in curation-file item {i}: {item}")
            targets.append(
                {
                    "session": str(item["session"]),
                    "region": str(item["region"]),
                    "uri": str(item["uri"]),
                }
            )
        return targets

    if args.session and args.region and args.uri:
        return [{"session": args.session, "region": args.region, "uri": args.uri}]

    raise ValueError(
        "Provide either (--session --region --uri) for single mode, "
        "or --curation-file for batch mode."
    )


def get_optional_accept(curated_sorting) -> Optional[np.ndarray]:
    try:
        accept = curated_sorting.get_property("accept")
        if accept is None:
            return None
        return np.asarray(accept)
    except Exception:
        return None


def apply_one_target(
    output_root: Path,
    session: str,
    region: str,
    uri: str,
    save_curated_sorting: bool,
) -> Dict[str, str]:
    session_root = output_root / f"sorting_results_{session}"
    sorting_folder = session_root / region / "sorting"
    out_dir = session_root / region / "curation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sorting_folder.exists():
        raise FileNotFoundError(f"sorting folder not found: {sorting_folder}")

    sorting = si.read_sorter_folder(str(sorting_folder))
    curated = scur.apply_sortingview_curation(sorting, uri_or_json=uri)
    unit_ids = list(curated.get_unit_ids())

    # Save curated unit list and accept flags.
    accept_flags = get_optional_accept(curated)
    accept_csv = out_dir / "curated_accept_flags.csv"
    if accept_flags is not None and len(accept_flags) == len(unit_ids):
        pd.DataFrame({"unit_id": unit_ids, "accept": accept_flags}).to_csv(
            accept_csv, index=False
        )
    else:
        pd.DataFrame({"unit_id": unit_ids}).to_csv(accept_csv, index=False)

    # Save curated spike trains (seconds) into npz for downstream analysis.
    spike_map = {}
    for unit_id in unit_ids:
        st = curated.get_unit_spike_train(unit_id=unit_id, segment_index=0)
        spike_map[f"unit_{unit_id}"] = np.asarray(st, dtype=np.int64)
    spikes_npz = out_dir / "curated_spike_trains_samples.npz"
    np.savez_compressed(spikes_npz, **spike_map)

    curated_sorting_dir = None
    if save_curated_sorting:
        # Best effort: API may vary across spikeinterface versions.
        try:
            curated_sorting_dir = out_dir / "sorting_curated"
            curated.save(folder=str(curated_sorting_dir), overwrite=True)
        except Exception:
            curated_sorting_dir = None

    meta = {
        "applied_at": datetime.now().isoformat(timespec="seconds"),
        "session": session,
        "region": region,
        "sorting_folder": str(sorting_folder),
        "uri": uri,
        "num_units_curated": len(unit_ids),
        "accept_csv": str(accept_csv),
        "spikes_npz": str(spikes_npz),
        "curated_sorting_dir": str(curated_sorting_dir) if curated_sorting_dir else None,
    }
    meta_json = out_dir / "curation_applied.json"
    meta_json.write_text(json.dumps(meta, indent=2))

    return {
        "session": session,
        "region": region,
        "uri": uri,
        "status": "ok",
        "meta_json": str(meta_json),
    }


def main() -> None:
    args = parse_args()
    targets = build_targets(args)
    output_root = args.output_root

    results = []
    for t in targets:
        session = t["session"]
        region = t["region"]
        uri = t["uri"]
        print(f"[START] Applying curation: {session}/{region} -> {uri}")
        try:
            result = apply_one_target(
                output_root=output_root,
                session=session,
                region=region,
                uri=uri,
                save_curated_sorting=args.save_curated_sorting,
            )
            print(f"[OK] {session}/{region}: {result['meta_json']}")
            results.append(result)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERROR] {session}/{region}: {exc}")
            results.append(
                {
                    "session": session,
                    "region": region,
                    "uri": uri,
                    "status": "error",
                    "error": repr(exc),
                }
            )

    run_log = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "results": results,
    }
    log_path = output_root / f"curation_apply_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(run_log, indent=2))
    print(f"[DONE] Run log: {log_path}")


if __name__ == "__main__":
    main()
