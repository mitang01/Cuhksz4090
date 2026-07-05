#!/usr/bin/env python
"""
Generate sortingview/figurl links for web-based spike sorting inspection and curation.

This script is intended for non-graphical cluster jobs (e.g., sbatch).
It does not run sorting again; it only loads existing SortingAnalyzer folders.
"""

from __future__ import annotations

import argparse
import json
import re
import importlib
import inspect
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import spikeinterface.full as si
import spikeinterface.widgets as sw


DEFAULT_OUTPUT_ROOT = Path("/share/home/mitan/spike_sorting")
DEFAULT_ANALYZER_ROOT = Path("/share/home/mitan/spike_sorting/mountainsort4")
DEFAULT_MANIFEST_PATH = Path(
    "/share/home/mitan/spike_sorting/"
    "sortingview_links_mountainsort4_rawmat_sub5_6.json"
)
DEFAULT_REGIONS = ["ATL", "HG", "VMPFC", "Amygdala"]


def ensure_figurl_backend_dependencies() -> None:
    missing = []
    for module_name in ("figpack", "sortingview", "figpack_spike_sorting"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if missing:
        missing_str = ", ".join(missing)
        raise SystemExit(
            "[FATAL] Missing required package(s): "
            f"{missing_str}. Install in your active env with:\n"
            "  pip install figpack figpack-spike-sorting sortingview kachery-cloud"
        )


def patch_sortingcuration_api_if_needed() -> None:
    """
    Compatibility shim:
    Older figpack-spike-sorting uses 'default_label_options' while newer
    spikeinterface passes 'label_choices'. We alias transparently.
    """
    try:
        from figpack_spike_sorting.views import SortingCuration
    except Exception:
        return

    sig = inspect.signature(SortingCuration.__init__)
    params = sig.parameters

    if "label_choices" in params:
        return
    if "default_label_options" not in params:
        return

    original_init = SortingCuration.__init__

    def _patched_init(self, *args, **kwargs):
        if "label_choices" in kwargs and "default_label_options" not in kwargs:
            kwargs["default_label_options"] = kwargs.pop("label_choices")
        else:
            kwargs.pop("label_choices", None)
        return original_init(self, *args, **kwargs)

    SortingCuration.__init__ = _patched_init
    print(
        "[INFO] Applied compatibility shim for figpack_spike_sorting.SortingCuration "
        "(label_choices -> default_label_options)."
    )

    # spikeinterface 0.104.0 may access vv_views.LayoutItem, while current
    # figpack_spike_sorting exposes LayoutItem in figpack.views instead.
    try:
        import figpack.views as figpack_views
        import figpack_spike_sorting.views as spike_views

        if not hasattr(spike_views, "LayoutItem"):
            spike_views.LayoutItem = figpack_views.LayoutItem
            print(
                "[INFO] Applied compatibility shim for figpack_spike_sorting.LayoutItem "
                "(aliased from figpack.views.LayoutItem)."
            )
    except Exception:
        pass

    # Some versions emit numpy scalar values into JSON payloads without conversion.
    # Patch figpack_spike_sorting UnitsTable JSON serializer to handle numpy types.
    try:
        import json as _json
        import numpy as _np

        units_table_mod = importlib.import_module("figpack_spike_sorting.views.UnitsTable")
        original_dumps = _json.dumps

        def _dumps_with_numpy(obj, *args, **kwargs):
            def _default(value):
                if isinstance(value, _np.integer):
                    return int(value)
                if isinstance(value, _np.floating):
                    return float(value)
                if isinstance(value, _np.ndarray):
                    return value.tolist()
                raise TypeError(
                    f"Object of type {type(value).__name__} is not JSON serializable"
                )

            if "default" not in kwargs:
                kwargs["default"] = _default
            return original_dumps(obj, *args, **kwargs)

        units_table_mod.json.dumps = _dumps_with_numpy
        print(
            "[INFO] Applied compatibility shim for figpack_spike_sorting UnitsTable "
            "JSON numpy serialization."
        )
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sortingview links from existing analyzer outputs."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root metadata field (default: /share/home/mitan/spike_sorting).",
    )
    parser.add_argument(
        "--analyzer-root",
        type=Path,
        default=DEFAULT_ANALYZER_ROOT,
        help=(
            "Root containing sorting result folders (sorting_results_*_v2_rawmat), e.g. "
            "/share/home/mitan/spike_sorting/mountainsort4."
        ),
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help=(
            "Sessions to process, e.g. sub5_Temp_260121_095012_Temp_260121_095012. "
            "Default: auto-discover all sorting_results_*_v2_rawmat folders under --analyzer-root."
        ),
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        default=DEFAULT_REGIONS,
        help="Regions to process. Default: ATL HG VMPFC Amygdala",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=(
            "Path to write JSON manifest. "
            "Default: /share/home/mitan/spike_sorting/"
            "sortingview_links_mountainsort4_rawmat_sub5_6.json"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["local", "upload"],
        default="local",
        help=(
            "URL generation mode. "
            "'local' does not require FIGPACK_API_KEY and is suitable for SSH tunneling. "
            "'upload' requires FIGPACK_API_KEY and returns shareable links."
        ),
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=50005,
        help="Base port for local mode; each session/region pair uses an incremented port.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=0,
        help=(
            "In local mode, keep process alive for this many seconds after URL generation "
            "(useful for manual curation via SSH tunnel)."
        ),
    )
    parser.add_argument(
        "--add-upload-urls",
        action="store_true",
        help=(
            "Keep original URL fields based on --mode, and additionally generate "
            "FIGPACK uploaded shareable URLs in new manifest fields. "
            "Requires FIGPACK_API_KEY."
        ),
    )
    return parser.parse_args()


def extract_url(widget_obj) -> Optional[str]:
    # Common attributes used by sortingview widget wrappers.
    for attr in ("url", "view_url", "figure_url"):
        val = getattr(widget_obj, attr, None)
        if isinstance(val, str) and val.startswith("http"):
            return val

    # Fallback: parse URL from string representation.
    txt = str(widget_obj)
    m = re.search(r"https?://\S+", txt)
    if m:
        return m.group(0).rstrip("',\")")
    return None


def discover_sorting_result_sessions(analyzer_root: Path) -> Dict[str, Path]:
    """
    Auto-discover sorting result folders under analyzer_root.

    Matches folders named like 'sorting_results_<dataset>_v2_rawmat' (e.g.
    sorting_results_sub5_Temp_260121_095012_Temp_260121_095012_v2_rawmat)
    and keys them by the dataset name with prefix/suffix removed.
    Legacy '_v2' folders are also supported.
    """
    sessions: Dict[str, Path] = {}
    for p in sorted(analyzer_root.glob("sorting_results_*")):
        if not p.is_dir():
            continue
        m = re.match(r"^sorting_results_(.+?)(?:_v2_rawmat|_v2)?$", p.name)
        if not m:
            continue
        session = m.group(1)
        sessions[session] = p
    return sessions


def resolve_session_root(analyzer_root: Path, session: str) -> Path:
    """
    Resolve a session token to its sorting-results folder.

    Accepts bare session token or folder-like variants with:
      - sorting_results_<session>_v2_rawmat
      - sorting_results_<session>_v2
      - sorting_results_<session>
      - full folder name
    """
    candidates = [
        analyzer_root / session,
        analyzer_root / f"sorting_results_{session}_v2_rawmat",
        analyzer_root / f"sorting_results_{session}",
        analyzer_root / f"sorting_results_{session}_v2",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        f"Session folder not found for '{session}' under {analyzer_root}. "
        "Expected e.g. sorting_results_<session>_v2_rawmat."
    )


def resolve_analyzer_path(session_root: Path, region: str) -> Path:
    """
    Resolve a region's SortingAnalyzer folder within a session root.

    Supports both layouts:
      - '<session_root>/<region>/analyzer' (current batch v2 layout)
      - '<session_root>/<region>_analyzer' (older story-listen layout)
    Returns the first existing candidate; if none exist, returns the preferred
    (batch v2) candidate so the caller's existence check produces a clear skip.
    """
    preferred = session_root / region / "analyzer"
    if preferred.exists():
        return preferred
    legacy = session_root / f"{region}_analyzer"
    if legacy.exists():
        return legacy
    return preferred


def generate_links_for_analyzer(
    analyzer_path: Path,
    mode: str,
    local_port: Optional[int] = None,
    add_upload_urls: bool = False,
) -> Dict[str, Optional[str]]:
    analyzer = si.load_sorting_analyzer(str(analyzer_path))

    # Use the modern backend that powers shareable figurl links.
    w_qm = sw.plot_quality_metrics(analyzer, display=False, backend="figpack")
    w_summary = sw.plot_sorting_summary(
        analyzer, display=False, curation=True, backend="figpack"
    )

    qm_url = None
    summary_url = None
    upload_qm_url = None
    upload_summary_url = None
    if hasattr(w_qm, "view") and hasattr(w_qm.view, "show"):
        if mode == "upload":
            qm_url = w_qm.view.show(
                title=f"quality_metrics::{analyzer_path}",
                open_in_browser=False,
                upload=True,
                wait_for_input=False,
            )
        else:
            qm_url = w_qm.view.show(
                title=f"quality_metrics::{analyzer_path}",
                open_in_browser=False,
                upload=False,
                wait_for_input=False,
                port=local_port,
            )
    if hasattr(w_summary, "view") and hasattr(w_summary.view, "show"):
        if mode == "upload":
            summary_url = w_summary.view.show(
                title=f"sorting_summary::{analyzer_path}",
                open_in_browser=False,
                upload=True,
                wait_for_input=False,
            )
        else:
            summary_url = w_summary.view.show(
                title=f"sorting_summary::{analyzer_path}",
                open_in_browser=False,
                upload=False,
                wait_for_input=False,
                port=local_port,
            )

    # Optional second set of links: always uploaded/shareable URLs.
    if add_upload_urls:
        if not os.environ.get("FIGPACK_API_KEY"):
            raise RuntimeError(
                "--add-upload-urls requested but FIGPACK_API_KEY is not set."
            )
        if hasattr(w_qm, "view") and hasattr(w_qm.view, "show"):
            upload_qm_url = w_qm.view.show(
                title=f"quality_metrics_upload::{analyzer_path}",
                open_in_browser=False,
                upload=True,
                wait_for_input=False,
            )
        if hasattr(w_summary, "view") and hasattr(w_summary.view, "show"):
            upload_summary_url = w_summary.view.show(
                title=f"sorting_summary_upload::{analyzer_path}",
                open_in_browser=False,
                upload=True,
                wait_for_input=False,
            )

    return {
        "quality_metrics_url": qm_url or extract_url(w_qm),
        "sorting_summary_curation_url": summary_url or extract_url(w_summary),
        "figpack_quality_metrics_url": upload_qm_url,
        "figpack_sorting_summary_curation_url": upload_summary_url,
    }


def main() -> None:
    ensure_figurl_backend_dependencies()
    patch_sortingcuration_api_if_needed()

    args = parse_args()
    output_root = args.output_root
    analyzer_root = args.analyzer_root
    discovered = discover_sorting_result_sessions(analyzer_root)
    sessions = args.sessions if args.sessions else sorted(discovered.keys())
    regions = args.regions
    mode = args.mode

    if not sessions:
        raise SystemExit(
            f"No sessions found under {analyzer_root}. "
            "Expected folders like sorting_results_<session>_v2_rawmat."
        )

    manifest_path = args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "analyzer_root": str(analyzer_root),
        "mode": mode,
        "host": os.uname().nodename,
        "items": [],
    }

    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Analyzer root: {analyzer_root}")
    print(f"[INFO] Sessions: {sessions}")
    print(f"[INFO] Regions: {regions}")
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Add upload URLs: {args.add_upload_urls}")

    if mode == "upload" and not os.environ.get("FIGPACK_API_KEY"):
        raise SystemExit(
            "[FATAL] --mode upload requires FIGPACK_API_KEY. "
            "Use --mode local if API key is not available."
        )
    if args.add_upload_urls and not os.environ.get("FIGPACK_API_KEY"):
        raise SystemExit(
            "[FATAL] --add-upload-urls requires FIGPACK_API_KEY to be set."
        )

    port_counter = 0

    for session in sessions:
        if session in discovered:
            session_root = discovered[session]
        else:
            session_root = resolve_session_root(analyzer_root, session)
        for region in regions:
            analyzer_path = resolve_analyzer_path(session_root, region)
            local_port = args.base_port + port_counter if mode == "local" else None
            port_counter += 1
            item = {
                "session": session,
                "region": region,
                "analyzer_path": str(analyzer_path),
                "quality_metrics_url": None,
                "sorting_summary_curation_url": None,
                "figpack_quality_metrics_url": None,
                "figpack_sorting_summary_curation_url": None,
                "local_port": local_port,
                "ssh_tunnel_hint": None,
                "status": "skipped",
                "error": None,
            }

            if not analyzer_path.exists():
                item["error"] = "analyzer path does not exist"
                results["items"].append(item)
                print(f"[SKIP] {session}/{region}: analyzer not found")
                continue

            try:
                links = generate_links_for_analyzer(
                    analyzer_path=analyzer_path,
                    mode=mode,
                    local_port=local_port,
                    add_upload_urls=args.add_upload_urls,
                )
                item["quality_metrics_url"] = links["quality_metrics_url"]
                item["sorting_summary_curation_url"] = links[
                    "sorting_summary_curation_url"
                ]
                item["figpack_quality_metrics_url"] = links[
                    "figpack_quality_metrics_url"
                ]
                item["figpack_sorting_summary_curation_url"] = links[
                    "figpack_sorting_summary_curation_url"
                ]
                if mode == "local" and local_port is not None:
                    item["ssh_tunnel_hint"] = (
                        f"ssh -L {local_port}:{results['host']}:{local_port} <login-node>"
                    )
                if item["sorting_summary_curation_url"]:
                    item["status"] = "ok"
                    print(
                        f"[OK] {session}/{region} summary URL: {item['sorting_summary_curation_url']}"
                    )
                    if item["figpack_sorting_summary_curation_url"]:
                        print(
                            f"[OK] {session}/{region} uploaded summary URL: "
                            f"{item['figpack_sorting_summary_curation_url']}"
                        )
                    if mode == "local":
                        print(
                            f"[HINT] Tunnel with: ssh -L {local_port}:{results['host']}:{local_port} <login-node>"
                        )
                else:
                    item["status"] = "warning"
                    print(f"[WARN] {session}/{region}: URL extraction failed")
            except Exception as exc:  # pylint: disable=broad-except
                item["status"] = "error"
                item["error"] = repr(exc)
                print(f"[ERROR] {session}/{region}: {exc}")
                if "FIGPACK_API_KEY" in str(exc):
                    print(
                        "[HINT] Set FIGPACK_API_KEY in your sbatch script environment "
                        "to allow upload=True shareable URL generation."
                    )

            results["items"].append(item)

    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"[DONE] Link manifest written: {manifest_path}")
    print("[INFO] Use sorting_summary_curation_url for manual merge/label/curation.")
    if mode == "local" and args.hold_seconds > 0:
        print(
            f"[INFO] Holding process for {args.hold_seconds} seconds so local URLs stay available..."
        )
        time.sleep(args.hold_seconds)
        print("[INFO] Hold finished. Exiting.")


if __name__ == "__main__":
    main()
