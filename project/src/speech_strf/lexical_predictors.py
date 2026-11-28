from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

from .alignments import Interval
from .timebase import nearest_frame


WORD_TIERS = ("word", "words")
WORD_TIMING_COLUMNS = ("word_onset", "word_boundary", "word_duration")
TOP_LEVEL_FAMILIES = ("acoustic", "prosodic", "phonetic", "word")


@dataclass(frozen=True)
class ContextWindow:
    window_index: int
    start_word: int
    end_word: int
    emit_start_word: int
    emit_end_word: int
    content_tokens: int


def ordered_word_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Return nonempty forced-aligned words in stable temporal order."""
    words = [
        row
        for row in intervals
        if row.tier.lower() in WORD_TIERS and row.label.strip()
    ]
    return sorted(words, key=lambda row: (row.start, row.end))


def plan_context_windows(
    piece_counts: Sequence[int],
    *,
    max_tokens: int,
    special_tokens: int,
    overlap_words: int,
) -> list[ContextWindow]:
    """Plan deterministic, overlapping word-boundary windows.

    A word is owned by the first window in which it appears. Repeated tail
    words provide left context to newly emitted words in the following
    window, while every occurrence is emitted exactly once.
    """
    if max_tokens <= special_tokens:
        raise ValueError("max_tokens must exceed the number of special tokens")
    if overlap_words < 0:
        raise ValueError("overlap_words must be nonnegative")
    counts = [int(value) for value in piece_counts]
    if any(value < 0 for value in counts):
        raise ValueError("piece counts must be nonnegative")
    budget = max_tokens - special_tokens
    oversized = [index for index, value in enumerate(counts) if value > budget]
    if oversized:
        raise ValueError(
            "Words exceed the tokenizer content budget at indexes "
            f"{oversized}; refusing to split a word occurrence"
        )
    if not counts:
        return []

    # Missing tokenizer mappings have count zero. Reserve one position for
    # planning so a long run of missing mappings cannot create an unbounded
    # window.
    planned = [max(1, value) for value in counts]
    windows: list[ContextWindow] = []
    previous_end = 0
    while previous_end < len(planned):
        if not windows:
            start = 0
        else:
            start = max(0, previous_end - overlap_words)
            # Reduce overlap if it would leave no room for the next new word.
            while start < previous_end and sum(
                planned[start : previous_end + 1]
            ) > budget:
                start += 1
        end = previous_end + 1
        used = sum(planned[start:end])
        if used > budget:
            raise ValueError(
                f"Word {previous_end} cannot fit after applying overlap"
            )
        while end < len(planned) and used + planned[end] <= budget:
            used += planned[end]
            end += 1
        emit_start = 0 if not windows else previous_end
        windows.append(
            ContextWindow(
                window_index=len(windows),
                start_word=start,
                end_word=end,
                emit_start_word=emit_start,
                emit_end_word=end,
                content_tokens=used,
            )
        )
        previous_end = end
    return windows


def _word_ids(encoding: Any) -> list[int | None]:
    if not hasattr(encoding, "word_ids"):
        raise TypeError(
            "Tokenizer output does not expose word_ids(); a fast tokenizer is required"
        )
    values = encoding.word_ids(batch_index=0)
    if values is None:
        raise TypeError("Fast tokenizer returned no presegmented word mapping")
    return list(values)


def _model_inputs(encoding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }


def infer_context_limit(tokenizer: Any, model: Any) -> int:
    candidates: list[int] = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and tokenizer_limit < 1_000_000:
        candidates.append(tokenizer_limit)
    config_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(config_limit, int):
        candidates.append(config_limit)
    if not candidates:
        raise ValueError("Cannot infer a finite tokenizer/model context limit")
    return min(candidates)


def extract_recording_embeddings(
    words: Sequence[Interval],
    tokenizer: Any,
    model: Any,
    *,
    overlap_words: int = 32,
    max_tokens: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Extract one final-layer vector per aligned word occurrence."""
    labels = [row.label.strip() for row in words]
    if any(not label for label in labels):
        raise ValueError("Embedding extraction received an empty word label")
    if not getattr(tokenizer, "is_fast", False):
        raise TypeError("A fast tokenizer is required for word-to-subword mapping")

    import torch

    model.eval()
    model.requires_grad_(False)
    model.to(device)
    checkpoint_limit = infer_context_limit(tokenizer, model)
    limit = checkpoint_limit if max_tokens is None else int(max_tokens)
    if not 1 <= limit <= checkpoint_limit:
        raise ValueError(f"max_tokens must be between 1 and {checkpoint_limit}")
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))

    full = tokenizer(
        labels,
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    full_word_ids = _word_ids(full)
    piece_counts = np.bincount(
        [value for value in full_word_ids if value is not None],
        minlength=len(labels),
    ).tolist() if any(value is not None for value in full_word_ids) else [0] * len(labels)
    windows = plan_context_windows(
        piece_counts,
        max_tokens=limit,
        special_tokens=special_tokens,
        overlap_words=overlap_words,
    )

    hidden_size = int(getattr(model.config, "hidden_size"))
    vectors = np.zeros((len(words), hidden_size), dtype=np.float32)
    mapped_piece_counts = np.zeros(len(words), dtype=np.int32)
    unknown_piece_counts = np.zeros(len(words), dtype=np.int32)
    owner_window = np.full(len(words), -1, dtype=np.int32)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    window_rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for window in windows:
            window_labels = labels[window.start_word : window.end_word]
            encoded = tokenizer(
                window_labels,
                is_split_into_words=True,
                add_special_tokens=True,
                truncation=False,
                return_tensors="pt",
            )
            ids = _word_ids(encoded)
            token_length = len(ids)
            if token_length > limit:
                raise ValueError(
                    f"Planned window {window.window_index} has {token_length} "
                    f"tokens, exceeding context limit {limit}"
                )
            inputs = {
                key: value.to(device)
                for key, value in _model_inputs(encoded).items()
            }
            hidden = model(**inputs).last_hidden_state[0].detach().cpu()
            input_ids = encoded["input_ids"][0].detach().cpu().numpy()
            positions_by_word: list[list[int]] = [[] for _ in window_labels]
            for position, word_id in enumerate(ids):
                if word_id is not None:
                    positions_by_word[word_id].append(position)

            for global_index in range(
                window.emit_start_word, window.emit_end_word
            ):
                local_index = global_index - window.start_word
                positions = positions_by_word[local_index]
                owner_window[global_index] = window.window_index
                mapped_piece_counts[global_index] = len(positions)
                if not positions:
                    continue
                vectors[global_index] = (
                    hidden[positions].mean(dim=0).numpy().astype(np.float32)
                )
                if unk_id is not None:
                    unknown_piece_counts[global_index] = int(
                        np.count_nonzero(input_ids[positions] == unk_id)
                    )

            window_rows.append(
                {
                    **asdict(window),
                    "token_count_with_special_tokens": token_length,
                    "start_seconds": float(words[window.start_word].start),
                    "end_seconds": float(words[window.end_word - 1].end),
                }
            )

    emitted = owner_window >= 0
    if not np.all(emitted):
        raise RuntimeError(
            f"Context plan failed to emit word indexes {np.flatnonzero(~emitted).tolist()}"
        )
    missing = np.flatnonzero(mapped_piece_counts == 0).astype(int).tolist()
    return {
        "vectors": vectors,
        "word_index": np.arange(len(words), dtype=np.int32),
        "words": np.asarray(labels),
        "start_seconds": np.asarray([row.start for row in words], dtype=np.float64),
        "end_seconds": np.asarray([row.end for row in words], dtype=np.float64),
        "piece_counts": mapped_piece_counts,
        "unknown_piece_counts": unknown_piece_counts,
        "owner_window": owner_window,
        "windows": window_rows,
        "missing_word_indexes": missing,
        "word_count": len(words),
        "mapped_word_count": len(words) - len(missing),
        "unknown_word_count": int(np.count_nonzero(unknown_piece_counts)),
        "context_limit": limit,
        "overlap_words": overlap_words,
    }


def resolved_huggingface_revision(tokenizer: Any, model: Any) -> str:
    values = [
        getattr(getattr(model, "config", None), "_commit_hash", None),
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
    ]
    revisions = {str(value) for value in values if value}
    if not revisions:
        raise ValueError(
            "Hugging Face did not expose a resolved immutable revision; "
            "refusing to write unpinned embeddings"
        )
    if len(revisions) != 1:
        raise ValueError(f"Tokenizer/model resolved revisions disagree: {sorted(revisions)}")
    return revisions.pop()


@lru_cache(maxsize=256)
def _sha256_file_cached(path: str, file_identity: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return _sha256_file_cached(str(resolved), identity)


def write_npz_atomic(
    destination: str | Path,
    arrays: Mapping[str, Any],
    *,
    compressed: bool = False,
    overwrite: bool = False,
) -> Path:
    """Atomically write an NPZ and SHA-256 sidecar."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_name(f"{path.name}.sha256")
    if not overwrite and (path.exists() or sidecar.exists()):
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary_path: Path | None = None
    sidecar_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz.tmp", prefix=f".{path.name}.",
            dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = np.savez_compressed if compressed else np.savez
            writer(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        digest = sha256_file(temporary_path)
        os.replace(temporary_path, path)
        temporary_path = None
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sha256.tmp", prefix=f".{path.name}.",
            dir=path.parent, encoding="ascii", delete=False
        ) as temporary:
            sidecar_path = Path(temporary.name)
            temporary.write(f"{digest}  {path.name}\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(sidecar_path, sidecar)
        sidecar_path = None
        return path
    finally:
        for temporary in (temporary_path, sidecar_path):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def save_lexical_recording(
    destination: str | Path,
    result: Mapping[str, Any],
    *,
    recording_id: str,
    model_id: str,
    requested_revision: str | None,
    resolved_revision: str,
    overwrite: bool = False,
) -> Path:
    metadata = {
        "schema_version": 1,
        "recording_id": recording_id,
        "model_id": model_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "context_limit": result["context_limit"],
        "overlap_words": result["overlap_words"],
        "windows": result["windows"],
        "missing_word_indexes": result["missing_word_indexes"],
        "word_count": result["word_count"],
        "mapped_word_count": result["mapped_word_count"],
        "unknown_word_count": result["unknown_word_count"],
    }
    return write_npz_atomic(
        destination,
        {
            "vectors": result["vectors"],
            "word_index": result["word_index"],
            "words": result["words"],
            "start_seconds": result["start_seconds"],
            "end_seconds": result["end_seconds"],
            "piece_counts": result["piece_counts"],
            "unknown_piece_counts": result["unknown_piece_counts"],
            "owner_window": result["owner_window"],
            "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        },
        compressed=False,
        overwrite=overwrite,
    )


def load_lexical_recording(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    sidecar = path.with_name(f"{path.name}.sha256")
    if sidecar.read_text(encoding="ascii").split() != [sha256_file(path), path.name]:
        raise ValueError(f"Lexical archive integrity check failed: {path}")
    with np.load(path, allow_pickle=False) as stored:
        result = {key: stored[key] for key in stored.files if key != "metadata_json"}
        result["metadata"] = json.loads(stored["metadata_json"].item())
    return result


def verify_words_against_alignment(
    lexical: Mapping[str, Any], intervals: Sequence[Interval]
) -> None:
    words = ordered_word_intervals(intervals)
    expected_labels = [row.label.strip() for row in words]
    actual_labels = [str(value) for value in lexical["words"].tolist()]
    if actual_labels != expected_labels:
        raise ValueError("Lexical word labels/order differ from the current alignment")
    np.testing.assert_allclose(
        lexical["start_seconds"],
        [row.start for row in words],
        rtol=0,
        atol=1e-12,
        err_msg="Lexical word starts differ from the current alignment",
    )
    np.testing.assert_allclose(
        lexical["end_seconds"],
        [row.end for row in words],
        rtol=0,
        atol=1e-12,
        err_msg="Lexical word ends differ from the current alignment",
    )


def validate_activation_grids(
    recording_id: str,
    feature_times: np.ndarray,
    activation_stores: Sequence[str | Path],
) -> list[dict[str, Any]]:
    validations: list[dict[str, Any]] = []
    for store_path in activation_stores:
        path = Path(store_path)
        with h5py.File(path, "r") as store:
            if recording_id not in store:
                raise KeyError(f"{recording_id} is absent from activation store {path}")
            times = np.asarray(
                store[recording_id]["canonical_timestamps"][:], dtype=np.float64
            )
        if times.shape != feature_times.shape:
            raise ValueError(
                f"Frame mismatch for {recording_id}: features={len(feature_times)}, "
                f"{path}={len(times)}"
            )
        if not np.array_equal(times, feature_times):
            maximum = float(np.max(np.abs(times - feature_times)))
            raise ValueError(
                f"Non-exact timestamp mismatch for {recording_id} in {path}; "
                f"maximum absolute difference={maximum}"
            )
        validations.append(
            {
                "activation_store": str(path),
                "activation_store_sha256": sha256_file(path),
                "frames": len(times),
                "maximum_timestamp_error_seconds": float(
                    np.max(np.abs(times - feature_times), initial=0.0)
                ),
                "status": "PASS",
            }
        )
    return validations


def rasterize_lexical_events(
    vectors: np.ndarray, onset_indices: np.ndarray, n_frames: int
) -> np.ndarray:
    """Rasterize event vectors using the existing additive collision rule."""
    vectors = np.asarray(vectors)
    indices = np.asarray(onset_indices, dtype=int)
    if vectors.ndim != 2 or indices.ndim != 1 or len(vectors) != len(indices):
        raise ValueError("Lexical vectors/onset indices have incompatible shapes")
    if np.any(indices < 0) or np.any(indices >= n_frames):
        raise ValueError("Lexical onset index is outside the feature grid")
    result = np.zeros((n_frames, vectors.shape[1]), dtype=vectors.dtype)
    np.add.at(result, indices, vectors)
    return result


def _load_feature_archive(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as stored:
        required = {"matrix", "times", "names", "families"}
        missing = required - set(stored.files)
        if missing:
            raise ValueError(f"{path} is missing fields {sorted(missing)}")
        result = {key: stored[key] for key in required}
        if "log" in stored.files:
            result["log"] = json.loads(stored["log"].item())
    return result


def build_revised_recording(
    feature_archive: str | Path,
    lexical_archive: str | Path,
    *,
    recording_id: str,
    design_name: str,
    activation_stores: Sequence[str | Path],
    analysis_rate_hz: float = 50.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _load_feature_archive(feature_archive)
    lexical = load_lexical_recording(lexical_archive)
    if lexical["metadata"]["recording_id"] != recording_id:
        raise ValueError(f"Lexical archive belongs to a different recording: {lexical_archive}")
    matrix = np.asarray(source["matrix"], dtype=np.float32)
    times = np.asarray(source["times"], dtype=np.float64)
    names = [str(value) for value in source["names"].tolist()]
    families = [str(value) for value in source["families"].tolist()]
    if matrix.shape != (len(times), len(names)) or len(names) != len(families):
        raise ValueError(f"Malformed source feature archive {feature_archive}")
    expected_times = np.arange(len(times), dtype=np.float64) / analysis_rate_hz
    if not np.array_equal(times, expected_times):
        raise ValueError(
            f"{feature_archive} is not on the exact {analysis_rate_hz:g} Hz grid"
        )
    vectors = np.asarray(lexical["vectors"])
    lexical_lengths = {
        len(np.asarray(lexical[key]))
        for key in (
            "word_index",
            "words",
            "start_seconds",
            "end_seconds",
            "piece_counts",
            "unknown_piece_counts",
            "owner_window",
        )
    }
    if vectors.ndim != 2 or lexical_lengths != {len(vectors)}:
        raise ValueError(f"Malformed lexical archive {lexical_archive}")
    if not np.isfinite(vectors).all() or not np.isfinite(matrix).all():
        raise ValueError(f"Nonfinite lexical or dense predictors for {recording_id}")
    if not np.array_equal(
        lexical["word_index"], np.arange(len(vectors), dtype=np.int32)
    ):
        raise ValueError("Lexical word indexes are not consecutive from zero")

    onset_positions = [index for index, name in enumerate(names) if name == "recording_onset"]
    if len(onset_positions) != 1:
        raise ValueError(
            f"Expected exactly one recording_onset column in {feature_archive}; "
            f"found {len(onset_positions)}"
        )
    keep = [index for index in range(len(names)) if index not in onset_positions]
    revised_names = [names[index] for index in keep]
    revised_families = [families[index] for index in keep]
    revised_matrix = matrix[:, keep]
    if "recording_onset" in revised_names:
        raise AssertionError("recording_onset survived name-based removal")
    if set(revised_families) != set(TOP_LEVEL_FAMILIES):
        raise ValueError(
            f"Expected {TOP_LEVEL_FAMILIES} after onset removal; found "
            f"{sorted(set(revised_families))}"
        )
    word_names = [
        name for name, family in zip(revised_names, revised_families) if family == "word"
    ]
    if tuple(word_names) != WORD_TIMING_COLUMNS:
        raise ValueError(
            f"Word timing columns must be exactly {WORD_TIMING_COLUMNS}; found {word_names}"
        )

    grid_validation = validate_activation_grids(
        recording_id, times, activation_stores
    )
    starts = np.asarray(lexical["start_seconds"], dtype=np.float64)
    onset_indices = nearest_frame(starts, analysis_rate_hz, len(times)).astype(np.int32)
    unique_bins, counts = np.unique(onset_indices, return_counts=True)
    collision_mask = counts > 1
    collision_bins = unique_bins[collision_mask].astype(np.int32)
    collision_counts = counts[collision_mask].astype(np.int32)
    collision_words = {
        str(int(frame)): np.flatnonzero(onset_indices == frame).astype(int).tolist()
        for frame in collision_bins
    }

    blocks = []
    for family in ("acoustic", "prosodic", "phonetic"):
        indexes = [
            index for index, value in enumerate(revised_families) if value == family
        ]
        blocks.append(
            {
                "family": family,
                "block": family,
                "dense_column_indexes": indexes,
                "feature_names": [revised_names[index] for index in indexes],
            }
        )
    timing_indexes = [
        index for index, value in enumerate(revised_names) if value in WORD_TIMING_COLUMNS
    ]
    blocks.extend(
        [
            {
                "family": "word",
                "block": "word_timing",
                "dense_column_indexes": timing_indexes,
                "feature_names": list(WORD_TIMING_COLUMNS),
            },
            {
                "family": "word",
                "block": "lexical",
                "storage": "event_vectors_and_onset_indices",
                "rasterization": "nearest_frame_additive_after_fold_specific_reduction",
                "hidden_dimension": int(lexical["vectors"].shape[1]),
            },
        ]
    )
    schema = {
        "schema_version": 1,
        "recording_id": recording_id,
        "design": design_name,
        "analysis_rate_hz": analysis_rate_hz,
        "top_level_families": list(TOP_LEVEL_FAMILIES),
        "blocks": blocks,
        "recording_onset_removed_by_name": True,
        "lexical_model": lexical["metadata"]["model_id"],
        "lexical_resolved_revision": lexical["metadata"]["resolved_revision"],
        "lexical_collision_policy": "sum",
        "collision_word_indexes_by_frame": collision_words,
        "grid_validation": grid_validation,
        "source_feature_archive": str(feature_archive),
        "source_feature_sha256": sha256_file(feature_archive),
        "source_lexical_archive": str(lexical_archive),
        "source_lexical_sha256": sha256_file(lexical_archive),
    }
    arrays = {
        "matrix": revised_matrix,
        "times": times,
        "names": np.asarray(revised_names),
        "families": np.asarray(revised_families),
        "blocks": np.asarray(
            [
                "word_timing" if name in WORD_TIMING_COLUMNS else family
                for name, family in zip(revised_names, revised_families)
            ]
        ),
        "lexical_vectors": np.asarray(vectors, dtype=np.float32),
        "lexical_mapped": np.asarray(lexical["piece_counts"]) > 0,
        "lexical_onset_indices": onset_indices,
        "lexical_word_indices": np.asarray(lexical["word_index"], dtype=np.int32),
        "lexical_words": np.asarray(lexical["words"]),
        "lexical_start_seconds": starts,
        "lexical_end_seconds": np.asarray(lexical["end_seconds"], dtype=np.float64),
        "lexical_collision_bins": collision_bins,
        "lexical_collision_counts": collision_counts,
        "schema_json": json.dumps(schema, ensure_ascii=False, sort_keys=True),
    }
    report = {
        "recording_id": recording_id,
        "design": design_name,
        "frames": len(times),
        "source_dense_dimensions": matrix.shape[1],
        "revised_dense_dimensions": revised_matrix.shape[1],
        "lexical_events": len(onset_indices),
        "lexical_dimensions": lexical["vectors"].shape[1],
        "collision_bins": len(collision_bins),
        "colliding_events": int(np.sum(collision_counts)),
        "grid_status": "PASS",
    }
    return arrays, report


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
