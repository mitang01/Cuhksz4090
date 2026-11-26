from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import torch

from speech_strf.alignments import Interval
from speech_strf.lexical_predictors import (
    build_revised_recording,
    extract_recording_embeddings,
    plan_context_windows,
    rasterize_lexical_events,
    save_lexical_recording,
)


class FakeEncoding(dict):
    def __init__(self, *args, word_ids, **kwargs):
        super().__init__(*args, **kwargs)
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        assert batch_index == 0
        return self._word_ids


class FakeTokenizer:
    is_fast = True
    model_max_length = 6
    unk_token_id = 99
    init_kwargs = {"_commit_hash": "abc123"}

    def num_special_tokens_to_add(self, pair=False):
        assert pair is False
        return 2

    def __call__(
        self,
        words,
        *,
        is_split_into_words,
        add_special_tokens,
        truncation,
        return_tensors=None,
        **kwargs,
    ):
        assert is_split_into_words
        assert truncation is False
        ids = []
        mapping = []
        if add_special_tokens:
            ids.append(1)
            mapping.append(None)
        for word_index, word in enumerate(words):
            if word == "<missing>":
                continue
            for character in word:
                ids.append(99 if character == "?" else 10 + ord(character) % 50)
                mapping.append(word_index)
        if add_special_tokens:
            ids.append(2)
            mapping.append(None)
        if return_tensors == "pt":
            values = torch.tensor([ids], dtype=torch.long)
            payload = {
                "input_ids": values,
                "attention_mask": torch.ones_like(values),
                "token_type_ids": torch.zeros_like(values),
            }
        else:
            payload = {"input_ids": ids}
        return FakeEncoding(payload, word_ids=mapping)


class FakeModel:
    config = SimpleNamespace(
        hidden_size=2,
        max_position_embeddings=6,
        _commit_hash="abc123",
    )

    def eval(self):
        return self

    def requires_grad_(self, value):
        assert value is False
        return self

    def to(self, device):
        assert device == "cpu"
        return self

    def __call__(self, input_ids, **kwargs):
        values = input_ids.to(torch.float32)
        hidden = torch.stack([values, values + 0.5], dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


def test_multichar_words_are_mean_pooled_from_presegmented_mapping():
    words = [
        Interval("words", 0.0, 0.1, "中国"),
        Interval("words", 0.2, 0.3, "语"),
    ]
    result = extract_recording_embeddings(
        words, FakeTokenizer(), FakeModel(), overlap_words=1
    )

    assert result["piece_counts"].tolist() == [2, 1]
    assert result["mapped_word_count"] == 2
    first_ids = [10 + ord(character) % 50 for character in "中国"]
    np.testing.assert_allclose(
        result["vectors"][0],
        [np.mean(first_ids), np.mean(first_ids) + 0.5],
    )


def test_overlapping_windows_emit_every_word_once_and_report_coverage():
    windows = plan_context_windows(
        [2, 2, 2, 2], max_tokens=6, special_tokens=2, overlap_words=1
    )
    assert [(row.start_word, row.end_word) for row in windows] == [
        (0, 2),
        (1, 3),
        (2, 4),
    ]
    emitted = [
        index
        for row in windows
        for index in range(row.emit_start_word, row.emit_end_word)
    ]
    assert emitted == [0, 1, 2, 3]

    words = [
        Interval("word", 0.0, 0.1, "甲乙"),
        Interval("word", 0.1, 0.2, "<missing>"),
        Interval("word", 0.2, 0.3, "?"),
    ]
    result = extract_recording_embeddings(
        words, FakeTokenizer(), FakeModel(), overlap_words=1, max_tokens=5
    )
    assert result["missing_word_indexes"] == [1]
    assert result["mapped_word_count"] == 2
    assert result["unknown_word_count"] == 1
    assert sorted(result["owner_window"].tolist()) == [0, 0, 1]


def test_onset_collisions_use_existing_additive_event_rule():
    vectors = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    raster = rasterize_lexical_events(vectors, np.asarray([0, 0, 2]), 4)
    np.testing.assert_array_equal(
        raster,
        np.asarray([[4.0, 6.0], [0.0, 0.0], [5.0, 6.0], [0.0, 0.0]]),
    )


def test_revised_archive_removes_onset_and_matches_activation_grid(tmp_path):
    times = np.arange(6, dtype=np.float64) / 50.0
    features = tmp_path / "features.npz"
    np.savez(
        features,
        matrix=np.arange(42, dtype=np.float32).reshape(6, 7),
        times=times,
        names=np.asarray(
            [
                "acoustic_1",
                "prosody_1",
                "recording_onset",
                "phone_onset",
                "word_onset",
                "word_boundary",
                "word_duration",
            ]
        ),
        families=np.asarray(
            [
                "acoustic",
                "prosodic",
                "onset",
                "phonetic",
                "word",
                "word",
                "word",
            ]
        ),
        log="{}",
    )
    activation_store = tmp_path / "activations.h5"
    with h5py.File(activation_store, "w") as store:
        group = store.create_group("recording")
        group.create_dataset("canonical_timestamps", data=times)

    lexical_path = tmp_path / "lexical.npz"
    lexical_result = {
        "vectors": np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        "word_index": np.arange(3, dtype=np.int32),
        "words": np.asarray(["甲", "乙", "中国"]),
        "start_seconds": np.asarray([0.001, 0.009, 0.08]),
        "end_seconds": np.asarray([0.005, 0.015, 0.10]),
        "piece_counts": np.asarray([1, 1, 2], dtype=np.int32),
        "unknown_piece_counts": np.zeros(3, dtype=np.int32),
        "owner_window": np.asarray([0, 0, 0], dtype=np.int32),
        "windows": [],
        "missing_word_indexes": [],
        "word_count": 3,
        "mapped_word_count": 3,
        "unknown_word_count": 0,
        "context_limit": 6,
        "overlap_words": 1,
    }
    save_lexical_recording(
        lexical_path,
        lexical_result,
        recording_id="recording",
        model_id="hfl/chinese-roberta-wwm-ext",
        requested_revision=None,
        resolved_revision="abc123",
    )

    arrays, report = build_revised_recording(
        features,
        lexical_path,
        recording_id="recording",
        design_name="original",
        activation_stores=[activation_store],
    )

    assert arrays["names"].tolist() == [
        "acoustic_1",
        "prosody_1",
        "phone_onset",
        "word_onset",
        "word_boundary",
        "word_duration",
    ]
    assert "recording_onset" not in arrays["names"]
    assert arrays["lexical_onset_indices"].tolist() == [0, 0, 4]
    assert arrays["lexical_collision_bins"].tolist() == [0]
    assert arrays["lexical_collision_counts"].tolist() == [2]
    np.testing.assert_array_equal(arrays["times"], times)
    assert report["grid_status"] == "PASS"
