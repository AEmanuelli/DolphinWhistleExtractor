import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")  # imported by evaluate_models, which this script reuses
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Train_model"))

import check_dataset_overlap as chk


def recording(seed, n=200_000):
    rng = np.random.default_rng(seed)
    return (2000 * rng.standard_normal(n)).astype(np.int16)


def fingerprint_all(clips):
    hashes, ids, positions = [], [], []
    for index, clip in enumerate(clips):
        h, p = chk.fingerprints(clip)
        hashes.append(h)
        positions.append(p)
        ids.append(np.full(len(h), index))
    return np.concatenate(hashes), np.concatenate(ids), np.concatenate(positions)


def test_metadata_overlaps_and_sessions():
    metadata = pd.DataFrame([
        ("train", "a.wav", "rec1", 0.0, 0.4),
        ("train", "b.wav", "rec1", 0.4, 0.8),          # adjacent: no overlap
        ("test", "c.wav", "rec1", 0.7, 1.1),           # overlaps b by 0.1 s, across splits
        ("test", "d.wav", "rec2_channel_1", 0.0, 0.4),
        ("train", "e.wav", "rec2_channel_0", 0.0, 0.4),  # other recording name, same session
    ], columns=["split", "file_name", "recording", "onset", "offset"])

    pairs = chk.metadata_overlaps(metadata)
    assert [(p["file_a"], p["file_b"], p["overlap_seconds"]) for p in pairs] == [("b.wav", "c.wav", 0.1)]
    assert {p["split_a"] for p in pairs} | {p["split_b"] for p in pairs} == {"train", "test"}
    assert chk.shared_sessions(metadata) == {("test", "train"): ["rec1", "rec2"]}


def test_shared_audio_is_found_whatever_the_offset():
    source, other = recording(0), recording(1)
    clips = [
        source[10_000:48_400],    # 0: reference window
        source[25_000:63_400],    # 1: starts 15,000 samples later -> overlaps 0 by 23,400 samples
        source[48_400:86_800],    # 2: adjacent to 0 -> no common audio; overlaps 1 by 15,000 samples
        other[10_000:48_400],     # 3: another recording
        source[100_000:138_400],  # 4: same recording, far away
    ]
    shared = chk.find_shared_audio(*fingerprint_all(clips))

    assert set(shared) == {(0, 1), (1, 2)}
    assert shared[(0, 1)][0] == 15_000 and shared[(1, 2)][0] == 23_400
    assert shared[(0, 1)][1] >= chk.MIN_SHARED


def test_silence_and_short_clips_are_not_matched():
    silence = np.zeros(38_400, dtype=np.int16)
    clipped = np.full(38_400, 32767, dtype=np.int16)
    hashes, ids, positions = fingerprint_all([silence, silence.copy(), clipped, clipped.copy()])
    assert len(hashes) == 0
    assert len(chk.fingerprints(np.zeros(10, dtype=np.int16))[0]) == 0

    # a short stretch of digital silence inside otherwise unrelated windows is not an overlap
    a, b = recording(2, 38_400), recording(3, 38_400)
    a[1000:3000] = 0
    b[5000:7000] = 0
    assert chk.find_shared_audio(*fingerprint_all([a, b])) == {}
