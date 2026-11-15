#!/usr/bin/env python3
"""Check that the windows of the OpenWhistle-CNN dataset do not overlap, in particular across splits.

Dataset: https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN

Two independent checks:

1. **Metadata** (fast, no audio needed): every window has a recording name and an onset / offset. Two
   windows overlap when they belong to the same recording and their time intervals intersect. Sessions
   (recordings without their ``_channel_N`` suffix) must not be shared between splits.

2. **Audio content** (downloads the audio, trusts no metadata): two windows cut from the same stretch of
   a recording contain an identical run of samples, at some offset. For every window, short runs of
   samples are fingerprinted at positions chosen from the signal itself (local maxima), so that the
   same audio yields the same fingerprints whatever its position in the window. Windows sharing several
   fingerprints at one consistent offset overlap, by ``window length - |offset|`` samples.
   The overlaps known from the metadata serve as a control: the content check must find them too.

Examples:
    python Train_model/check_dataset_overlap.py --metadata_only
    python Train_model/check_dataset_overlap.py                   # both checks, ~6 GB of downloads
    python Train_model/check_dataset_overlap.py --splits test     # audio check inside the test split only

Requires: pandas, pyarrow, scipy.
"""
import argparse
import collections
import io
import os
import re
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_models import REPO_ROOT, SPLIT_SHARDS, download_split  # noqa: E402

METADATA_URL = ("https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN/resolve/main/"
                "artifacts/openwhistle_simple_metadata.csv")
ANCHOR_SPACING = 257     # one anchor per ~257 samples on average (local maximum over that many samples)
FINGERPRINT_HALF = 32    # a fingerprint covers 64 samples around its anchor
MIN_SHARED = 3           # fingerprints two windows must share, at the same offset, to be reported
MIN_OVERLAP_SECONDS = 1e-3  # shorter intersections of time intervals are rounding effects of the onsets
_WEIGHTS = np.random.default_rng(12345).integers(1, 2 ** 63, 2 * FINGERPRINT_HALF, dtype=np.uint64) | np.uint64(1)


def session_id(recording):
    """Simultaneous channels of one recording belong to the same session."""
    return re.sub(r"_channel_\d+$", "", str(recording).strip())


# --------------------------------------------------------------------------------------------------
# 1. metadata check
# --------------------------------------------------------------------------------------------------

def metadata_overlaps(metadata, min_overlap=MIN_OVERLAP_SECONDS):
    """
    Pairs of windows of the same recording whose time intervals intersect by at least ``min_overlap`` seconds.

    Args:
        metadata: DataFrame with columns split, recording, onset, offset, file_name.

    Returns:
        list of dicts (file_a, file_b, split_a, split_b, recording, overlap_seconds).
    """
    pairs = []
    for recording, group in metadata.groupby("recording", sort=False):
        group = group.sort_values("onset")
        onset, offset = group["onset"].to_numpy(), group["offset"].to_numpy()
        names, splits = group["file_name"].to_numpy(), group["split"].to_numpy()
        for i in range(len(group)):
            j = i + 1
            while j < len(group) and onset[j] < offset[i] - min_overlap:
                pairs.append({"file_a": names[i], "file_b": names[j], "split_a": splits[i], "split_b": splits[j],
                              "recording": recording,
                              "overlap_seconds": round(float(min(offset[i], offset[j]) - onset[j]), 4)})
                j += 1
    return pairs


def shared_sessions(metadata):
    """Sessions present in more than one split: {(split_a, split_b): sorted list of sessions}."""
    sessions = {split: set(group["recording"].map(session_id)) for split, group in metadata.groupby("split")}
    names = sorted(sessions)
    return {(a, b): sorted(sessions[a] & sessions[b]) for i, a in enumerate(names) for b in names[i + 1:]}


# --------------------------------------------------------------------------------------------------
# 2. audio content check
# --------------------------------------------------------------------------------------------------

def fingerprints(samples):
    """
    Position-independent fingerprints of a signal.

    Returns:
        (hashes, positions): uint64 hash of the 64 samples around each anchor, and the anchor positions.
        Anchors are the local maxima of the signal over ``ANCHOR_SPACING`` samples; constant runs
        (digital silence, clipping) are ignored because they are identical in unrelated recordings.
    """
    from scipy.ndimage import maximum_filter1d

    x = np.asarray(samples)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.int64)
    half = FINGERPRINT_HALF
    if len(x) < 2 * half + 1:
        return np.array([], dtype=np.uint64), np.array([], dtype=np.int64)

    is_anchor = x == maximum_filter1d(x, size=ANCHOR_SPACING, mode="constant", cval=np.iinfo(np.int64).min)
    positions = np.flatnonzero(is_anchor)
    positions = positions[(positions >= half) & (positions <= len(x) - half)]
    if len(positions) == 0:
        return np.array([], dtype=np.uint64), positions

    windows = x[positions[:, None] + np.arange(-half, half)]
    varied = windows.min(axis=1) != windows.max(axis=1)
    windows, positions = windows[varied], positions[varied]
    hashes = (windows.astype(np.uint64) * _WEIGHTS).sum(axis=1, dtype=np.uint64)
    return hashes, positions


def find_shared_audio(hashes, clip_ids, positions, min_shared=MIN_SHARED, max_clips_per_hash=50):
    """
    Pairs of clips sharing at least ``min_shared`` fingerprints at one consistent offset.

    Args:
        hashes, clip_ids, positions: parallel arrays, one entry per fingerprint.

    Returns:
        dict {(clip_a, clip_b): (offset_samples, shared_fingerprints)} with clip_a < clip_b and
        offset = position in clip_a - position in clip_b of the common audio.
    """
    order = np.argsort(hashes, kind="stable")
    hashes, clip_ids, positions = hashes[order], clip_ids[order], positions[order]
    boundaries = np.flatnonzero(np.diff(hashes)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(hashes)]])

    votes = collections.Counter()
    for start, end in zip(starts[ends - starts > 1], ends[ends - starts > 1]):
        clips, where = clip_ids[start:end], positions[start:end]
        if len(np.unique(clips)) < 2 or end - start > max_clips_per_hash:
            continue
        for i in range(len(clips)):
            for j in range(i + 1, len(clips)):
                if clips[i] == clips[j]:
                    continue
                a, b = (i, j) if clips[i] < clips[j] else (j, i)
                votes[(int(clips[a]), int(clips[b]), int(where[a] - where[b]))] += 1

    best = {}
    for (a, b, offset), count in votes.items():
        if count >= min_shared and count > best.get((a, b), (0, 0))[1]:
            best[(a, b)] = (offset, count)
    return best


def fingerprint_dataset(paths_by_split):
    """Fingerprint every window of the given parquet files. Returns (hashes, clip_ids, positions, clips)."""
    import pyarrow.parquet as pq
    from scipy.io import wavfile

    all_hashes, all_ids, all_positions, clips = [], [], [], []
    for split, paths in paths_by_split.items():
        for path in paths:
            parquet = pq.ParquetFile(path)
            for group in range(parquet.num_row_groups):
                table = parquet.read_row_group(group, columns=["audio", "file_name"])
                for audio, file_name in zip(table.column("audio").to_pylist(), table.column("file_name").to_pylist()):
                    fs, x = wavfile.read(io.BytesIO(audio["bytes"]))
                    hashes, positions = fingerprints(x)
                    all_hashes.append(hashes)
                    all_positions.append(positions.astype(np.int32))
                    all_ids.append(np.full(len(hashes), len(clips), dtype=np.int32))
                    clips.append({"split": split, "file_name": file_name, "samples": len(x), "fs": fs})
            print(f"  fingerprinted {os.path.basename(path)} ({len(clips)} windows so far)", flush=True)
    return np.concatenate(all_hashes), np.concatenate(all_ids), np.concatenate(all_positions), clips


# --------------------------------------------------------------------------------------------------

def summarise(pairs, label):
    """Print how many overlapping pairs are inside one split / across two splits."""
    counts = collections.Counter(
        (p["split_a"],) if p["split_a"] == p["split_b"] else tuple(sorted((p["split_a"], p["split_b"])))
        for p in pairs)
    cross = sum(count for key, count in counts.items() if len(key) == 2)
    print(f"{label}: {len(pairs)} overlapping pairs of windows, {cross} of them across two splits")
    for key, count in sorted(counts.items()):
        print(f"    {' / '.join(key):24s} {count}")
    return cross


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", choices=sorted(SPLIT_SHARDS), default=sorted(SPLIT_SHARDS),
                        help="Splits included in the audio content check")
    parser.add_argument("--metadata_only", action="store_true", help="Skip the audio content check")
    parser.add_argument("--cache_dir", default=os.path.join(REPO_ROOT, "openwhistle_cnn_cache"),
                        help="Where the dataset files are downloaded")
    parser.add_argument("--output_csv", help="Write the overlapping pairs found in the audio to this CSV file")
    args = parser.parse_args(argv)

    import pandas as pd

    os.makedirs(args.cache_dir, exist_ok=True)
    metadata_path = os.path.join(args.cache_dir, "openwhistle_simple_metadata.csv")
    if not os.path.exists(metadata_path):
        urllib.request.urlretrieve(METADATA_URL, metadata_path)
    metadata = pd.read_csv(metadata_path)
    print(f"{len(metadata)} windows in the metadata: {metadata['split'].value_counts().to_dict()}")

    print("\n== 1. Metadata check ==")
    leaks = 0
    for (a, b), sessions in shared_sessions(metadata).items():
        print(f"sessions shared between {a} and {b}: {len(sessions)}" + (f" {sessions[:5]}" if sessions else ""))
        leaks += len(sessions)
    from_metadata = metadata_overlaps(metadata)
    leaks += summarise(from_metadata, "time intervals")

    if not args.metadata_only:
        print("\n== 2. Audio content check ==")
        paths = {split: download_split(split, args.cache_dir) for split in args.splits}
        hashes, clip_ids, positions, clips = fingerprint_dataset(paths)
        print(f"{len(clips)} windows, {len(hashes)} fingerprints")
        shared = find_shared_audio(hashes, clip_ids, positions)
        from_audio = []
        for (a, b), (offset, count) in sorted(shared.items()):
            overlap = (min(clips[a]["samples"], clips[b]["samples"]) - abs(offset)) / clips[a]["fs"]
            from_audio.append({"file_a": clips[a]["file_name"], "file_b": clips[b]["file_name"],
                               "split_a": clips[a]["split"], "split_b": clips[b]["split"],
                               "overlap_seconds": round(overlap, 4), "shared_fingerprints": count})
        leaks += summarise(from_audio, "identical audio")

        # control: overlaps known from the metadata must be found in the audio as well
        known = {frozenset((p["file_a"], p["file_b"])): p["overlap_seconds"] for p in from_metadata
                 if p["split_a"] in args.splits and p["split_b"] in args.splits}
        found = {frozenset((p["file_a"], p["file_b"])) for p in from_audio}
        long_enough = [pair for pair, seconds in known.items() if seconds >= 0.05]
        print(f"control: {sum(pair in found for pair in long_enough)} of the {len(long_enough)} overlaps of at least "
              f"0.05 s announced by the metadata are found in the audio")
        print(f"overlaps found in the audio but absent from the metadata: {len(found - set(known))}")
        if args.output_csv:
            pd.DataFrame(from_audio).to_csv(args.output_csv, index=False)
            print(f"pairs written to {args.output_csv}")

    print("\nRESULT:", "no window is shared between splits" if leaks == 0
          else f"{leaks} overlap(s) or shared session(s) between splits")
    return leaks


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
