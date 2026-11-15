import numpy as np
import pytest
from scipy.io import wavfile

sf = pytest.importorskip("soundfile")

import process_predictions as pp
import utils
from conftest import SAMPLE_RATE, make_signal


def write_predictions(root, stem, rows):
    folder = root / stem
    folder.mkdir(parents=True)
    utils.save_csv([f"{stem}.wav"] * len(rows), [r[0] for r in rows], [r[1] for r in rows],
                   [0.9] * len(rows), str(folder / f"{stem}.wav_predictions.csv"))
    return folder


def test_audioextraits_is_lossless(wav_file, tmp_path):
    paths = pp.audioextraits([(1.0, 2.5), (3.5, 10.0)], str(wav_file), str(tmp_path / "clips"))
    assert len(paths) == 2

    fs, clip = wavfile.read(paths[0])
    assert fs == SAMPLE_RATE and clip.dtype == np.int16
    assert np.array_equal(clip, make_signal()[SAMPLE_RATE:int(2.5 * SAMPLE_RATE)])

    # an interval running past the end of the file is clamped
    fs, clip = wavfile.read(paths[1])
    assert len(clip) == int(0.5 * SAMPLE_RATE)


def detections(count, first=0):
    return [(round(0.4 * i, 1), round(0.4 * i + 0.4, 1)) for i in range(first, first + count)]


def test_extraction_from_predictions_folder(wav_file, tmp_path):
    root = tmp_path / "analysed"
    stem = wav_file.stem
    write_predictions(root, stem, detections(9))  # 0.0 -> 3.6 s of contiguous detections
    short_folder = write_predictions(root, "short", detections(2))  # 0.8 s: below the 2 s minimum
    empty_folder = write_predictions(root, "Exp_02_Aug_2023_0845_channel_1", [])

    pp.process_prediction_files_in_folder(str(root), str(wav_file.parent), max_workers=2, audio_only=True)

    clips = list((root / "extracted_audio").rglob("*.wav"))
    assert [c.name for c in clips] == ["extrait_0.0_3.6.wav"]
    assert sf.info(str(clips[0])).frames == int(3.6 * SAMPLE_RATE)
    assert (short_folder / "pas_d_extraits" / "No_sequence_long_enough.txt").exists()
    assert (empty_folder / "pas_d_extraits" / "No_whistles_detected.txt").exists()


def test_sequence_rules_are_configurable(tmp_path):
    recordings = tmp_path / "rec"
    recordings.mkdir()
    wavfile.write(str(recordings / "long.wav"), SAMPLE_RATE, make_signal(duration=8.0))
    root = tmp_path / "analysed"
    # two groups of detections 2 s apart: 0-2.4 s and 4.4-6.8 s
    write_predictions(root, "long", detections(6) + detections(6, first=11))

    out = tmp_path / "clips"
    pp.process_prediction_files_in_folder(str(root), str(recordings), audio_only=True, exit=str(out))
    assert sorted(c.name for c in (out / "long").glob("*.wav")) == ["extrait_0.0_6.8.wav"]  # gap < 6 s: one sequence

    out = tmp_path / "clips_split"
    pp.process_prediction_files_in_folder(str(root), str(recordings), audio_only=True, exit=str(out), max_gap=1.0)
    assert sorted(c.name for c in (out / "long").glob("*.wav")) == ["extrait_0.0_2.4.wav", "extrait_4.4_6.8.wav"]


def test_missing_recording_leaves_a_note(tmp_path):
    root = tmp_path / "analysed"
    folder = write_predictions(root, "gone", detections(15))
    pp.process_prediction_files_in_folder(str(root), str(tmp_path), audio_only=True)
    assert (folder / "pas_d_extraits" / "No_Audio_found.txt").exists()
