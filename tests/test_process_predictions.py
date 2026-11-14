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


def test_extraction_from_predictions_folder(wav_file, tmp_path):
    root = tmp_path / "analysed"
    stem = wav_file.stem
    rows = [(round(0.4 * i, 1), round(0.4 * i + 0.4, 1)) for i in range(9)]
    write_predictions(root, stem, rows)
    empty_folder = write_predictions(root, "Exp_02_Aug_2023_0845_channel_1", [])

    pp.process_prediction_files_in_folder(str(root), str(wav_file.parent), max_workers=2, audio_only=True)

    # merged interval is 3.6 s long, below the 5 s minimum duration: nothing extracted for it
    assert not list((root / "extracted_audio").rglob("*.wav"))
    assert (empty_folder / "pas_d_extraits" / "No_whistles_detected.txt").exists()


def test_long_detection_is_extracted(tmp_path):
    recordings = tmp_path / "rec"
    recordings.mkdir()
    stem = "long"
    wavfile.write(str(recordings / f"{stem}.wav"), SAMPLE_RATE, make_signal(duration=8.0))
    root = tmp_path / "analysed"
    write_predictions(root, stem, [(round(0.4 * i, 1), round(0.4 * i + 0.4, 1)) for i in range(15)])

    out = tmp_path / "clips"
    pp.process_prediction_files_in_folder(str(root), str(recordings), audio_only=True, exit=str(out))
    clips = list((out / stem).glob("*.wav"))
    assert [c.name for c in clips] == ["extrait_0_6.0.wav"]
    assert sf.info(str(clips[0])).frames == 6 * SAMPLE_RATE


def test_missing_recording_leaves_a_note(tmp_path):
    root = tmp_path / "analysed"
    folder = write_predictions(root, "gone", [(round(0.4 * i, 1), round(0.4 * i + 0.4, 1)) for i in range(15)])
    pp.process_prediction_files_in_folder(str(root), str(tmp_path), audio_only=True)
    assert (folder / "pas_d_extraits" / "No_Audio_found.txt").exists()
