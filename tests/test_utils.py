import os

import cv2
import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

import utils
from conftest import SAMPLE_RATE, make_signal


def test_fusionner_intervalles_merges_and_pads():
    intervals = [(10.2, 10.6), (30.0, 30.4), (11.0, 11.4)]
    assert utils.fusionner_intervalles(intervals, hwindow=4) == [(6, 16), (26, 35)]
    # the caller's list is left untouched
    assert intervals == [(10.2, 10.6), (30.0, 30.4), (11.0, 11.4)]


def test_fusionner_intervalles_clamps_start_to_zero():
    assert utils.fusionner_intervalles([(1.0, 2.0)], hwindow=4) == [(0, 6)]
    assert utils.fusionner_intervalles([]) == []


def test_fusionner_intervalles_avec_seuil():
    intervals = [(0.0, 0.4), (0.4, 0.8), (3.0, 4.0), (20.0, 20.4), (40.0, 44.0)]
    merged = utils.fusionner_intervalles_avec_seuil(intervals, fusion_threshold=3, duration_threshold=3)
    # first three merge into (0, 4); the isolated short one is dropped; the long one is kept
    assert merged == [(0.0, 4.0), (40.0, 44.0)]
    assert utils.fusionner_intervalles_avec_seuil([]) == []


def test_lire_csv_extraits_handles_both_layouts(tmp_path):
    with_names = tmp_path / "a.csv"
    with_names.write_text("file_name,initial_point,finish_point,confidence\nrec.wav,1.2,1.6,0.9\n\nrec.wav,oops\n")
    assert utils.lire_csv_extraits(str(with_names)) == [(1.2, 1.6)]

    without_names = tmp_path / "b.csv"
    without_names.write_text("start,end\n3,4.5\n")
    assert utils.lire_csv_extraits(str(without_names)) == [(3.0, 4.5)]

    empty = tmp_path / "c.csv"
    empty.write_text("")
    assert utils.lire_csv_extraits(str(empty)) == []


def test_save_csv_roundtrip_and_empty(tmp_path):
    path = tmp_path / "p.csv"
    utils.save_csv(["rec.wav"], [0.4], [0.8], [0.75], str(path))
    assert utils.lire_csv_extraits(str(path)) == [(0.4, 0.8)]

    utils.save_csv([], [], [], [], str(path))
    df = pd.read_csv(path)
    assert list(df.columns) == ["file_name", "initial_point", "finish_point", "confidence"]
    assert df.empty


def test_transform_file_name():
    assert utils.transform_file_name("Exp_01_Aug_2023_0845_channel_1.wav") == "01_Aug_23_0845_c1"
    assert utils.transform_file_name("other.wav") == "other"


def test_process_audio_file_window_count_and_shape(wav_file):
    # 4 s of audio, 0.8 s windows every 0.4 s -> 9 windows
    images = utils.process_audio_file(str(wav_file), start_time=0, end_time=None, batch_size=50)
    assert len(images) == 9
    assert all(img.shape == (875, 1167, 3) and img.dtype == np.uint8 for img in images)
    # there is real contrast in the image
    assert images[0].max() > images[0].min()

    assert len(utils.process_audio_file(str(wav_file), end_time=None, batch_size=4)) == 4
    assert len(utils.process_audio_file(str(wav_file), start_time=2.0, end_time=3.0)) == 1
    assert utils.process_audio_file(str(wav_file), start_time=3.9, end_time=None) == []


def test_batches_match_single_pass(wav_file):
    """Processing in several batches gives the same images as a single pass."""
    fs, x = utils.load_audio(str(wav_file))
    full = utils.audio_to_spectrogram_images(x, fs, end_time=None, batch_size=50)
    batched = []
    for batch in range(3):
        batched += utils.audio_to_spectrogram_images(x, fs, start_time=batch * 4 * 0.4, end_time=None, batch_size=4)
    assert len(batched) == len(full)
    assert all(np.array_equal(a, b) for a, b in zip(full, batched))


def test_frequency_band_changes_image(wav_file):
    low_band = utils.process_audio_file(str(wav_file), end_time=1, cut_low_frequency=3, cut_high_frequency=20)[0]
    high_band = utils.process_audio_file(str(wav_file), end_time=1, cut_low_frequency=25, cut_high_frequency=40)[0]
    assert not np.array_equal(low_band, high_band)


def test_silence_gives_black_images_without_nan(tmp_path):
    path = tmp_path / "silence.wav"
    wavfile.write(str(path), SAMPLE_RATE, np.zeros(SAMPLE_RATE, dtype=np.int16))
    images = utils.process_audio_file(str(path), end_time=None)
    assert len(images) == 1
    assert not images[0].any()


def test_stereo_uses_first_channel(tmp_path, wav_file):
    mono = make_signal()
    stereo = np.stack([mono, np.zeros_like(mono)], axis=1)
    path = tmp_path / "stereo.wav"
    wavfile.write(str(path), SAMPLE_RATE, stereo)
    a = utils.process_audio_file(str(path), end_time=1)
    b = utils.process_audio_file(str(wav_file), end_time=1)
    assert np.array_equal(a[0], b[0])


def test_save_writes_images(wav_file, tmp_path):
    out = tmp_path / "imgs"
    utils.process_audio_file(str(wav_file), saving_folder=str(out), end_time=2, save=True)
    assert len(list(out.glob("*.jpg"))) == 4


# --- paper-faithful spectrograms (arXiv:2609.34839) ---------------------------------------------

BENCHMARK_SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "openwhistle_cnn")


def test_paper_settings_match_the_paper():
    settings = utils.PAPER_SPECTROGRAM
    assert (settings.window_duration, settings.window_hop) == (0.4, 0.4)  # non-overlapping 0.4 s windows
    assert (settings.wlen, settings.nfft, settings.noverlap) == (1024, 1024, 512)  # hop of 512 samples
    assert (settings.cut_low_frequency, settings.cut_high_frequency) == (2.0, 22.0)
    assert settings.target_fs == 96000
    assert utils.LEGACY_SPECTROGRAM.noverlap == 2048 - round(0.8 * 2048)


@pytest.mark.parametrize("name", ["whistle", "noise"])
def test_spectrogram_reproduces_openwhistle_cnn_images(name):
    """The pipeline computes the spectrograms shipped with the OpenWhistle-CNN benchmark."""
    expected = cv2.imread(os.path.join(BENCHMARK_SAMPLES, name + ".png"))
    fs, x = utils.load_audio(os.path.join(BENCHMARK_SAMPLES, name + ".wav"))
    fs, x = utils.prepare_audio(x, fs)
    image = utils.spectrogram_image(x, fs)

    assert image.shape == expected.shape == (224, 224, 3)
    difference = np.abs(image.astype(int) - expected.astype(int))
    # identical up to float rounding across library versions: at most one grey level on a few pixels
    assert difference.max() <= 1
    assert (difference > 0).mean() < 0.001


def test_prepare_audio_scales_and_selects_first_channel():
    stereo = np.array([[16384, 0], [-32768, 0]], dtype=np.int16)
    fs, x = utils.prepare_audio(stereo, 96000)
    assert fs == 96000 and x.dtype == np.float32
    assert x.tolist() == [0.5, -1.0]

    fs, x = utils.prepare_audio(np.array([0.25, -0.5]), 48000, utils.SpectrogramSettings(target_fs=None))
    assert fs == 48000 and x.dtype == np.float32 and x.tolist() == [0.25, -0.5]

    # legacy settings: native samples, no resampling
    fs, x = utils.prepare_audio(stereo, 48000, utils.LEGACY_SPECTROGRAM)
    assert fs == 48000 and x.dtype == np.int16 and x.tolist() == [16384, -32768]


def test_legacy_settings_reproduce_the_previous_pipeline(wav_file):
    """`--spectrogram legacy` gives the images this repository produced before following the paper."""
    previous = utils.process_audio_file(str(wav_file), start_time=0, end_time=None, batch_size=50)
    fs, x = utils.prepare_audio(*reversed(utils.load_audio(str(wav_file))), utils.LEGACY_SPECTROGRAM)
    starts, size = utils.window_starts(len(x), fs, utils.LEGACY_SPECTROGRAM)
    assert len(starts) == len(previous) == 9
    for low, expected in zip(starts, previous):
        assert np.array_equal(utils.spectrogram_image(x[low:low + size], fs, utils.LEGACY_SPECTROGRAM), expected)


def test_prepare_audio_resamples_to_target_rate():
    pytest.importorskip("librosa")
    fs, x = utils.prepare_audio(make_signal(duration=1.0, fs=48000), 48000)
    assert fs == 96000 and len(x) == 96000


def test_window_starts():
    fs = SAMPLE_RATE
    starts, size = utils.window_starts(4 * fs, fs)  # 4 s -> ten non-overlapping 0.4 s windows
    assert size == 38400 and starts.tolist() == [38400 * k for k in range(10)]

    starts, _ = utils.window_starts(4 * fs, fs, start_time=1.0, end_time=2.0)  # two complete windows in [1, 2]
    assert starts.tolist() == [96000, 134400]
    assert len(utils.window_starts(fs // 4, fs)[0]) == 0  # shorter than one window

    starts, size = utils.window_starts(4 * fs, fs, utils.LEGACY_SPECTROGRAM)  # 0.8 s windows every 0.4 s
    assert size == 76800 and len(starts) == 9 and starts[1] == 38400


def test_spectrogram_image_sizes_and_silence():
    fs, x = utils.prepare_audio(make_signal(duration=0.8), SAMPLE_RATE)
    assert utils.spectrogram_image(x[:38400], fs, size=(100, 60)).shape == (60, 100, 3)
    legacy = utils.spectrogram_image(make_signal(duration=0.8), fs, utils.LEGACY_SPECTROGRAM)
    assert legacy.shape == (875, 1167, 3) and legacy.dtype == np.uint8
    assert not utils.spectrogram_image(np.zeros(38400, dtype=np.float32), fs).any()
    assert not utils.spectrogram_image(np.zeros(76800, dtype=np.int16), fs, utils.LEGACY_SPECTROGRAM).any()


def test_build_sequences_paper_rule():
    step = lambda k: (round(0.4 * k, 1), round(0.4 * k + 0.4, 1))
    # 0-2.4 s; a detection 4 s later joins it (gap < 6 s); one 10 s later starts a new, too short, sequence
    detections = [step(k) for k in range(6)] + [(6.4, 6.8), (16.8, 17.2)]
    assert utils.build_sequences(detections) == [(0.0, 6.8)]
    assert utils.build_sequences([]) == []
    assert utils.build_sequences([(0.0, 0.4)]) == []  # shorter than 2 s

    # a 50 s run of detections is split into sequences of at most 20 s
    sequences = utils.build_sequences([step(k) for k in range(125)])
    assert sequences == [(0.0, 20.0), (20.0, 40.0), (40.0, 50.0)]
    assert all(end - start <= 20 for start, end in sequences)

    # custom rule: no grouping across 1 s gaps, keep everything
    assert utils.build_sequences([(0, 0.4), (2, 2.4)], max_gap=1, min_duration=0) == [(0, 0.4), (2, 2.4)]
