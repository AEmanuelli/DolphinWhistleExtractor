import numpy as np
import pandas as pd
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
