import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Predict_and_extract"))
import frontends  # noqa: E402

pytest.importorskip("scipy")
T = np.arange(frontends.WINDOW_SAMPLES) / frontends.SAMPLE_RATE


def windows(signal):
    noise = 0.01 * np.random.default_rng(0).standard_normal((2, frontends.WINDOW_SAMPLES))
    return (noise + signal).astype(np.float32)


@pytest.mark.parametrize("name", frontends.AUDIO_FRONTENDS)
def test_images_have_the_dataset_shape(name):
    images = frontends.window_images(name, windows(0.0))
    assert images.shape == (2, 224, 224) and images.dtype == np.uint8


def test_wide_band_keeps_a_tone_above_22_khz_at_the_right_row():
    image = frontends.window_images("wide", windows(0.05 * np.sin(2 * np.pi * 40000 * T)))[0]
    expected = (48000 - 40000) / (48000 - 2000) * 224  # high frequencies at the top
    assert abs(int(image.mean(axis=1).argmax()) - expected) <= 2


def test_whitening_removes_stationary_tones_and_keeps_a_short_whistle():
    tone = frontends.window_images("whiten", windows(0.05 * np.sin(2 * np.pi * 10000 * T)))[0]
    burst = np.zeros_like(T)
    part = slice(10000, 20000)
    burst[part] = 0.05 * np.sin(2 * np.pi * (8000 * T[part] + 20000 * (T[part] - T[part][0]) ** 2 / 0.1))
    whistle = frontends.window_images("whiten", windows(burst))[0]
    assert tone.max() < 128 < whistle.max()


def test_errors():
    with pytest.raises(ValueError):
        frontends.window_images("nope", windows(0.0))
    with pytest.raises(ValueError):
        frontends.window_images("wide", windows(0.0), fs=48000)
