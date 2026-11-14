import os
import sys

import numpy as np
import pytest
from scipy.io import wavfile

# The pipeline modules import each other as top-level modules (script layout).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Predict_and_extract"))

SAMPLE_RATE = 96000


def make_signal(duration=4.0, fs=SAMPLE_RATE, seed=0):
    """Synthetic 'whistle': a frequency sweep around 6-10 kHz plus noise (int16)."""
    t = np.arange(int(fs * duration)) / fs
    rng = np.random.default_rng(seed)
    x = 3000 * np.sin(2 * np.pi * (6000 * t + 400 * t ** 2)) + 800 * rng.standard_normal(t.size)
    return x.astype(np.int16)


@pytest.fixture
def wav_file(tmp_path):
    path = tmp_path / "recordings" / "Exp_01_Aug_2023_0845_channel_1.wav"
    path.parent.mkdir()
    wavfile.write(str(path), SAMPLE_RATE, make_signal())
    return path
