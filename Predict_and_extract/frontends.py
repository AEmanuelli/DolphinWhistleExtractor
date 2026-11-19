"""
Spectrogram front-ends computed from the audio of 0.4 s windows, as alternatives to the images of the
OpenWhistle paper ("paper": 2-22 kHz, log power, min-max per window, computed by utils.spectrogram_image).

- "wide": 2-48 kHz instead of 2-22 kHz, so that whistle harmonics above 22 kHz are kept; log power, min-max
  per window, like the paper.
- "whiten": 2-48 kHz, log power minus the median of each frequency band over the window (stationary noise,
  spectral tilt), then minus the median of each time frame (broadband clicks), clipped to 0-30 dB.
- "pcen": 2-48 kHz, per-channel energy normalisation of the magnitude (Wang et al., 2017; parameters of
  librosa.pcen), an adaptive gain that flattens slowly varying noise, then min-max per window.

All use the short-time Fourier transform of the paper (1024-sample Blackman window, hop 512, 96 kHz, i.e.
74 frames per window), average the power over 224 frequency bands, repeat the frames to 224 columns
(nearest neighbour, as the paper's images) and return uint8 images of 224 x 224 pixels, high frequencies
at the top. The training script (``prepare``) and the detection pipeline both call ``window_images``.
"""
import numpy as np

FRONTENDS = ("paper", "wide", "whiten", "pcen")
AUDIO_FRONTENDS = FRONTENDS[1:]
SAMPLE_RATE = 96000
WINDOW_SAMPLES = 38400  # 0.4 s
WLEN, HOP = 1024, 512
LOW_HZ, HIGH_HZ = 2000.0, 48000.0
SIZE = 224
WHITEN_RANGE_DB = 30.0
PCEN = {"gain": 0.98, "bias": 2.0, "power": 0.5, "time_constant": 0.4, "eps": 1e-6}


def _band_pooling():
    """(SIZE, bins) matrix averaging the FFT bins of LOW_HZ-HIGH_HZ into SIZE bands of equal width."""
    freqs = np.fft.rfftfreq(WLEN, 1.0 / SAMPLE_RATE)
    keep = np.flatnonzero((freqs >= LOW_HZ) & (freqs <= HIGH_HZ))
    band = np.minimum((np.arange(len(keep)) * SIZE) // len(keep), SIZE - 1)
    pooling = np.zeros((SIZE, len(freqs)), dtype=np.float32)
    pooling[band, keep] = 1.0
    return pooling / pooling.sum(axis=1, keepdims=True)


_POOLING = _band_pooling()
_WINDOW = np.blackman(WLEN + 1)[:-1].astype(np.float32)  # periodic Blackman window, as scipy's get_window


def band_power(windows):
    """Power in the SIZE bands (high frequencies first) of each frame: float32 (n, SIZE, frames)."""
    windows = np.asarray(windows, dtype=np.float32)
    frames = 1 + (windows.shape[1] - WLEN) // HOP
    index = np.arange(WLEN)[None, :] + HOP * np.arange(frames)[:, None]
    segments = windows[:, index]
    segments = (segments - segments.mean(axis=2, keepdims=True)) * _WINDOW  # constant detrend, as scipy
    from scipy import fft

    spectrum = fft.rfft(segments, axis=2)  # float32 in, complex64 out
    power = spectrum.real ** 2 + spectrum.imag ** 2  # (n, frames, bins)
    return np.flip((power @ _POOLING.T).transpose(0, 2, 1), axis=1)  # (n, bands, frames)


def _minmax(values):
    low = values.min(axis=(1, 2), keepdims=True)
    span = values.max(axis=(1, 2), keepdims=True) - low
    return np.where(span > 0, (values - low) / np.where(span > 0, span, 1.0), 0.0) * 255.0


def _pcen(magnitude):
    from scipy.signal import lfilter, lfilter_zi

    frames_per_second = SAMPLE_RATE / HOP
    t = PCEN["time_constant"] * frames_per_second
    b = (np.sqrt(1 + 4 * t ** 2) - 1) / (2 * t ** 2)
    zi = lfilter_zi([b], [1, b - 1])[None, None, :] * magnitude[:, :, :1]
    smooth, _ = lfilter([b], [1, b - 1], magnitude, axis=2, zi=zi)
    gain = np.exp(-PCEN["gain"] * (np.log(PCEN["eps"]) + np.log1p(smooth / PCEN["eps"])))
    bias, power = PCEN["bias"], PCEN["power"]
    return bias ** power * np.expm1(power * np.log1p(magnitude * gain / bias))


def window_images(frontend, windows, fs=SAMPLE_RATE, power=None):
    """
    uint8 images (n, 224, 224) of 0.4 s windows (n, 38400) of float audio in [-1, 1] at 96 kHz.
    ``power`` (from ``band_power``) skips the Fourier transform when several front-ends are computed.
    """
    if frontend not in AUDIO_FRONTENDS:
        raise ValueError(f"Unknown audio front-end {frontend!r}, expected one of {AUDIO_FRONTENDS}")
    if int(fs) != SAMPLE_RATE:
        raise ValueError(f"Front-ends expect {SAMPLE_RATE} Hz audio, got {fs} Hz (resample first)")
    power = band_power(windows) if power is None else power
    if frontend == "pcen":
        values = _minmax(_pcen(np.sqrt(power) * 2.0 ** 31))  # librosa's scaling for float audio
    else:
        log_power = 10.0 * np.log10(power + 1e-19)
        if frontend == "wide":
            values = _minmax(log_power)
        else:
            log_power = log_power - np.median(log_power, axis=2, keepdims=True)  # per band, over time
            log_power = log_power - np.median(log_power, axis=1, keepdims=True)  # per frame, over bands
            values = np.clip(log_power, 0.0, WHITEN_RANGE_DB) * (255.0 / WHITEN_RANGE_DB)
    frames = values.shape[2]
    columns = (np.arange(SIZE) * frames) // SIZE  # nearest neighbour, as cv2.INTER_NEAREST
    return np.clip(np.rint(values[:, :, columns]), 0, 255).astype(np.uint8)
