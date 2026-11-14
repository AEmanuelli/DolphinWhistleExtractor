import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

import predict_and_extract_online as pipeline
from conftest import SAMPLE_RATE, make_signal


class FakeDetector:
    """Stands in for the Keras model: flags every other window as a whistle."""
    image_size = (32, 24)

    def __init__(self, model_path=None):
        self.batches = []

    def predict(self, batch):
        self.batches.append(batch)
        scores = np.where(np.arange(len(batch)) % 2 == 0, 0.9, 0.1)
        return np.stack([1 - scores, scores], axis=1)


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(pipeline, "WhistleDetector", FakeDetector)


def run(wav_file, out, **kwargs):
    kwargs.setdefault("end_time", None)
    kwargs.setdefault("max_workers", 1)
    return pipeline.process_predict_extract(str(wav_file.parent), str(out), model_path="unused", **kwargs)


def test_positive_scores_output_layouts():
    mask, scores = pipeline.positive_scores(np.array([[0.2], [0.7]]), 0.5)
    assert mask.tolist() == [False, True] and scores.tolist() == [0.2, 0.7]

    mask, scores = pipeline.positive_scores(np.array([[0.8, 0.2], [0.3, 0.7]]), 0.6)
    assert mask.tolist() == [False, True] and scores.tolist() == [0.2, 0.7]

    mask, _ = pipeline.positive_scores(np.array([[0.1, 0.5, 0.4], [0.5, 0.1, 0.4]]), 0.99)
    assert mask.tolist() == [True, False]


def test_prepare_image_batch():
    config = pipeline.ProcessingConfig(batch_duration=2, batch_size=5, cut_low_frequency=3, cut_high_frequency=20,
                                       image_normalize=True, save_positive_examples=False, image_size=(32, 24))
    images = [np.full((100, 120, 3), 255, dtype=np.uint8) for _ in range(3)]
    batch, time_info = pipeline.prepare_image_batch(images, 10.0, config)
    assert batch.shape == (3, 24, 32, 3) and batch.dtype == np.float32
    assert batch.max() == 1.0
    assert [(s, e) for _, s, e in time_info] == [(10.0, 10.4), (10.4, 10.8), (10.8, 11.2)]
    assert pipeline.prepare_image_batch([], 0, config)[1] == []


def test_pipeline_writes_predictions(wav_file, tmp_path, fake_model):
    out = tmp_path / "out"
    progress = []
    results = run(wav_file, out, batch_size=4, progress_callback=lambda pct, msg: progress.append(pct))

    stem = wav_file.stem
    # 9 windows in batches of 4 (4 + 4 + 1); even indices of each batch are positive
    assert results == {wav_file.name: 5}
    df = pd.read_csv(out / stem / f"{stem}.wav_predictions.csv")
    assert list(df.columns) == list(pipeline.PREDICTION_COLUMNS)
    assert df["initial_point"].tolist() == [0.0, 0.8, 1.6, 2.4, 3.2]
    assert df["finish_point"].tolist() == [0.4, 1.2, 2.0, 2.8, 3.6]
    assert (df["file_name"] == wav_file.name).all()
    assert len(list((out / stem / "positive").glob("*.jpg"))) == 5
    assert progress == [100.0]


def test_pipeline_skips_processed_files(wav_file, tmp_path, fake_model):
    out = tmp_path / "out"
    assert run(wav_file, out) != {}
    assert run(wav_file, out) == {}


def test_pipeline_respects_time_range_and_threshold(wav_file, tmp_path, fake_model):
    results = run(wav_file, tmp_path / "a", start_time=1.0, end_time=2.0, save_positives=False)
    assert results == {wav_file.name: 1}  # a single 0.8 s window fits in [1, 2]
    assert not (tmp_path / "a" / wav_file.stem / "positive").exists()

    assert run(wav_file, tmp_path / "b", binary_threshold=0.95) == {wav_file.name: 0}
    df = pd.read_csv(tmp_path / "b" / wav_file.stem / f"{wav_file.stem}.wav_predictions.csv")
    assert df.empty and list(df.columns) == list(pipeline.PREDICTION_COLUMNS)


def test_failed_file_is_not_marked_as_processed(wav_file, tmp_path, fake_model):
    broken = wav_file.parent / "broken.wav"
    broken.write_bytes(b"this is not audio")
    out = tmp_path / "out"

    results = run(wav_file, out)
    assert results["broken.wav"] is None
    assert results[wav_file.name] == 5
    assert not (out / "broken" / "broken.wav_predictions.csv").exists()


def test_specific_files_and_non_audio_are_filtered(wav_file, tmp_path, fake_model):
    other = wav_file.parent / "other.wav"
    wavfile.write(str(other), SAMPLE_RATE, make_signal(duration=1.0))
    (wav_file.parent / "notes.txt").write_text("not audio")

    results = run(wav_file, tmp_path / "out", specific_files=["other.wav", "missing.wav", "notes.txt", ""])
    assert list(results) == ["other.wav"]


def test_missing_recordings_folder_raises(tmp_path, fake_model):
    with pytest.raises(FileNotFoundError):
        pipeline.process_predict_extract(str(tmp_path / "nope"), str(tmp_path / "out"))


def test_cli_end_to_end_with_extraction(wav_file, tmp_path, fake_model, monkeypatch):
    pytest.importorskip("soundfile")
    import main

    monkeypatch.setattr(main, "CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setattr(main, "save_config", lambda config, config_path=None: None)
    model = tmp_path / "model.keras"
    model.write_bytes(b"")
    out = tmp_path / "out"

    code = main.main(["--recordings", str(wav_file.parent), "--saving_folder", str(out),
                      "--model_path", str(model), "--end_time", "-1", "--batch_size", "50",
                      "--max_workers", "1", "--save_p", "--extract"])
    assert code == 0
    assert (out / wav_file.stem / f"{wav_file.stem}.wav_predictions.csv").exists()
    assert (out / "extracted_audio").is_dir()
