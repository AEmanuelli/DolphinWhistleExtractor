import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("sklearn")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "Train_model"))

import torch_models  # noqa: E402  (Predict_and_extract is on sys.path, see conftest.py)
import train_openwhistle as tr  # noqa: E402


def fake_split(n, seed):
    """Images where whistles are a bright diagonal line, noise is uniform: learnable in a few steps."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 2
    images = rng.integers(0, 60, (n, 224, 224)).astype(np.uint8)
    for image, label in zip(images, labels):
        if label:
            offset = rng.integers(0, 100)
            for i in range(120):
                image[180 - i, offset + i] = 255
    return images, labels.astype(np.int64), np.array([f"rec{i % 4}" for i in range(n)])


def test_augment_keeps_shape_and_time_shift_permutes_columns():
    x = torch.rand(4, 1, 224, 224)
    shifted = tr.augment(x, {"time_shift": 0.25})
    assert shifted.shape == x.shape
    # a circular time shift only moves columns: each row keeps its values
    assert torch.allclose(shifted.sort(dim=3).values, x.sort(dim=3).values)
    masked = tr.augment(x, {"time_masks": 2, "freq_masks": 2, "freq_shift": 8})
    assert masked.shape == x.shape and torch.isfinite(masked).all()


def test_paper_checkpoint_without_arch_is_the_vgg16():
    model = torch_models.build_model(torch_models.PAPER_VGG)
    assert {"fc1.weight", "fc2.weight", "out.weight"} <= set(model.state_dict())


def test_unknown_architecture():
    with pytest.raises(ValueError):
        torch_models.build_model("nope")


def test_train_run_writes_a_checkpoint_the_pipeline_can_load(tmp_path):
    data = tr.Data(train=fake_split(96, 0), validation=fake_split(32, 1))
    config = {"arch": "mobilenet_v2", "pretrained": False, "epochs": 2, "batch_size": 16, "lr": 1e-3,
              "time_shift": 0.1, "time_masks": 1, "freq_masks": 1, "mixup": 0.2}
    summary = tr.train_run(config, str(tmp_path / "run"), data, device="cpu", log=lambda message: None)
    assert summary["epochs_run"] == 2 and 0 <= summary["best"]["val_accuracy"] <= 1
    assert os.path.exists(tmp_path / "run" / "best.pt") and not os.path.exists(tmp_path / "run" / "last.pt")
    # a finished run is not trained again
    assert tr.train_run(config, str(tmp_path / "run"), data, device="cpu")["best"] == summary["best"]

    from predict_and_extract_online import TorchWhistleDetector, load_detector

    detector = load_detector(str(tmp_path / "run" / "best.pt"))
    assert isinstance(detector, TorchWhistleDetector) and detector.image_size == (224, 224)
    images = data.validation[0][:8]
    batch = np.repeat(images[..., np.newaxis], 3, axis=3).astype(np.float32)
    pipeline = detector.predict(batch)[:, 1]

    model, metadata = torch_models.load_checkpoint(str(tmp_path / "run" / "best.pt"))
    assert metadata["arch"] == "mobilenet_v2"
    mean = torch.tensor(torch_models.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(torch_models.IMAGENET_STD).view(1, 3, 1, 1)
    training = tr.evaluate_scores(model, images, 8, "cpu", mean, std)
    np.testing.assert_allclose(pipeline, training, atol=1e-5)


def test_interrupted_run_resumes(tmp_path):
    data = tr.Data(train=fake_split(32, 0), validation=fake_split(16, 1))
    config = {"arch": "mobilenet_v2", "pretrained": False, "epochs": 2, "batch_size": 16}
    out = str(tmp_path / "run")

    def interrupt_at_epoch_2(message):
        if message.strip().startswith("epoch 2 step"):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        tr.train_run(config, out, data, device="cpu", log=interrupt_at_epoch_2)
    assert os.path.exists(os.path.join(out, "last.pt")) and not os.path.exists(os.path.join(out, "summary.json"))

    logs = []
    summary = tr.train_run(config, out, data, device="cpu", log=logs.append)
    assert any("resumed after epoch 1" in line for line in logs)
    assert summary["epochs_run"] == 2 and [row["epoch"] for row in _history(out)] == [1, 2]


def _history(out):
    import csv

    with open(os.path.join(out, "history.csv")) as f:
        return [{"epoch": int(row["epoch"])} for row in csv.DictReader(f)]


def test_model_trained_at_lower_resolution_sees_the_same_images_in_the_pipeline(tmp_path):
    data = tr.Data(train=fake_split(32, 0), validation=fake_split(16, 1))
    config = {"arch": "mobilenet_v2", "pretrained": False, "epochs": 1, "batch_size": 16, "img_size": 112}
    tr.train_run(config, str(tmp_path / "run"), data, device="cpu", log=lambda message: None)

    from predict_and_extract_online import ProcessingConfig, load_detector, prepare_image_batch

    detector = load_detector(str(tmp_path / "run" / "best.pt"))
    assert detector.image_size == (112, 112)
    images = data.validation[0][:8]
    processing = ProcessingConfig(batch_duration=60, batch_size=8, cut_low_frequency=2, cut_high_frequency=22,
                                  image_normalize=False, save_positive_examples=False,
                                  image_size=detector.image_size)
    batch, _ = prepare_image_batch(list(images), 0.0, processing)  # 224x224 images resized by OpenCV
    pipeline = detector.predict(batch)[:, 1]

    model, _ = torch_models.load_checkpoint(str(tmp_path / "run" / "best.pt"))
    mean = torch.tensor(torch_models.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(torch_models.IMAGENET_STD).view(1, 3, 1, 1)
    training = tr.evaluate_scores(model, images, 8, "cpu", mean, std, size=112)
    # OpenCV rounds the resized uint8 pixels, PyTorch does not: allow a small difference
    np.testing.assert_allclose(pipeline, training, atol=2e-2)


@pytest.mark.parametrize("arch", ["tinycnn_8", "tinycnn_16", "mobilenet_v3_small"])
def test_small_models_and_their_cost(arch):
    model = torch_models.build_model(arch)
    for size in (224, 112, 64):
        assert model.eval()(torch.zeros(2, 3, size, size)).shape == (2, 2)
    cost = torch_models.model_cost(model, size=112, repeats=3)
    assert cost["params"] == sum(p.numel() for p in model.parameters()) and cost["macs"] > 0 and cost["cpu_ms"] > 0
    if arch == "tinycnn_16":
        assert 20_000 < cost["params"] < 80_000


def test_tiny_cnn_reads_one_channel():
    model = torch_models.build_model("tinycnn_8").eval()
    x = torch.rand(1, 3, 64, 64)
    y = x.clone()
    y[:, 1:] = 0  # the other channels are ignored
    assert torch.equal(model(x), model(y))
