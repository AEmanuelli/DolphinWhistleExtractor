import os
import sys

import numpy as np
import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Train_model"))

import evaluate_models as ev


def test_compute_metrics():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.6, 0.2, 0.7, 0.3, 0.2, 0.1])
    m = ev.compute_metrics(labels, scores, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 3)
    assert m["accuracy"] == 0.75 and m["recall"] == 0.75 and m["precision"] == 0.75
    assert m["specificity"] == 0.75 and m["f1"] == pytest.approx(0.75)
    assert m["roc_auc"] == pytest.approx(12.5 / 16)  # 12 ordered pairs + one tie out of 16

    # a model that never fires: no division by zero
    m = ev.compute_metrics(labels, np.zeros(8))
    assert m["recall"] == 0 and m["precision"] == 0 and m["f1"] == 0 and m["accuracy"] == 0.5


def test_accuracy_interval_brackets_accuracy():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 600)
    scores = np.where(rng.random(600) < 0.9, labels, 1 - labels).astype(float)  # ~90% correct
    recordings = np.array([f"rec{i % 30}" for i in range(600)])
    low, high = ev.accuracy_interval(labels, scores, recordings)
    accuracy = ev.compute_metrics(labels, scores)["accuracy"]
    assert low <= accuracy <= high and 0 < high - low < 0.1


def test_predict_handles_sigmoid_and_two_class_outputs():
    class Detector:
        image_size = (224, 224)

        def __init__(self, two_class):
            self.two_class = two_class

        def predict(self, batch):
            score = batch.mean(axis=(1, 2, 3)) / 255.0
            return np.stack([1 - score, score], axis=1) if self.two_class else score[:, None]

    images = np.stack([np.zeros((224, 224), np.uint8), np.full((224, 224), 255, np.uint8)])
    for two_class in (False, True):
        assert ev.predict(Detector(two_class), images, "raw").tolist() == [0.0, 1.0]
    assert ev.predict(Detector(False), images, "div255").tolist() == pytest.approx([0.0, 1 / 255])

    # a detector that normalises its inputs itself always receives raw 0-255 images
    detector = Detector(False)
    detector.handles_normalization = True
    assert ev.predict(detector, images, "div255").tolist() == [0.0, 1.0]

    # images are resized to the detector's input size
    detector.image_size = (64, 32)
    detector.predict = lambda batch: np.full((len(batch), 1), float(batch.shape[1:3] == (32, 64)))
    assert ev.predict(detector, images).tolist() == [1.0, 1.0]


def test_pipeline_image_is_the_benchmark_image():
    import cv2
    import utils

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "openwhistle_cnn")
    with open(os.path.join(folder, "whistle.wav"), "rb") as f:
        image = ev.pipeline_image(f.read(), utils.PAPER_SPECTROGRAM)
    expected = cv2.imread(os.path.join(folder, "whistle.png"))
    assert np.abs(image.astype(int) - expected.astype(int)).max() <= 1

    with open(os.path.join(folder, "whistle.wav"), "rb") as f:
        legacy = ev.pipeline_image(f.read(), utils.LEGACY_SPECTROGRAM)  # the whole clip is one window
    assert legacy.shape == (875, 1167, 3)
