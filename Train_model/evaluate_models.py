#!/usr/bin/env python3
"""Evaluate whistle-detection models on the public OpenWhistle-CNN benchmark.

Dataset: https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN
(0.4 s windows at 96 kHz, balanced whistle / noise, session-disjoint splits;
the test split holds 16,708 windows from 261 recordings of 2019-2020).

Each window can be shown to the model in two ways:

* ``--source dataset``  : the 224x224 spectrogram image shipped with the dataset;
* ``--source pipeline`` : the spectrogram computed from the audio window by this
  repository's own pipeline (``Predict_and_extract/utils.py``), i.e. what the
  model actually sees when ``main.py`` or the GUI is used. With the default
  "paper" settings both are the same images (up to one grey level on a few pixels).

Models can be Keras files (.keras / .h5) or the PyTorch VGG16 checkpoint of the
paper (.pt, https://huggingface.co/dolphinteam/OpenWhistle-CNN-VGG16).

Examples:
    # all models of models/, test split, dataset spectrograms
    python Train_model/evaluate_models.py

    # the detector published with the paper
    python Train_model/evaluate_models.py --models model_vgg_final_best.pt

    # one model, pipeline spectrograms, quick run on 2000 windows
    python Train_model/evaluate_models.py --models models/base_MobileNetV2.keras --source pipeline --limit 2000

Requires: pyarrow, pillow, scikit-learn, and tensorflow and/or torch + torchvision
(plus the pipeline requirements).
"""
import argparse
import csv
import dataclasses
import glob
import io
import os
import sys
import urllib.request

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "Predict_and_extract"))

DATASET_URL = "https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN/resolve/main/data"
SPLIT_SHARDS = {"test": 4, "validation": 2, "train": 11}
SCALES = {
    "raw": lambda batch: batch,                 # 0-255, what the pipeline feeds by default
    "div255": lambda batch: batch / 255.0,      # pipeline option --image_norm
}


def download_split(split, cache_dir):
    """Download the parquet shards of a split (skipped when already cached)."""
    os.makedirs(cache_dir, exist_ok=True)
    count = SPLIT_SHARDS[split]
    paths = []
    for i in range(count):
        name = f"{split}-{i:05d}-of-{count:05d}.parquet"
        path = os.path.join(cache_dir, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(f"{DATASET_URL}/{name}", path + ".part")
            os.replace(path + ".part", path)
        paths.append(path)
    return paths


def pipeline_image(wav_bytes, settings):
    """Spectrogram of one audio clip, computed like the detection pipeline does (the whole clip is one window)."""
    import utils
    from scipy.io import wavfile

    fs, x = wavfile.read(io.BytesIO(wav_bytes))
    fs, x = utils.prepare_audio(x, fs, settings)
    settings = dataclasses.replace(settings, window_duration=len(x) / fs)
    return utils.spectrogram_image(x, fs, settings)


def load_examples(paths, source, limit, settings, seed=0):
    """Return (images uint8 [n, 224, 224], labels, recordings) for a split."""
    import cv2
    import pyarrow.parquet as pq
    from PIL import Image

    columns = ["label", "recording", "spectrogram" if source == "dataset" else "audio"]
    files = [pq.ParquetFile(path) for path in paths]
    total = sum(f.metadata.num_rows for f in files)
    keep = None
    if limit and limit < total:  # pick the subset first: only those windows are decoded
        keep = set(np.random.default_rng(seed).choice(total, limit, replace=False).tolist())

    images, labels, recordings = [], [], []
    index = 0
    for parquet in files:
        for group in range(parquet.num_row_groups):
            for row in parquet.read_row_group(group, columns=columns).to_pylist():
                index += 1
                if keep is not None and index - 1 not in keep:
                    continue
                if source == "dataset":
                    image = np.array(Image.open(io.BytesIO(row["spectrogram"]["bytes"])).convert("L"))
                else:
                    image = cv2.resize(pipeline_image(row["audio"]["bytes"], settings), (224, 224))[..., 0]
                images.append(image)
                labels.append(row["label"])
                recordings.append(row["recording"])
    return np.array(images), np.array(labels), np.array(recordings)


def predict(detector, images, scale="raw", batch_size=64):
    """
    Whistle score of every image (grayscale uint8 [n, h, w] -> 3 channels -> detector).

    ``detector`` exposes ``predict(batch) -> np.ndarray`` and ``image_size`` (width, height),
    like the detectors of ``predict_and_extract_online``.
    """
    import cv2
    from predict_and_extract_online import positive_scores

    width, height = detector.image_size
    if getattr(detector, "handles_normalization", False):
        scale = "raw"  # the detector normalises 0-255 images itself
    scores = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        if batch.shape[1:3] != (height, width):
            batch = np.array([cv2.resize(image, (width, height)) for image in batch])
        batch = np.repeat(batch[..., np.newaxis], 3, axis=3).astype(np.float32)
        scores.append(positive_scores(detector.predict(SCALES[scale](batch)), 0.5)[1])
    return np.concatenate(scores)


def compute_metrics(labels, scores, threshold=0.5):
    """Standard binary metrics; whistle is the positive class."""
    from sklearn.metrics import roc_auc_score

    predicted = scores >= threshold
    positives, negatives = labels == 1, labels == 0
    tp, fp = int((predicted & positives).sum()), int((predicted & negatives).sum())
    fn, tn = int((~predicted & positives).sum()), int((~predicted & negatives).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(labels),
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "roc_auc": roc_auc_score(labels, scores) if positives.any() and negatives.any() else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def accuracy_interval(labels, scores, recordings, threshold=0.5, n_boot=1000, seed=0):
    """95% interval of the accuracy, resampling whole recordings (windows of a recording are correlated)."""
    correct = (scores >= threshold) == (labels == 1)
    names, inverse = np.unique(recordings, return_inverse=True)
    hits = np.bincount(inverse, weights=correct, minlength=len(names))
    sizes = np.bincount(inverse, minlength=len(names))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(n_boot, len(names)))
    accuracies = hits[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return float(np.percentile(accuracies, 2.5)), float(np.percentile(accuracies, 97.5))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+",
                        help="Model files: Keras (.keras / .h5) or PyTorch checkpoint of the paper's VGG16 (.pt). "
                             "Default: every Keras model of models/")
    parser.add_argument("--split", choices=sorted(SPLIT_SHARDS), default="test")
    parser.add_argument("--source", choices=["dataset", "pipeline"], default="dataset",
                        help="Spectrograms shipped with the dataset, or recomputed by this repository's pipeline")
    parser.add_argument("--scale", choices=sorted(SCALES), default="raw",
                        help="Pixel scaling fed to the model (div255 = pipeline option --image_norm)")
    parser.add_argument("--spectrogram", choices=["paper", "legacy"], default="paper",
                        help="Spectrogram settings for --source pipeline (see Predict_and_extract/utils.py)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate a random subset of this many windows")
    parser.add_argument("--cache_dir", default=os.path.join(REPO_ROOT, "openwhistle_cnn_cache"),
                        help="Where the dataset files are downloaded")
    parser.add_argument("--output_csv", help="Also write the metrics to this CSV file")
    args = parser.parse_args(argv)

    model_paths = args.models or sorted(
        glob.glob(os.path.join(REPO_ROOT, "models", "*.keras")) + glob.glob(os.path.join(REPO_ROOT, "models", "*.h5")))
    if not model_paths:
        parser.error("No model found")

    import utils
    from predict_and_extract_online import load_detector
    settings = utils.SPECTROGRAM_PRESETS[args.spectrogram]

    paths = download_split(args.split, args.cache_dir)
    print(f"Preparing {args.source} spectrograms of the {args.split} split ...")
    images, labels, recordings = load_examples(paths, args.source, args.limit, settings)
    print(f"{len(labels)} windows ({int((labels == 1).sum())} whistle, {int((labels == 0).sum())} noise) "
          f"from {len(set(recordings))} recordings")

    rows = []
    for model_path in model_paths:
        scores = predict(load_detector(model_path), images, args.scale)
        metrics = compute_metrics(labels, scores, args.threshold)
        low, high = accuracy_interval(labels, scores, recordings, args.threshold)
        name = os.path.splitext(os.path.basename(model_path))[0]
        print(f"{name}: accuracy {metrics['accuracy']:.4f} (95% CI {low:.4f}-{high:.4f}), "
              f"precision {metrics['precision']:.4f}, recall {metrics['recall']:.4f}, "
              f"specificity {metrics['specificity']:.4f}, F1 {metrics['f1']:.4f}, ROC AUC {metrics['roc_auc']:.4f}")
        rows.append({"model": name, "split": args.split, "source": args.source, "scale": args.scale,
                     "spectrogram": args.spectrogram if args.source == "pipeline" else "dataset",
                     "threshold": args.threshold, **metrics, "accuracy_ci_low": low, "accuracy_ci_high": high})

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Metrics written to {args.output_csv}")
    return rows


if __name__ == "__main__":
    main()
