#!/usr/bin/env python3
"""Evaluate whistle-detection models on the public OpenWhistle-CNN benchmark.

Dataset: https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN
(0.4 s windows at 96 kHz, balanced whistle / noise, session-disjoint splits;
the test split holds 16,708 windows from 261 recordings of 2019-2020).

Each window can be shown to the model in two ways:

* ``--source dataset``  : the 224x224 spectrogram image shipped with the dataset;
* ``--source pipeline`` : the spectrogram computed from the audio window by this
  repository's own pipeline (``Predict_and_extract/utils.py``), i.e. what the
  model actually sees when ``main.py`` or the GUI is used.

Comparing both tells whether the scores measured on a spectrogram dataset carry
over to the detection pipeline.

Examples:
    # all models of models/, test split, dataset spectrograms
    python Train_model/evaluate_models.py

    # one model, pipeline spectrograms, quick run on 2000 windows
    python Train_model/evaluate_models.py --models models/base_MobileNetV2.keras --source pipeline --limit 2000

Requires: tensorflow, pyarrow, pillow, scikit-learn (plus the pipeline requirements).
"""
import argparse
import csv
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


def pipeline_image(wav_bytes, overlap, cut_low_frequency, cut_high_frequency):
    """Spectrogram of one audio window, computed like the detection pipeline does."""
    import utils
    from scipy.io import wavfile

    fs, x = wavfile.read(io.BytesIO(wav_bytes))
    return utils.audio_to_spectrogram_images(
        x, fs, end_time=None, batch_size=1, window_duration=len(x) / fs, stft_overlap=overlap,
        cut_low_frequency=cut_low_frequency, cut_high_frequency=cut_high_frequency)[0]


def load_examples(paths, source, limit, overlap, cut_low_frequency, cut_high_frequency, seed=0):
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
                    image = pipeline_image(row["audio"]["bytes"], overlap, cut_low_frequency, cut_high_frequency)
                    image = cv2.resize(image, (224, 224))[..., 0]
                images.append(image)
                labels.append(row["label"])
                recordings.append(row["recording"])
    return np.array(images), np.array(labels), np.array(recordings)


def predict(model, images, scale, batch_size=128):
    """Whistle score of every image (grayscale uint8 -> 3 channels -> model)."""
    height, width = model.input_shape[1] or 224, model.input_shape[2] or 224
    scores = []
    for i in range(0, len(images), batch_size):
        batch = np.repeat(images[i:i + batch_size, ..., np.newaxis], 3, axis=3).astype(np.float32)
        if (height, width) != batch.shape[1:3]:
            import cv2
            batch = np.array([cv2.resize(image, (width, height)) for image in batch])
        output = np.asarray(model(SCALES[scale](batch), training=False))
        # single sigmoid output, or class 1 of a two-class output
        scores.append(output[:, 0] if output.shape[1] == 1 else output[:, 1])
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
    parser.add_argument("--models", nargs="+", help="Model files (default: every .keras / .h5 file of models/)")
    parser.add_argument("--split", choices=sorted(SPLIT_SHARDS), default="test")
    parser.add_argument("--source", choices=["dataset", "pipeline"], default="dataset",
                        help="Spectrograms shipped with the dataset, or recomputed by this repository's pipeline")
    parser.add_argument("--scale", choices=sorted(SCALES), default="raw",
                        help="Pixel scaling fed to the model (div255 = pipeline option --image_norm)")
    parser.add_argument("--overlap", type=float, default=None,
                        help="STFT overlap fraction for --source pipeline (default: the pipeline's own value)")
    parser.add_argument("--CLF", type=int, default=3, help="Cut low frequency (kHz), --source pipeline")
    parser.add_argument("--CHF", type=int, default=20, help="Cut high frequency (kHz), --source pipeline")
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
    overlap = utils.STFT_OVERLAP if args.overlap is None else args.overlap

    paths = download_split(args.split, args.cache_dir)
    print(f"Preparing {args.source} spectrograms of the {args.split} split ...")
    images, labels, recordings = load_examples(paths, args.source, args.limit, overlap, args.CLF, args.CHF)
    print(f"{len(labels)} windows ({int((labels == 1).sum())} whistle, {int((labels == 0).sum())} noise) "
          f"from {len(set(recordings))} recordings")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    rows = []
    for model_path in model_paths:
        model = tf.keras.models.load_model(model_path, compile=False)
        scores = predict(model, images, args.scale)
        metrics = compute_metrics(labels, scores, args.threshold)
        low, high = accuracy_interval(labels, scores, recordings, args.threshold)
        name = os.path.splitext(os.path.basename(model_path))[0]
        print(f"{name}: accuracy {metrics['accuracy']:.4f} (95% CI {low:.4f}-{high:.4f}), "
              f"precision {metrics['precision']:.4f}, recall {metrics['recall']:.4f}, "
              f"specificity {metrics['specificity']:.4f}, F1 {metrics['f1']:.4f}, ROC AUC {metrics['roc_auc']:.4f}")
        rows.append({"model": name, "split": args.split, "source": args.source, "scale": args.scale,
                     "stft_overlap": overlap if args.source == "pipeline" else "",
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
