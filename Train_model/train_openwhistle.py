#!/usr/bin/env python3
"""Train whistle detectors on the OpenWhistle-CNN dataset (PyTorch).

Dataset: https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN — 0.4 s windows at 96 kHz,
balanced whistle / noise, split by recording session: train 53,828 windows (2021-2024),
validation 5,980 (2021-2024), test 16,708 (2019-2020). The 224x224 spectrograms shipped with the
dataset are the images the detection pipeline computes (``--spectrogram paper``), so a model
trained here can be used directly with ``main.py --model_path best.pt``.

Protocol: the checkpoint of a run is the epoch with the lowest validation loss (as in the paper);
runs of a sweep are ranked on the validation accuracy of that checkpoint. The test split is only
used by the ``test`` command, once a model has been chosen.

Commands:
    prepare   download the splits and decode the images once (uint8 .npy files, ~3.6 GB)
    train     one training run
    sweep     a list of runs (JSON file), resumable, ranked on the validation split
    test      metrics of checkpoints on the test split

Examples:
    python Train_model/train_openwhistle.py prepare
    python Train_model/train_openwhistle.py train --arch mobilenet_v2 --epochs 8 --out runs/mnv2
    python Train_model/train_openwhistle.py sweep Train_model/sweeps/openwhistle_sweep.json --out runs/sweep
    python Train_model/train_openwhistle.py test runs/sweep/<run>/best.pt

Requires: torch, torchvision, pyarrow, pillow, scikit-learn. A GPU is strongly recommended.
"""
import argparse
import csv
import dataclasses
import io
import json
import math
import os
import random
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "Predict_and_extract"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Train_model"))

import evaluate_models as em  # noqa: E402  (download, metrics; imports only numpy at module level)

DEFAULT_CACHE = os.path.join(REPO_ROOT, "openwhistle_cnn_cache")


# --------------------------------------------------------------------------- data

def prepare_split(split, cache_dir):
    """Decode the spectrograms of a split once: <cache>/<split>_{images,labels,recordings}.npy."""
    import pyarrow.parquet as pq
    from PIL import Image

    target = os.path.join(cache_dir, f"{split}_images.npy")
    if os.path.exists(target):
        return
    paths = em.download_split(split, cache_dir)
    files = [pq.ParquetFile(path) for path in paths]
    total = sum(f.metadata.num_rows for f in files)
    images = np.lib.format.open_memmap(target + ".part", mode="w+", dtype=np.uint8, shape=(total, 224, 224))
    labels, recordings, index = [], [], 0
    for parquet in files:
        for group in range(parquet.num_row_groups):
            for row in parquet.read_row_group(group, columns=["label", "recording", "spectrogram"]).to_pylist():
                image = Image.open(io.BytesIO(row["spectrogram"]["bytes"])).convert("L")
                if image.size != (224, 224):
                    image = image.resize((224, 224), Image.NEAREST)
                images[index] = np.asarray(image)
                labels.append(row["label"])
                recordings.append(row["recording"])
                index += 1
        print(f"  {split}: {index}/{total}", flush=True)
    images.flush()
    del images
    np.save(os.path.join(cache_dir, f"{split}_labels.npy"), np.array(labels, dtype=np.int64))
    np.save(os.path.join(cache_dir, f"{split}_recordings.npy"), np.array(recordings))
    os.replace(target + ".part", target)


def prepare_frontends(split, cache_dir, names):
    """
    Compute the images of audio front-ends (see Predict_and_extract/frontends.py) from the audio of a split,
    in one pass over the parquet files: <cache>/<split>_<front-end>_images.npy, same order as the paper images.
    """
    import pyarrow.parquet as pq
    import soundfile
    import frontends

    names = [name for name in names if not os.path.exists(os.path.join(cache_dir, f"{split}_{name}_images.npy"))]
    if not names:
        return
    prepare_split(split, cache_dir)
    labels = np.load(os.path.join(cache_dir, f"{split}_labels.npy"))
    files = [pq.ParquetFile(path) for path in em.download_split(split, cache_dir)]
    targets = {name: np.lib.format.open_memmap(os.path.join(cache_dir, f"{split}_{name}_images.npy.part"),
                                               mode="w+", dtype=np.uint8, shape=(len(labels), 224, 224))
               for name in names}
    index = 0
    for parquet in files:
        for group in range(parquet.num_row_groups):
            rows = parquet.read_row_group(group, columns=["label", "audio"]).to_pylist()
            windows, rates = [], set()
            for row in rows:
                samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
                windows.append(samples[:, 0])
                rates.add(rate)
            if len(rates) != 1:
                raise ValueError(f"{split}: windows of different sample rates in a row group")
            length = frontends.WINDOW_SAMPLES  # a few windows are a sample or so short or long: pad / crop
            windows = [np.pad(w[:length], (0, max(0, length - len(w)))) for w in windows]
            if [row["label"] for row in rows] != labels[index:index + len(rows)].tolist():
                raise ValueError(f"{split}: audio rows are not in the order of the prepared images")
            windows, rate = np.stack(windows), rates.pop()
            power = frontends.band_power(windows)  # shared by the front-ends
            for name in names:
                targets[name][index:index + len(rows)] = frontends.window_images(name, windows, rate, power)
            index += len(rows)
        print(f"  {split} {'/'.join(names)}: {index}/{len(labels)}", flush=True)
    for name, images in targets.items():
        images.flush()
        del images
        os.replace(os.path.join(cache_dir, f"{split}_{name}_images.npy.part"),
                   os.path.join(cache_dir, f"{split}_{name}_images.npy"))


def load_split(split, cache_dir, in_memory=True, frontend="paper"):
    """
    (images uint8 [n, 224, 224], labels int64, recordings) of a prepared split, for a front-end.

    ``in_memory`` reads the images once, sequentially (train: 2.7 GB of RAM). Training draws random batches,
    and random reads in a memory-mapped file are very slow on a hard disk.
    """
    prepare_split(split, cache_dir)
    name = f"{split}_images.npy"
    if frontend != "paper":
        prepare_frontends(split, cache_dir, [frontend])
        name = f"{split}_{frontend}_images.npy"
    started = time.time()
    images = np.load(os.path.join(cache_dir, name), mmap_mode=None if in_memory else "r")
    if in_memory:
        print(f"{split} ({frontend}): {images.nbytes / 2**30:.1f} GB loaded in {time.time() - started:.0f} s",
              flush=True)
    labels = np.load(os.path.join(cache_dir, f"{split}_labels.npy"))
    recordings = np.load(os.path.join(cache_dir, f"{split}_recordings.npy"))
    return images, labels, recordings


class Batches:
    """
    Batches of (uint8 images, targets) read by a background worker pool. ``indices`` restricts the batches
    to these windows (a subset without copying the images); ``targets`` is indexed like ``images``.
    """

    def __init__(self, images, labels, batch_size, shuffle, seed=0, workers=3, indices=None):
        import torch
        from torch.utils.data import DataLoader, Dataset

        class _Dataset(Dataset):
            def __init__(self, order):
                self.order = order

            def __len__(self):
                return len(self.order)

            def __getitem__(self, i):
                index = np.sort(self.order[i])  # sorted reads are faster on a memmap
                return torch.from_numpy(np.ascontiguousarray(images[index])), torch.from_numpy(labels[index])

        rng = np.random.default_rng(seed)
        indices = np.arange(len(labels)) if indices is None else np.asarray(indices)
        order = rng.permutation(indices) if shuffle else indices
        chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        self.loader = DataLoader(_Dataset(chunks), batch_size=None, shuffle=False, num_workers=workers,
                                 pin_memory=True, persistent_workers=False)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


# --------------------------------------------------------------------------- augmentation (on the GPU)

def augment(x, config, generator=None):
    """
    Spectrogram augmentations on a float batch (n, 1, H, W), time on the width, frequency on the height.

    - time_shift: circular shift along time, up to this fraction of the window;
    - freq_shift: shift along frequency, up to this many pixels (edges replicated);
    - time_masks / freq_masks: number of SpecAugment masks, each up to mask_width pixels, set to the image mean.
    """
    import torch

    n, _, height, width = x.shape
    device = x.device
    if config.get("time_shift", 0) > 0:
        max_shift = int(config["time_shift"] * width)
        shifts = torch.randint(-max_shift, max_shift + 1, (n,), device=device)
        columns = (torch.arange(width, device=device)[None, :] - shifts[:, None]) % width
        x = torch.gather(x, 3, columns[:, None, None, :].expand(n, 1, height, width))
    if config.get("freq_shift", 0) > 0:
        max_shift = int(config["freq_shift"] * width / 224)
        shifts = torch.randint(-max_shift, max_shift + 1, (n,), device=device)
        rows = (torch.arange(height, device=device)[None, :] - shifts[:, None]).clamp(0, height - 1)
        x = torch.gather(x, 2, rows[:, None, :, None].expand(n, 1, height, width))
    scale = width / 224  # shifts and masks are given in pixels of the 224x224 images
    mask_width = int(config.get("mask_width", 20) * scale)
    for axis, count in ((3, config.get("time_masks", 0)), (2, config.get("freq_masks", 0))):
        size = x.shape[axis]
        for _ in range(count):
            widths = torch.randint(0, mask_width + 1, (n,), device=device)
            starts = (torch.rand(n, device=device) * (size - widths)).long()
            position = torch.arange(size, device=device)[None, :]
            mask = (position >= starts[:, None]) & (position < (starts + widths)[:, None])
            shape = (n, 1, 1, size) if axis == 3 else (n, 1, size, 1)
            x = torch.where(mask.view(shape), x.mean(dim=(2, 3), keepdim=True), x)
    return x


def to_model_input(images_uint8, device, mean, std, size=224):
    """
    uint8 (n, H, W) -> float (n, 1, size, size) in 0-1, and the function that normalises it for the model
    (3 channels, ImageNet mean / std), like TorchWhistleDetector. A smaller ``size`` resizes bilinearly,
    as cv2.resize does in the pipeline (the checkpoint records the size, the pipeline resizes to it).
    """
    import torch.nn.functional as F

    x = images_uint8.to(device, non_blocking=True).float().div_(255.0).unsqueeze(1)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x, (lambda batch: (batch.expand(-1, 3, -1, -1) - mean) / std)


# --------------------------------------------------------------------------- training

DEFAULTS = {
    "arch": "mobilenet_v2", "pretrained": True, "epochs": 8, "batch_size": 32, "lr": 3e-4,
    "weight_decay": 1e-4, "warmup_epochs": 0.3, "label_smoothing": 0.0, "mixup": 0.0,
    "time_shift": 0.0, "freq_shift": 0, "time_masks": 0, "freq_masks": 0, "mask_width": 20,
    "patience": 4, "seed": 7, "amp": False, "train_fraction": 1.0, "img_size": 224,
    # input images: "paper" (the dataset's spectrograms) or an audio front-end of frontends.py
    "frontend": "paper",
    # robustness to label noise: generalised cross-entropy (Zhang & Sabuncu, 2018), weight averaging
    "loss": "ce", "gce_q": 0.7, "ema": 0.0,
    # dataset diagnostics: k-fold cross-validation over the train recordings (out-of-fold scores)
    "cv_folds": 0, "cv_fold": 0,
    # cleaning: leave out the train windows whose out-of-fold score contradicts the label by >= threshold
    "exclude_runs": "", "exclude_threshold": 0.9,
    # leave out the train recordings whose name contains one of these comma-separated strings (e.g. "channel_")
    "exclude_recordings": "",
    # soft targets: out-of-fold scores ("oof") or a trained teacher's scores ("teacher"), distillation weight
    "soft_source": "", "soft_runs": "", "soft_weight": 0.0, "distill_temperature": 2.0,
}


@dataclasses.dataclass
class Data:
    train: tuple
    validation: tuple


class SplitCache:
    """The train and validation images of one front-end at a time (the train images take 2.7 GB of RAM)."""

    def __init__(self, cache_dir):
        self.cache_dir, self.frontend, self.data = cache_dir, None, None

    def __call__(self, frontend="paper"):
        if frontend != self.frontend:
            self.data = None  # free the previous front-end before loading the next one
            import gc
            gc.collect()
            self.data = Data(train=load_split("train", self.cache_dir, frontend=frontend),
                             validation=load_split("validation", self.cache_dir, frontend=frontend))
            self.frontend = frontend
        return self.data


def evaluate_scores(model, images, batch_size, device, mean, std, size=224, indices=None):
    """Whistle probability of every image, or of ``indices`` only (the same computation as the pipeline)."""
    import torch

    scores = []
    model.eval()
    with torch.inference_mode():
        for batch, _ in Batches(images, np.zeros(len(images), dtype=np.int64), batch_size * 2, shuffle=False,
                                indices=indices):
            x, normalise = to_model_input(batch, device, mean, std, size)
            scores.append(torch.softmax(model(normalise(x)).float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def binary_log_loss(labels, scores):
    p = np.clip(scores, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def recording_type(name):
    """OpenWhistle recording types: multichannel recordings (2023-2024) end in channel_<n>, older ones do not."""
    import re

    match = re.search(r"channel_\d+", str(name))
    return match.group(0) if match else "single_channel"


def accuracy_by_recording_type(labels, scores, recordings):
    """Accuracy (threshold 0.5) of each recording type: the splits mix recording set-ups of different difficulty."""
    types = np.array([recording_type(r) for r in recordings])
    return {kind: float(((scores[types == kind] >= 0.5) == (labels[types == kind] == 1)).mean())
            for kind in sorted(set(types))}


def recording_folds(recordings, n_folds, seed=0):
    """
    Fold of every window for a k-fold split by recording: a recording is never split between folds,
    and folds hold similar numbers of windows (largest recordings first, each to the smallest fold).
    """
    names, counts = np.unique(recordings, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(names))
    order = sorted(order, key=lambda i: -counts[i])  # stable: random order among equal sizes
    sizes = np.zeros(n_folds, dtype=np.int64)
    fold_of = {}
    for i in order:
        fold = int(np.argmin(sizes))
        fold_of[names[i]] = fold
        sizes[fold] += counts[i]
    return np.array([fold_of[r] for r in recordings])


def resolve_runs(names, out_dir):
    """Run folders named in a config (comma-separated; relative names are siblings of ``out_dir``)."""
    parent = os.path.dirname(os.path.abspath(out_dir))
    return [name if os.path.isabs(name) else os.path.join(parent, name)
            for name in (part.strip() for part in names.split(",")) if name]


def gather_oof(run_dirs, n):
    """Out-of-fold whistle scores of the train split, assembled from the folds of a cross-validation."""
    scores = np.full(n, np.nan, dtype=np.float32)
    for run_dir in run_dirs:
        indices = np.load(os.path.join(run_dir, "oof_indices.npy"))
        scores[indices] = np.load(os.path.join(run_dir, "oof_scores.npy"))
    if np.isnan(scores).any():  # RuntimeError: the sweep reports it and goes on with the next run
        raise RuntimeError(f"out-of-fold scores cover {int((~np.isnan(scores)).sum())} of {n} train windows "
                           f"in {run_dirs}")
    return scores


def run_scores(run_dir, split, images, device, batch_size=64):
    """Whistle scores of a trained run on a split (cached in the run folder as <split>_scores.npy)."""
    import torch
    import torch_models

    path = os.path.join(run_dir, f"{split}_scores.npy")
    if os.path.exists(path):
        return np.load(path)
    model, metadata = torch_models.load_checkpoint(os.path.join(run_dir, "best.pt"))
    mean = torch.tensor(torch_models.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(torch_models.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    scores = evaluate_scores(model.to(device), images, batch_size, device, mean, std,
                             int(metadata.get("img_size", 224) or 224)).astype(np.float32)
    np.save(path, scores)
    return scores


def hard_loss(config):
    """Loss on the labels: cross-entropy (with optional label smoothing) or generalised cross-entropy."""
    import torch

    if config["loss"] == "gce":  # (1 - p_y^q) / q: bounded, so confidently wrong labels weigh less
        q = config["gce_q"]

        def gce(logits, y):
            p = torch.softmax(logits.float(), dim=1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-7)
            return ((1 - p ** q) / q).mean()
        return gce
    if config["loss"] != "ce":
        raise ValueError(f"Unknown loss {config['loss']!r} (ce or gce)")
    return torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])


def distillation_loss(logits, soft, temperature):
    """Cross-entropy to soft whistle probabilities, both sides softened by ``temperature`` (x T^2)."""
    import torch

    target = torch.sigmoid(torch.logit(soft.float().clamp(1e-4, 1 - 1e-4)) / temperature)
    log_p = torch.log_softmax(logits.float() / temperature, dim=1)
    return -(target * log_p[:, 1] + (1 - target) * log_p[:, 0]).mean() * temperature ** 2


class WandbLogger:
    """Optional Weights & Biases logging of a run (resumes the same W&B run after an interruption)."""

    def __init__(self, project, out_dir, config, log=print):
        self.run = None
        if not project:
            return
        try:
            import wandb

            id_path = os.path.join(out_dir, "wandb_id.txt")
            if os.path.exists(id_path):
                with open(id_path) as f:
                    run_id = f.read().strip()
            else:
                run_id = wandb.util.generate_id()
                with open(id_path, "w") as f:
                    f.write(run_id)
            self.run = wandb.init(project=project, name=os.path.basename(os.path.normpath(out_dir)), id=run_id,
                                  resume="allow", config=config, dir=out_dir, reinit=True)
        except Exception as exc:  # no wandb, offline, not logged in...: train anyway
            log(f"  W&B logging disabled: {exc}")

    def log(self, values, step=None):
        if self.run is not None:
            self.run.log(values, step=step)

    def finish(self, summary):
        if self.run is not None:
            self.run.summary.update(summary)
            self.run.finish()


def train_run(config, out_dir, data: Data, device="cuda", log=print, wandb_project=None):
    """Train one model; writes best.pt, last.pt (resume state), history.csv and summary.json in out_dir."""
    import torch
    import torch_models

    config = {**DEFAULTS, **config}
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)

    tracker = WandbLogger(wandb_project, out_dir, config, log)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = True  # fixed input size: let cuDNN pick the fastest kernels

    train_images, train_labels, train_recordings = data.train
    train_idx = np.arange(len(train_labels))  # the windows this run trains on (no copy of the images)
    notes, held_out = {}, None
    if config["cv_folds"]:
        folds = recording_folds(train_recordings, config["cv_folds"])
        held_out = np.flatnonzero(folds == config["cv_fold"])
        train_idx = np.flatnonzero(folds != config["cv_fold"])
        notes["held_out_windows"] = int(len(held_out))
        log(f"  fold {config['cv_fold']} of {config['cv_folds']}: {len(train_idx)} train windows, "
            f"{len(held_out)} held out (whole recordings)")
    if config["exclude_runs"]:
        oof = gather_oof(resolve_runs(config["exclude_runs"], out_dir), len(train_labels))
        suspect = np.abs(oof - train_labels) >= config["exclude_threshold"]
        train_idx = train_idx[~suspect[train_idx]]
        notes["excluded_windows"] = int(suspect.sum())
        log(f"  left out {int(suspect.sum())} train windows whose out-of-fold score contradicts the label "
            f"by >= {config['exclude_threshold']}")
    if config["exclude_recordings"]:
        patterns = [p.strip() for p in config["exclude_recordings"].split(",") if p.strip()]
        dropped = np.array([any(p in str(r) for p in patterns) for r in train_recordings])
        train_idx = train_idx[~dropped[train_idx]]
        notes["excluded_recording_windows"] = int(dropped.sum())
        log(f"  left out {int(dropped.sum())} train windows of recordings matching {patterns}")
    if config["train_fraction"] < 1:  # learning-curve / quick runs: a fixed random subset
        train_idx = np.sort(np.random.default_rng(config["seed"]).choice(
            train_idx, int(config["train_fraction"] * len(train_idx)), replace=False))
    notes["train_windows"] = int(len(train_idx))
    targets, soft_weight = train_labels, (config["soft_weight"] if config["soft_source"] else 0.0)
    if soft_weight > 0:
        if config["mixup"] > 0:
            raise ValueError("soft targets and mixup cannot be combined")
        runs = resolve_runs(config["soft_runs"], out_dir)
        if config["soft_source"] == "oof":
            soft = gather_oof(runs, len(train_labels))
        elif config["soft_source"] == "teacher":
            teacher = torch.load(os.path.join(runs[0], "best.pt"), map_location="cpu", weights_only=True)
            if teacher.get("frontend", "paper") != config["frontend"] and \
                    not os.path.exists(os.path.join(runs[0], "train_scores.npy")):
                raise RuntimeError("the teacher was trained on another front-end: score its train split first")
            soft = run_scores(runs[0], "train", train_images, device)
        else:
            raise ValueError(f"Unknown soft_source {config['soft_source']!r} (oof or teacher)")
        targets = np.stack([train_labels, soft], axis=1).astype(np.float32)  # label, soft whistle probability
    val_images, val_labels, _ = data.validation

    model = torch_models.build_model(config["arch"], pretrained=config["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    steps_per_epoch = math.ceil(len(train_idx) / config["batch_size"])
    total_steps = steps_per_epoch * config["epochs"]
    warmup = max(1, int(config["warmup_epochs"] * steps_per_epoch))
    schedule = lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
    mean = torch.tensor(torch_models.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(torch_models.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    criterion = hard_loss(config)
    ema = None
    if config["ema"] > 0:  # exponential moving average of the weights (and batch-norm statistics)
        decay = config["ema"]
        ema = torch.optim.swa_utils.AveragedModel(
            model, avg_fn=lambda averaged, current, count: decay * averaged + (1 - decay) * current, use_buffers=True)

    history, start_epoch = [], 0
    best = {"val_loss": float("inf")}
    last_path = os.path.join(out_dir, "last.pt")
    if os.path.exists(last_path):  # resume an interrupted run
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if ema is not None and "ema_state_dict" in state:
            ema.load_state_dict(state["ema_state_dict"])
        history, best, start_epoch = state["history"], state["best"], state["epoch"]
        log(f"  resumed after epoch {start_epoch}")

    stale = history[-1]["epoch"] - best.get("epoch", 0) if history else 0
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        if stale >= config["patience"]:
            log(f"  early stop: no validation-loss improvement for {stale} epochs")
            break
        model.train()
        started, running, seen = time.time(), 0.0, 0
        for step, (batch, batch_targets) in enumerate(Batches(train_images, targets, config["batch_size"],
                                                              shuffle=True, seed=config["seed"] * 1000 + epoch,
                                                              indices=train_idx)):
            x, normalise = to_model_input(batch, device, mean, std, config["img_size"])
            x = augment(x, config)
            batch_targets = batch_targets.to(device, non_blocking=True)
            y = batch_targets[:, 0].long() if soft_weight > 0 else batch_targets
            with torch.autocast("cuda", dtype=torch.float16, enabled=config["amp"]):
                if config["mixup"] > 0:
                    lam = float(np.random.beta(config["mixup"], config["mixup"]))
                    perm = torch.randperm(len(y), device=device)
                    logits = model(normalise(lam * x + (1 - lam) * x[perm]))
                    loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y[perm])
                else:
                    logits = model(normalise(x))
                    loss = criterion(logits, y)
                    if soft_weight > 0:
                        loss = (1 - soft_weight) * loss + soft_weight * distillation_loss(
                            logits, batch_targets[:, 1], config["distill_temperature"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update_parameters(model)
            running += float(loss) * len(y)
            seen += len(y)
            if step % 200 == 0:
                log(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {running / seen:.4f} "
                    f"({seen / (time.time() - started):.0f} img/s)")

        evaluated = ema.module if ema is not None else model
        scores = evaluate_scores(evaluated, val_images, config["batch_size"], device, mean, std, config["img_size"])
        metrics = em.compute_metrics(val_labels, scores)
        row = {"epoch": epoch, "train_loss": running / seen, "val_loss": binary_log_loss(val_labels, scores),
               "val_accuracy": metrics["accuracy"], "val_precision": metrics["precision"],
               "val_recall": metrics["recall"], "val_auc": metrics["roc_auc"],
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - started}
        row = {key: int(value) if key == "epoch" else float(value) for key, value in row.items()}  # plain types:
        history.append(row)  # checkpoints are read back with weights_only=True, which refuses numpy scalars
        tracker.log(row, step=epoch)
        log(f"  epoch {epoch}: train loss {row['train_loss']:.4f}, val loss {row['val_loss']:.4f}, "
            f"val acc {row['val_accuracy']:.4f}, val AUC {row['val_auc']:.4f} ({row['seconds'] / 60:.1f} min)")

        if row["val_loss"] < best["val_loss"]:
            best = dict(row)
            stale = 0
            torch.save({"arch": config["arch"], "model_state_dict": evaluated.state_dict(),
                        "img_size": config["img_size"], "frontend": config["frontend"],
                        "normalization": {"mean": list(torch_models.IMAGENET_MEAN),
                                          "std": list(torch_models.IMAGENET_STD)},
                        "epoch": epoch, "config": config, "validation": best,
                        "dataset": "dolphinteam/OpenWhistle-CNN"},
                       os.path.join(out_dir, "best.pt"))
            # the score of every validation window: error analyses and label corrections need no new pass
            np.save(os.path.join(out_dir, "best_validation_scores.npy"), scores.astype(np.float32))
        else:
            stale += 1
        state = {"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "history": history, "best": best, "epoch": epoch}
        if ema is not None:
            state["ema_state_dict"] = ema.state_dict()
        torch.save(state, last_path)
        with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    if held_out is not None:  # out-of-fold scores of the held-out recordings, by the best epoch
        best_model, _ = torch_models.load_checkpoint(os.path.join(out_dir, "best.pt"))
        oof = evaluate_scores(best_model.to(device), train_images, config["batch_size"], device, mean, std,
                              config["img_size"], indices=held_out)
        np.save(os.path.join(out_dir, "oof_scores.npy"), oof.astype(np.float32))
        np.save(os.path.join(out_dir, "oof_indices.npy"), held_out)
        oof_metrics = em.compute_metrics(train_labels[held_out], oof)
        notes["held_out_accuracy"] = float(oof_metrics["accuracy"])
        log(f"  held-out recordings: accuracy {oof_metrics['accuracy']:.4f}")

    scores_path = os.path.join(out_dir, "best_validation_scores.npy")
    if os.path.exists(scores_path):
        by_type = accuracy_by_recording_type(val_labels, np.load(scores_path), data.validation[2])
        notes["val_accuracy_by_recording_type"] = by_type
        log("  validation accuracy by recording type: " + ", ".join(f"{k} {v:.4f}" for k, v in by_type.items()))
    cost = torch_models.model_cost(torch_models.build_model(config["arch"]), config["img_size"])
    log(f"  {cost['params']:,} parameters, {cost['macs'] / 1e6:.1f} M multiply-adds, "
        f"{cost['cpu_ms']:.1f} ms per image on 1 CPU thread")
    summary = {"config": config, "best": best, "epochs_run": len(history),
               "minutes": sum(r["seconds"] for r in history) / 60, "cost": cost, **notes}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    tracker.finish({**{f"best_{k}": v for k, v in best.items()}, **cost})
    if os.path.exists(last_path):
        os.remove(last_path)
    return summary


# --------------------------------------------------------------------------- commands

def run_sweep(sweep_file, out_dir, data, device, only=None, wandb_project=None):
    """``data``: a Data, or a function returning the Data of a front-end (SplitCache)."""
    with open(sweep_file) as f:
        sweep = json.load(f)
    base, runs = sweep.get("base", {}), sweep["runs"]
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for i, run in enumerate(runs):
        name = run.pop("name")
        if only and name not in only:
            continue
        run_dir = os.path.join(out_dir, name)
        print(f"=== [{i + 1}/{len(runs)}] {name}: {run}", flush=True)
        try:
            config = {**base, **run}
            run_data = data(config.get("frontend", "paper")) if callable(data) else data
            summary = train_run(config, run_dir, run_data, device,
                                log=lambda message: print(message, flush=True), wandb_project=wandb_project)
        except RuntimeError as exc:  # e.g. out of GPU memory: skip this run, keep the sweep going
            print(f"  FAILED: {exc}", flush=True)
            import torch
            torch.cuda.empty_cache()
            continue
        by_type = {f"val_acc_{k}": v for k, v in summary.get("val_accuracy_by_recording_type", {}).items()}
        results.append({"run": name, **{k: summary["best"].get(k) for k in
                                        ("epoch", "val_loss", "val_accuracy", "val_precision", "val_recall",
                                         "val_auc")}, **by_type,
                        **summary.get("cost", {}), "minutes": round(summary["minutes"], 1), **run})
        write_ranking(results, os.path.join(out_dir, "sweep_results.csv"))
    return results


def write_ranking(results, path):
    results = sorted(results, key=lambda r: (-r["val_accuracy"], r["val_loss"]))
    fields = []
    for r in results:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Ranking (validation accuracy) written to {path}:")
    for r in results:
        print(f"  {r['run']}: val acc {r['val_accuracy']:.4f}, val loss {r['val_loss']:.4f}, epoch {r['epoch']}, "
              f"{r.get('params', 0):,} params, {r.get('cpu_ms', 0):.1f} ms/image")
    # smallest models reaching each accuracy level (on the validation split)
    for level in (0.95, 0.96, 0.97):
        candidates = [r for r in results if r["val_accuracy"] >= level and "params" in r]
        if candidates:
            r = min(candidates, key=lambda r: r["params"])
            print(f"  smallest with val acc >= {level:.0%}: {r['run']} ({r['params']:,} params, "
                  f"val acc {r['val_accuracy']:.4f})")


def test_checkpoints(paths, cache_dir, device, batch_size=64, output_csv=None):
    import torch
    import torch_models

    mean = torch.tensor(torch_models.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(torch_models.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    rows, loaded = [], {}
    for path in paths:
        model, metadata = torch_models.load_checkpoint(path)
        frontend = metadata.get("frontend", "paper")
        if frontend not in loaded:
            loaded = {frontend: load_split("test", cache_dir, frontend=frontend)}  # one front-end in memory
        images, labels, recordings = loaded[frontend]
        size = int(metadata.get("img_size", 224) or 224)
        scores = evaluate_scores(model.to(device), images, batch_size, device, mean, std, size)
        metrics = em.compute_metrics(labels, scores)
        low, high = em.accuracy_interval(labels, scores, recordings)
        print(f"{path}: accuracy {metrics['accuracy']:.4f} (95% CI {low:.4f}-{high:.4f}), "
              f"precision {metrics['precision']:.4f}, recall {metrics['recall']:.4f}, "
              f"specificity {metrics['specificity']:.4f}, ROC AUC {metrics['roc_auc']:.4f} "
              f"(TP {metrics['tp']}, FP {metrics['fp']}, FN {metrics['fn']}, TN {metrics['tn']})")
        rows.append({"model": path, **metrics, "accuracy_ci_low": low, "accuracy_ci_high": high})
        np.save(os.path.splitext(path)[0] + "_test_scores.npy", scores)
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE, help="Dataset cache (shared with evaluate_models.py)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Download and decode the three splits")
    prepare.add_argument("--frontends", nargs="*", default=[],
                         help="Also compute these audio front-ends (wide, whiten, pcen) from the audio")

    train = commands.add_parser("train", help="One training run")
    train.add_argument("--out", required=True, help="Output folder of the run")
    for key, value in DEFAULTS.items():
        if isinstance(value, bool):
            train.add_argument(f"--{key}", type=lambda s: s.lower() in ("1", "true", "yes"), default=value)
        else:
            train.add_argument(f"--{key}", type=type(value), default=value)

    sweep = commands.add_parser("sweep", help="Runs listed in a JSON file (resumable)")
    sweep.add_argument("sweep_file")
    sweep.add_argument("--out", required=True)
    sweep.add_argument("--only", nargs="+", help="Run only these runs of the file")
    for command in (train, sweep):
        command.add_argument("--wandb_project", help="Also log the runs to this Weights & Biases project")

    test = commands.add_parser("test", help="Metrics on the test split (use once, on the chosen model)")
    test.add_argument("checkpoints", nargs="+")
    test.add_argument("--output_csv")

    args = parser.parse_args(argv)
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "prepare":
        for split in ("validation", "test", "train"):
            prepare_split(split, args.cache_dir)
            if args.frontends:
                prepare_frontends(split, args.cache_dir, args.frontends)
        return
    if args.command == "test":
        return test_checkpoints(args.checkpoints, args.cache_dir, device, output_csv=args.output_csv)

    data = SplitCache(args.cache_dir)
    if args.command == "train":
        config = {key: getattr(args, key) for key in DEFAULTS}
        return train_run(config, args.out, data(config["frontend"]), device, wandb_project=args.wandb_project)
    return run_sweep(args.sweep_file, args.out, data, device, args.only, args.wandb_project)


if __name__ == "__main__":
    main()
