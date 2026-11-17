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


def load_split(split, cache_dir, in_memory=True):
    """
    (images uint8 [n, 224, 224], labels int64, recordings) of a prepared split.

    ``in_memory`` reads the images once, sequentially (train: 2.7 GB of RAM). Training draws random batches,
    and random reads in a memory-mapped file are very slow on a hard disk.
    """
    prepare_split(split, cache_dir)
    started = time.time()
    images = np.load(os.path.join(cache_dir, f"{split}_images.npy"), mmap_mode=None if in_memory else "r")
    if in_memory:
        print(f"{split}: {images.nbytes / 2**30:.1f} GB loaded in {time.time() - started:.0f} s", flush=True)
    labels = np.load(os.path.join(cache_dir, f"{split}_labels.npy"))
    recordings = np.load(os.path.join(cache_dir, f"{split}_recordings.npy"))
    return images, labels, recordings


class Batches:
    """Batches of (uint8 images, labels) read from the memmap by a background worker pool."""

    def __init__(self, images, labels, batch_size, shuffle, seed=0, workers=3):
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
        order = rng.permutation(len(labels)) if shuffle else np.arange(len(labels))
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
}


@dataclasses.dataclass
class Data:
    train: tuple
    validation: tuple


def evaluate_scores(model, images, batch_size, device, mean, std, size=224):
    """Whistle probability of every image (the same computation as the detection pipeline)."""
    import torch

    scores = []
    model.eval()
    with torch.inference_mode():
        for batch, _ in Batches(images, np.zeros(len(images), dtype=np.int64), batch_size * 2, shuffle=False):
            x, normalise = to_model_input(batch, device, mean, std, size)
            scores.append(torch.softmax(model(normalise(x)).float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def binary_log_loss(labels, scores):
    p = np.clip(scores, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


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

    train_images, train_labels, _ = data.train
    if config["train_fraction"] < 1:  # learning-curve / quick runs: a fixed random subset
        keep = np.sort(np.random.default_rng(config["seed"]).choice(
            len(train_labels), int(config["train_fraction"] * len(train_labels)), replace=False))
        train_images, train_labels = train_images[keep], train_labels[keep]
    val_images, val_labels, _ = data.validation

    model = torch_models.build_model(config["arch"], pretrained=config["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    steps_per_epoch = math.ceil(len(train_labels) / config["batch_size"])
    total_steps = steps_per_epoch * config["epochs"]
    warmup = max(1, int(config["warmup_epochs"] * steps_per_epoch))
    schedule = lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
    mean = torch.tensor(torch_models.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(torch_models.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    history, start_epoch = [], 0
    best = {"val_loss": float("inf")}
    last_path = os.path.join(out_dir, "last.pt")
    if os.path.exists(last_path):  # resume an interrupted run
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best, start_epoch = state["history"], state["best"], state["epoch"]
        log(f"  resumed after epoch {start_epoch}")

    stale = history[-1]["epoch"] - best.get("epoch", 0) if history else 0
    for epoch in range(start_epoch + 1, config["epochs"] + 1):
        if stale >= config["patience"]:
            log(f"  early stop: no validation-loss improvement for {stale} epochs")
            break
        model.train()
        started, running, seen = time.time(), 0.0, 0
        for step, (batch, labels) in enumerate(Batches(train_images, train_labels, config["batch_size"],
                                                       shuffle=True, seed=config["seed"] * 1000 + epoch)):
            x, normalise = to_model_input(batch, device, mean, std, config["img_size"])
            x = augment(x, config)
            y = labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=config["amp"]):
                if config["mixup"] > 0:
                    lam = float(np.random.beta(config["mixup"], config["mixup"]))
                    perm = torch.randperm(len(y), device=device)
                    logits = model(normalise(lam * x + (1 - lam) * x[perm]))
                    loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y[perm])
                else:
                    loss = criterion(model(normalise(x)), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss) * len(y)
            seen += len(y)
            if step % 200 == 0:
                log(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {running / seen:.4f} "
                    f"({seen / (time.time() - started):.0f} img/s)")

        scores = evaluate_scores(model, val_images, config["batch_size"], device, mean, std, config["img_size"])
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
            torch.save({"arch": config["arch"], "model_state_dict": model.state_dict(), "img_size": config["img_size"],
                        "normalization": {"mean": list(torch_models.IMAGENET_MEAN),
                                          "std": list(torch_models.IMAGENET_STD)},
                        "epoch": epoch, "config": config, "validation": best,
                        "dataset": "dolphinteam/OpenWhistle-CNN"},
                       os.path.join(out_dir, "best.pt"))
        else:
            stale += 1
        torch.save({"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history, "best": best, "epoch": epoch},
                   last_path)
        with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

    cost = torch_models.model_cost(torch_models.build_model(config["arch"]), config["img_size"])
    log(f"  {cost['params']:,} parameters, {cost['macs'] / 1e6:.1f} M multiply-adds, "
        f"{cost['cpu_ms']:.1f} ms per image on 1 CPU thread")
    summary = {"config": config, "best": best, "epochs_run": len(history),
               "minutes": sum(r["seconds"] for r in history) / 60, "cost": cost}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    tracker.finish({**{f"best_{k}": v for k, v in best.items()}, **cost})
    if os.path.exists(last_path):
        os.remove(last_path)
    return summary


# --------------------------------------------------------------------------- commands

def run_sweep(sweep_file, out_dir, data, device, only=None, wandb_project=None):
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
            summary = train_run({**base, **run}, run_dir, data, device,
                                log=lambda message: print(message, flush=True), wandb_project=wandb_project)
        except RuntimeError as exc:  # e.g. out of GPU memory: skip this run, keep the sweep going
            print(f"  FAILED: {exc}", flush=True)
            import torch
            torch.cuda.empty_cache()
            continue
        results.append({"run": name, **{k: summary["best"].get(k) for k in
                                        ("epoch", "val_loss", "val_accuracy", "val_precision", "val_recall",
                                         "val_auc")},
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

    images, labels, recordings = load_split("test", cache_dir)
    mean = torch.tensor(torch_models.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(torch_models.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    rows = []
    for path in paths:
        model, metadata = torch_models.load_checkpoint(path)
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

    commands.add_parser("prepare", help="Download and decode the three splits")

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
        return
    if args.command == "test":
        return test_checkpoints(args.checkpoints, args.cache_dir, device, output_csv=args.output_csv)

    data = Data(train=load_split("train", args.cache_dir), validation=load_split("validation", args.cache_dir))
    if args.command == "train":
        config = {key: getattr(args, key) for key in DEFAULTS}
        return train_run(config, args.out, data, device, wandb_project=args.wandb_project)
    return run_sweep(args.sweep_file, args.out, data, device, args.only, args.wandb_project)


if __name__ == "__main__":
    main()
