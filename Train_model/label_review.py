#!/usr/bin/env python3
"""Audit the labels of OpenWhistle-CNN with trained models, and review doubtful windows by hand.

Runs trained with train_openwhistle.py save the whistle score of every validation window
(``best_validation_scores.npy``). The commands below start from these scores:

    overlap    how much the models fail on the same validation windows: a core of windows that every
               model gets wrong points to label errors or ambiguous windows rather than to the models
    context    for every validation window: the other channel of the same session at the same time, the
               labels of the neighbouring windows, and the run of consecutive whistle windows it belongs to
    page       a review page of the windows that chosen models misclassify: the model's image, the
               0-48 kHz spectrogram and the audio of each window, the scores of every model, and buttons
               to record a verdict (the model is wrong / the label is wrong / ambiguous)
    blind      a blind review page: windows drawn at random among those the models classify correctly,
               plus some of the reviewed errors, shown without label or score, to estimate the label
               error rate of the whole split and the agreement between annotators
    correct    corrected validation labels from the verdicts of the review page, and the accuracy of
               every run on them
    agreement  label error rate and agreement between annotators, from the answers of the blind review
    scores     the validation scores of runs trained before they were saved, or of a Keras model

The pages are static: serve their folder (``python -m http.server -d <folder>``) and open it in a
browser. Verdicts are kept in the browser and exported as JSON with a button. Published as a Claude
artifact with the ``db`` capability, they are stored online instead. See docs/LABEL_REVIEW.md.

Examples:
    python Train_model/label_review.py overlap runs/sweep/*/
    python Train_model/label_review.py page runs/sweep/*/ --panel runs/sweep/mnv2_224 --out review
    python Train_model/label_review.py blind --panel runs/sweep/mnv2_224 --seed 1234 --out blind \\
        --key runs/blind_key.csv
    python Train_model/label_review.py correct --verdicts verdicts.json --runs runs/sweep/*/ --out corrected
"""
import argparse
import base64
import csv
import glob
import itertools
import json
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "Predict_and_extract"))
sys.path.insert(0, os.path.join(REPO_ROOT, "Train_model"))

DEFAULT_CACHE = os.path.join(REPO_ROOT, "openwhistle_cnn_cache")
TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_pages")
SCORES_FILE = "best_validation_scores.npy"
WINDOW = 0.4  # seconds
CHUNK = 8  # windows per data file of a page (a Claude artifact version holds at most 511 files)
META_COLUMNS = ["label", "recording", "onset", "offset", "file_name"]
TAGS = [["faint", "Sifflement faible"], ["edge", "Coupé au bord de la fenêtre"], ["band", "Hors bande 2–22 kHz"],
        ["tonal", "Bruit tonal (sonar, moteur…)"], ["clicks", "Clics / burst pulses"],
        ["bio", "Autre son biologique"], ["artifact", "Saturation / artefact"], ["overlap", "Plusieurs sons mêlés"]]


# --------------------------------------------------------------------------- dataset
def split_files(cache_dir, split):
    paths = sorted(glob.glob(os.path.join(cache_dir, f"{split}-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no {split} parquet file in {cache_dir}: run train_openwhistle.py prepare first")
    return paths


def split_metadata(cache_dir, split):
    """Label, recording, onset, offset and file name of every window of a split, in dataset order."""
    import pyarrow.parquet as pq

    columns = {name: [] for name in META_COLUMNS}
    for path in split_files(cache_dir, split):
        table = pq.read_table(path, columns=META_COLUMNS)
        for name in META_COLUMNS:
            columns[name] += table.column(name).to_pylist()
    return {"label": np.array(columns["label"], dtype=np.int64),
            "recording": np.array(columns["recording"], dtype=object),
            "onset": np.array(columns["onset"], dtype=np.float64),
            "offset": np.array(columns["offset"], dtype=np.float64),
            "file_name": np.array(columns["file_name"], dtype=object)}


def session_and_channel(recording):
    """("Exp_07_Mar_2023_0945", "1") for "Exp_07_Mar_2023_0945_channel_1"; (recording, None) without channel."""
    if "_channel_" in recording:
        session, channel = recording.rsplit("_channel_", 1)
        return session, channel
    return recording, None


def window_payloads(cache_dir, split, indices):
    """{index: {"png": base64 spectrogram, "wav": base64 audio}} for these windows of a split."""
    import pyarrow.parquet as pq

    want = {int(i) for i in indices}
    payloads, index = {}, 0
    for path in split_files(cache_dir, split):
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            rows = parquet.metadata.row_group(group).num_rows
            needed = [k for k in range(rows) if index + k in want]
            if needed:
                full = parquet.read_row_group(group, columns=["audio", "spectrogram"]).to_pylist()
                for k in needed:
                    payloads[index + k] = {"png": base64.b64encode(full[k]["spectrogram"]["bytes"]).decode(),
                                           "wav": base64.b64encode(full[k]["audio"]["bytes"]).decode()}
            index += rows
    missing = want - set(payloads)
    if missing:
        raise IndexError(f"{len(missing)} windows are beyond the {index} windows of the {split} split")
    return payloads


# --------------------------------------------------------------------------- runs
def load_runs(specs, n):
    """
    Models given as "RUN_DIR" or "RUN_DIR=Display name", and their validation scores [models, n].
    A run folder must hold best_validation_scores.npy (written by train_openwhistle.py, or by ``scores``).
    """
    models, rows = [], []
    for spec in specs:
        path, _, name = spec.partition("=")
        path = os.path.normpath(path)
        file = os.path.join(path, SCORES_FILE)
        if not os.path.exists(file):
            raise FileNotFoundError(f"{file} is missing: run `label_review.py scores {path}`")
        scores = np.load(file).astype(np.float64)
        if scores.shape != (n,):
            raise ValueError(f"{file} holds {scores.shape} scores, the validation split has {n} windows")
        key = os.path.basename(path)
        if any(m["key"] == key for m in models):
            key = os.path.join(os.path.basename(os.path.dirname(path)), key)
        params = None
        if os.path.exists(os.path.join(path, "summary.json")):
            with open(os.path.join(path, "summary.json")) as f:
                params = json.load(f).get("cost", {}).get("params")
        models.append({"key": key, "name": name or key, "params": params, "dir": path})
        rows.append(scores)
    return models, np.stack(rows)


def misclassified(scores, labels):
    """[models, n] boolean: the model's decision at 0.5 differs from the label."""
    return (np.asarray(scores) >= 0.5) != (np.asarray(labels)[None, :] == 1)


def metrics(labels, scores, keep=None):
    keep = np.ones(len(labels), bool) if keep is None else keep
    pred, truth = (scores >= 0.5)[keep], (labels == 1)[keep]
    tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
    return {"accuracy": float((pred == truth).mean()), "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "missed": fn, "false_alarms": fp}


# --------------------------------------------------------------------------- overlap
def error_overlap(wrong, labels):
    """
    Windows by number of models wrong, the core that every model gets wrong (and its expected size if the
    models erred independently with the same error rates), and the overlap of every pair of error sets.
    """
    n_models, n = wrong.shape
    count = wrong.sum(axis=0)
    by_count = [{"models_wrong": j, "windows": int((count == j).sum()),
                 "labelled_whistle": int(((count == j) & (labels == 1)).sum()),
                 "labelled_noise": int(((count == j) & (labels == 0)).sum())} for j in range(n_models + 1)]
    rates = wrong.mean(axis=1)
    pairs = []
    for a, b in itertools.combinations(range(n_models), 2):
        common = int((wrong[a] & wrong[b]).sum())
        pairs.append({"a": a, "b": b, "common": common,
                      "jaccard": common / max(1, int((wrong[a] | wrong[b]).sum())),
                      "overlap": common / max(1, int(min(wrong[a].sum(), wrong[b].sum()))),
                      "expected_if_independent": float(n * rates[a] * rates[b])})
    return {"by_count": by_count, "core": np.flatnonzero(count == n_models),
            "expected_core": float(n * np.prod(rates)), "pairs": pairs}


# --------------------------------------------------------------------------- context
def window_context(splits, scores=None, score_name="model"):
    """
    One row per validation window: the windows of the other channel(s) of the same session that overlap it
    in time (from any split), the labels of the previous and next windows of the same recording, the
    distance to the nearest other window labelled whistle, and for whistle windows the length of the run of
    consecutive whistle windows and whether the window is at an edge of that run.
    ``splits`` maps split names to split_metadata(); ``scores`` (optional) are validation scores.
    """
    by_channel = {}  # (session, channel) -> sorted onsets, and (split, index) of each
    for split, meta in splits.items():
        for i, recording in enumerate(meta["recording"]):
            session, channel = session_and_channel(recording)
            if channel is not None:
                by_channel.setdefault((session, channel), []).append((meta["onset"][i], split, i))
    for key in by_channel:
        by_channel[key].sort()
    channels_of = {}
    for session, channel in by_channel:
        channels_of.setdefault(session, []).append(channel)
    onset_arrays = {key: np.array([w[0] for w in windows]) for key, windows in by_channel.items()}

    val = splits["validation"]
    n = len(val["label"])
    label_at, whistles = {}, {}
    for i in range(n):
        label_at[(val["recording"][i], round(val["onset"][i], 1))] = int(val["label"][i])
        if val["label"][i] == 1:
            whistles.setdefault(val["recording"][i], []).append(val["onset"][i])
    whistles = {rec: np.sort(np.array(onsets)) for rec, onsets in whistles.items()}

    rows = []
    for i in range(n):
        recording, onset, label = val["recording"][i], val["onset"][i], int(val["label"][i])
        row = {"index_in_split": i, "recording": recording, "onset": round(float(onset), 3),
               "offset": round(float(val["offset"][i]), 3), "label": label}
        if scores is not None:
            row["score"] = round(float(scores[i]), 4)
        session, channel = session_and_channel(recording)
        found = []
        if channel is None:
            status = "single_channel_recording"
        elif len(channels_of.get(session, [])) < 2:
            status = "other_channel_not_in_dataset"
        else:
            for other in sorted(c for c in channels_of[session] if c != channel):
                onsets = onset_arrays[(session, other)]
                lo, hi = np.searchsorted(onsets, onset - WINDOW + 1e-6), np.searchsorted(onsets, onset + WINDOW - 1e-6)
                for o, split, j in by_channel[(session, other)][lo:hi]:
                    meta = splits[split]
                    text = f"channel_{other} {o:.1f}-{meta['offset'][j]:.1f} s {'whistle' if meta['label'][j] else 'noise'}"
                    if split != "validation":
                        text += f" ({split})"
                    elif scores is not None:
                        text += f" {score_name}={scores[j]:.2f}"
                    found.append(text)
            status = "overlapping_window_found" if found else "other_channel_no_window_at_this_time"
        row["other_channel"] = status
        row["other_channel_windows"] = "; ".join(found)
        row["previous_label"] = label_at.get((recording, round(onset - WINDOW, 1)), "")
        row["next_label"] = label_at.get((recording, round(onset + WINDOW, 1)), "")
        others = whistles.get(recording, np.array([]))
        others = others[np.abs(others - onset) > 1e-3]
        row["nearest_whistle_s"] = round(float(np.min(np.abs(others - onset))), 3) if len(others) else ""
        if label == 1:
            on_grid = whistles[recording]
            has = lambda t: bool(np.any(np.abs(on_grid - t) < 0.05))  # noqa: E731
            before, after = 0, 0
            while has(onset - (before + 1) * WINDOW):
                before += 1
            while has(onset + (after + 1) * WINDOW):
                after += 1
            row["whistle_run"] = 1 + before + after
            row["run_edge"] = int(before == 0 or after == 0)
        else:
            row["whistle_run"], row["run_edge"] = "", ""
        rows.append(row)
    return rows


OTHER_TEXT = {
    "single_channel_recording": "Enregistrement à un seul canal : pas d'autre canal.",
    "other_channel_not_in_dataset": "Autre canal : absent du jeu de données pour cette session.",
    "other_channel_no_window_at_this_time": "Autre canal : présent pour cette session, mais aucune fenêtre à ce moment-là.",
}


def other_channel_note(row):
    """The sentence shown on a review card about the other channel."""
    if row["other_channel"] != "overlapping_window_found":
        return {"found": False, "text": OTHER_TEXT[row["other_channel"]]}
    text = (row["other_channel_windows"].replace(" whistle", ", étiqueté sifflement").replace(" noise", ", étiqueté bruit")
            .replace("(train)", "(split train)").replace("(test)", "(split test)"))
    return {"found": True, "text": "Autre canal au même moment : " + text}


# --------------------------------------------------------------------------- pages
def write_chunks(out_dir, folder, ids, payloads):
    """Pack the payloads CHUNK windows per JSON file; returns {item id: relative path of its file}."""
    os.makedirs(os.path.join(out_dir, folder), exist_ok=True)
    where, chunks = {}, {}
    for k, item_id in enumerate(ids):
        name = f"{folder}/c{k // CHUNK:03d}.json"
        where[item_id] = name
        chunks.setdefault(name, {})[item_id] = payloads[item_id]
    for name, content in chunks.items():
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(content, f, separators=(",", ":"))
    return where


# The templates are page bodies (the form a Claude artifact takes); a standalone page gets this skeleton.
SKELETON = ('<!doctype html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n')


def render(template, data, path):
    with open(os.path.join(TEMPLATES, template), encoding="utf-8") as f:
        html = f.read()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SKELETON + html.replace("__DATA__", payload) + "\n</body>\n</html>\n")


def build_review_page(models, scores, panel, val, payloads, out_dir, context_rows=None, previous=()):
    """
    Write the review page of the windows misclassified by at least one panel model. ``panel`` holds model
    keys, ``val`` is split_metadata("validation"), ``payloads`` maps window indices to their image and audio,
    ``previous`` lists windows shown in an earlier review (they form series 1, the others series 2).
    Returns the indices of the listed windows.
    """
    labels = val["label"]
    wrong = misclassified(scores, labels)
    for model, w, s in zip(models, wrong, scores):
        model.update(metrics(labels, s), panel=model["key"] in panel)
    order = np.argsort([0 if m["panel"] else 1 for m in models], kind="stable")
    models = [models[i] for i in order]
    scores, wrong = scores[order], wrong[order]
    panel_rows = [i for i, m in enumerate(models) if m["panel"]]
    selected = np.flatnonzero(wrong[panel_rows].any(axis=0))
    previous = {int(i) for i in previous}
    ids = [f"v{i:05d}" for i in selected]
    # pack the data files in the page's default order (clearest errors first): the first cards share files
    strength = np.abs(scores[panel_rows][:, selected] - labels[selected][None, :]).mean(axis=0)
    packed = [ids[k] for k in np.argsort(-strength, kind="stable")]
    where = write_chunks(out_dir, "items", packed, {f"v{i:05d}": payloads[i] for i in selected})
    items = []
    for i, item_id in zip(selected, ids):
        items.append({"id": item_id, "i": int(i), "y": int(labels[i]), "rec": val["recording"][i],
                      "oc": other_channel_note(context_rows[i]) if context_rows else None,
                      "r": 1 if not previous or i in previous else 2, "f": where[item_id],
                      "on": round(float(val["onset"][i]), 3), "off": round(float(val["offset"][i]), 3),
                      "file": val["file_name"][i], "s": [round(float(x), 4) for x in scores[:, i]]})
    recordings = []
    for rec in sorted(set(val["recording"].tolist())):
        mask = val["recording"] == rec
        recordings.append({"name": rec, "n": int(mask.sum()), "w": int((labels[mask] == 1).sum()),
                           "err": {models[i]["key"]: int(wrong[i][mask].sum()) for i in panel_rows}})
    recordings.sort(key=lambda r: -sum(r["err"].values()))
    shown = [{k: m[k] for k in ("key", "name", "params", "panel", "accuracy", "precision", "recall", "missed",
                                "false_alarms")} for m in models]
    for m in shown:
        m["short"] = m["key"]
        m.update(acc=m.pop("accuracy"), prec=m.pop("precision"), rec=m.pop("recall"), fn=m.pop("missed"),
                 fp=m.pop("false_alarms"))
    render("review.html", {"n": int(len(labels)), "models": shown, "items": items, "recordings": recordings,
                           "tags": TAGS}, os.path.join(out_dir, "index.html"))
    return selected


def blind_sample(n, reviewed, seed, n_random, n_errors):
    """
    Indices and sources of the blind review: ``n_random`` windows drawn among those outside ``reviewed``
    ("random_correct") and ``n_errors`` among ``reviewed`` ("reviewed_error"), in a random order.
    """
    reviewed = np.array(sorted({int(i) for i in reviewed}), dtype=np.int64)
    rest = np.setdiff1d(np.arange(n), reviewed)
    if n_random > len(rest) or n_errors > len(reviewed):
        raise ValueError(f"cannot draw {n_random} of {len(rest)} other windows and {n_errors} of {len(reviewed)} "
                         f"reviewed windows")
    rng = np.random.default_rng(seed)
    picked = np.concatenate([rng.choice(rest, n_random, replace=False), rng.choice(reviewed, n_errors, replace=False)])
    source = np.array(["random_correct"] * n_random + ["reviewed_error"] * n_errors)
    order = rng.permutation(len(picked))
    return picked[order], source[order], {"random_correct": int(len(rest)), "reviewed_error": int(len(reviewed))}


def build_blind_page(val, picked, source, payloads, out_dir, key_path, population, seed):
    """Write the blind page (no label, score or recording in it) and, separately, the key of its windows."""
    ids = [f"b{k + 1:03d}" for k in range(len(picked))]
    where = write_chunks(out_dir, "w", ids, {b: payloads[int(i)] for b, i in zip(ids, picked)})
    render("blind.html", {"items": [{"id": b, "f": where[b]} for b in ids], "tags": TAGS},
           os.path.join(out_dir, "index.html"))
    os.makedirs(os.path.dirname(os.path.abspath(key_path)), exist_ok=True)
    with open(key_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["blind_id", "index_in_split", "source", "label", "recording", "onset", "offset", "file_name"])
        for b, i, s in zip(ids, picked, source):
            writer.writerow([b, int(i), s, "whistle" if val["label"][i] == 1 else "noise", val["recording"][i],
                             round(float(val["onset"][i]), 3), round(float(val["offset"][i]), 3), val["file_name"][i]])
    with open(os.path.splitext(key_path)[0] + ".json", "w") as f:
        json.dump({"seed": seed, "population": population, "windows": len(ids)}, f, indent=1)


# --------------------------------------------------------------------------- verdicts and answers
def read_records(path):
    """
    Records with an "id" from a JSON export of a page (a list, or {"annotator": ..., "answers": [...]}) or
    from a folder of database documents, one JSON file per window (the file name is the id).
    """
    if os.path.isdir(path):
        records = []
        for file in sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True)):
            with open(file) as f:
                doc = json.load(f)
            data = doc.get("data", doc) if isinstance(doc, dict) else {}
            records.append(dict(data, id=doc.get("id") or os.path.splitext(os.path.basename(file))[0]))
        return records
    with open(path) as f:
        content = json.load(f)
    return content["answers"] if isinstance(content, dict) else content


def corrected_labels(labels, verdicts):
    """Flip the windows judged "label" (the label is wrong), keep "model", leave out the others."""
    labels = np.array(labels, copy=True)
    keep = np.ones(len(labels), bool)
    counts = {"label": 0, "model": 0, "other": 0}
    for record in verdicts:
        i = int(str(record["id"]).lstrip("v"))
        verdict = record.get("verdict", "")
        if verdict == "label":
            labels[i] = 1 - labels[i]
        elif verdict != "model":
            keep[i] = False
        counts[verdict if verdict in ("label", "model") else "other"] += 1
    return labels, keep, counts


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def cohen_kappa(a, b):
    """Cohen's kappa of two lists of categorical answers of equal length."""
    cats = sorted(set(a) | set(b))
    if len(a) == 0:
        return float("nan")
    observed = np.mean([x == y for x, y in zip(a, b)])
    expected = sum((a.count(c) / len(a)) * (b.count(c) / len(b)) for c in cats)
    return float("nan") if expected == 1 else float((observed - expected) / (1 - expected))


def blind_agreement(key_rows, annotators, population=None):
    """
    ``key_rows``: rows of the blind key; ``annotators``: {name: {blind id: answer}} with answers "whistle",
    "noise" or "unsure". Per annotator and source: how often a definite answer contradicts the dataset label
    (with a 95% Wilson interval, and the implied number of label errors in the population of that source);
    per pair of annotators: raw agreement and Cohen's kappa.
    """
    label_of = {r["blind_id"]: r["label"] for r in key_rows}
    source_of = {r["blind_id"]: r["source"] for r in key_rows}
    report = {"annotators": {}, "pairs": []}
    for name, answers in annotators.items():
        per_source = {}
        for source in sorted(set(source_of.values())):
            ids = [b for b in answers if source_of.get(b) == source]
            definite = [b for b in ids if answers[b] in ("whistle", "noise")]
            contradicted = sum(answers[b] != label_of[b] for b in definite)
            low, high = wilson_interval(contradicted, len(definite))
            entry = {"answered": len(ids), "unsure": len(ids) - len(definite), "definite": len(definite),
                     "contradict_label": contradicted,
                     "rate": contradicted / len(definite) if definite else float("nan"), "ci95": [low, high]}
            if population and source in population and definite:
                entry["implied_label_errors"] = [round(x * population[source], 1) for x in (entry["rate"], low, high)]
            per_source[source] = entry
        both = [b for b in answers if answers[b] in ("whistle", "noise")]
        per_source["kappa_with_dataset_label"] = cohen_kappa([answers[b] for b in both], [label_of[b] for b in both])
        report["annotators"][name] = per_source
    for a, b in itertools.combinations(sorted(annotators), 2):
        common = sorted(set(annotators[a]) & set(annotators[b]))
        x, y = [annotators[a][i] for i in common], [annotators[b][i] for i in common]
        definite = [k for k, i in enumerate(common) if x[k] != "unsure" and y[k] != "unsure"]
        report["pairs"].append({"a": a, "b": b, "common": len(common),
                                "agreement": float(np.mean([p == q for p, q in zip(x, y)])) if common else float("nan"),
                                "kappa_3_classes": cohen_kappa(x, y),
                                "kappa_definite": cohen_kappa([x[k] for k in definite], [y[k] for k in definite]),
                                "definite": len(definite)})
    return report


# --------------------------------------------------------------------------- scores of older runs
def score_runs(run_dirs, cache_dir, device, batch_size=64):
    """best_validation_scores.npy for runs that only have best.pt."""
    import torch
    import torch_models
    import train_openwhistle as tw

    mean = torch.tensor(torch_models.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(torch_models.IMAGENET_STD, device=device).view(1, 3, 1, 1)
    loaded = {}
    for run_dir in run_dirs:
        model, metadata = torch_models.load_checkpoint(os.path.join(run_dir, "best.pt"))
        frontend = metadata.get("frontend", "paper")
        if frontend not in loaded:
            loaded = {frontend: tw.load_split("validation", cache_dir, frontend=frontend)}
        images, labels, _ = loaded[frontend]
        scores = tw.evaluate_scores(model.to(device), images, batch_size, device, mean, std,
                                    int(metadata.get("img_size", 224) or 224)).astype(np.float32)
        np.save(os.path.join(run_dir, SCORES_FILE), scores)
        print(f"{run_dir}: validation accuracy {float(((scores >= 0.5) == (labels == 1)).mean()):.4f}")


def score_keras(model_path, out_dir, cache_dir):
    """Validation scores of a Keras model of this repository (e.g. models/base_MobileNetV2.keras)."""
    import evaluate_models as em
    import utils
    from predict_and_extract_online import load_detector

    images, labels, _ = em.load_examples(em.download_split("validation", cache_dir), "dataset", 0,
                                         utils.PAPER_SPECTROGRAM)
    scores = em.predict(load_detector(model_path), images).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, SCORES_FILE), scores)
    print(f"{model_path}: validation accuracy {float(((scores >= 0.5) == (labels == 1)).mean()):.4f}")


# --------------------------------------------------------------------------- command line
def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE, help="Dataset cache of train_openwhistle.py")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("overlap", help="Shared errors of runs on the validation split")
    p.add_argument("runs", nargs="+", help="Run folders (RUN_DIR or RUN_DIR=Name)")
    p.add_argument("--out", help="Save the indices of the windows every run gets wrong (.npy)")

    p = commands.add_parser("context", help="Other channel, neighbours and runs of every validation window")
    p.add_argument("--scores", help="Run folder whose scores are added (optional)")
    p.add_argument("--out", required=True, help="CSV file")

    p = commands.add_parser("page", help="Review page of the windows misclassified by the panel runs")
    p.add_argument("runs", nargs="+", help="Runs whose scores are shown (RUN_DIR or RUN_DIR=Name)")
    p.add_argument("--panel", nargs="+", required=True, help="Runs whose errors are listed (also shown)")
    p.add_argument("--previous", help="Text file of windows (vNNNNN or indices) shown in an earlier review")
    p.add_argument("--out", required=True, help="Output folder (index.html and items/)")

    p = commands.add_parser("blind", help="Blind review page and its key")
    p.add_argument("--panel", nargs="+", required=True, help="Runs whose errors were reviewed")
    p.add_argument("--seed", type=int, required=True, help="Keep it private until the blind review is over")
    p.add_argument("--n_random", type=int, default=300)
    p.add_argument("--n_errors", type=int, default=100)
    p.add_argument("--out", required=True, help="Output folder of the page")
    p.add_argument("--key", required=True, help="CSV linking each blind id to its window: keep it out of the page")

    p = commands.add_parser("correct", help="Corrected labels from the verdicts, and the accuracy of runs on them")
    p.add_argument("--verdicts", nargs="+", required=True, help="JSON exports or folders of verdict documents")
    p.add_argument("--runs", nargs="*", default=[], help="Run folders to score on the corrected labels")
    p.add_argument("--out", required=True, help="Output folder")

    p = commands.add_parser("agreement", help="Label error rate and agreement from the blind answers")
    p.add_argument("--key", required=True, help="Key CSV written by `blind`")
    p.add_argument("--answers", nargs="+", required=True,
                   help="One per annotator: NAME=PATH, a JSON export of the blind page or a folder of documents")
    p.add_argument("--out", help="JSON report")

    p = commands.add_parser("scores", help="Validation scores of runs trained before they were saved")
    p.add_argument("runs", nargs="*", default=[], help="Run folders holding best.pt")
    p.add_argument("--keras", help="A Keras model instead (writes <--out>/best_validation_scores.npy)")
    p.add_argument("--out", help="Output folder for --keras")
    p.add_argument("--device", default=None)

    args = parser.parse_args(argv)
    if args.command == "scores":
        if args.keras:
            score_keras(args.keras, args.out or os.path.splitext(args.keras)[0], args.cache_dir)
        else:
            import torch
            score_runs(args.runs, args.cache_dir, args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return 0
    if args.command == "agreement":
        with open(args.key) as f:
            key_rows = list(csv.DictReader(f))
        population = None
        if os.path.exists(os.path.splitext(args.key)[0] + ".json"):
            with open(os.path.splitext(args.key)[0] + ".json") as f:
                population = json.load(f).get("population")
        annotators = {}
        for spec in args.answers:
            name, _, path = spec.rpartition("=")
            annotators[name or os.path.basename(os.path.normpath(path))] = {
                r["id"]: r.get("a", "") for r in read_records(path) if r.get("a")}
        report = blind_agreement(key_rows, annotators, population)
        print(json.dumps(report, indent=1))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1)
        return 0

    val = split_metadata(args.cache_dir, "validation")
    labels, n = val["label"], len(val["label"])
    if args.command == "overlap":
        models, scores = load_runs(args.runs, n)
        result = error_overlap(misclassified(scores, labels), labels)
        print(f"{n} validation windows, {len(models)} runs: {', '.join(m['key'] for m in models)}")
        for row in result["by_count"]:
            print(f"  {row['models_wrong']:2d} runs wrong: {row['windows']:5d} windows "
                  f"({row['labelled_whistle']} labelled whistle, {row['labelled_noise']} labelled noise)")
        print(f"Wrong for every run: {len(result['core'])} windows "
              f"(if the runs erred independently: {result['expected_core']:.2g})")
        overlaps = [p["overlap"] for p in result["pairs"]]
        if overlaps:
            print(f"Pairs of runs, common errors / smaller error set: min {min(overlaps):.2f}, "
                  f"median {np.median(overlaps):.2f}, max {max(overlaps):.2f}")
        if args.out:
            np.save(args.out, result["core"])
        return 0
    if args.command == "context":
        splits = {"validation": val}
        for split in ("train", "test"):
            try:
                splits[split] = split_metadata(args.cache_dir, split)
            except FileNotFoundError as error:
                print(f"warning: {error}; the other channel is only searched in the splits found")
        scores, name = None, "model"
        if args.scores:
            (model,), scores = load_runs([args.scores], n)
            scores, name = scores[0], model["key"]
        rows = window_context(splits, scores, name)
        write_csv(args.out, rows)
        counts = {}
        for row in rows:
            counts[row["other_channel"]] = counts.get(row["other_channel"], 0) + 1
        print(f"{args.out}: {len(rows)} windows; other channel: {counts}")
        return 0
    if args.command == "page":
        run_dir = lambda spec: os.path.normpath(spec.partition("=")[0])  # noqa: E731
        named = {run_dir(p): p for p in args.panel if "=" in p}  # a name given with --panel wins
        specs = [s if "=" in s else named.get(run_dir(s), s) for s in args.runs]
        specs += [p for p in args.panel if run_dir(p) not in {run_dir(s) for s in specs}]
        models, scores = load_runs(specs, n)
        panel_dirs = {run_dir(p) for p in args.panel}
        panel = {m["key"] for m in models if m["dir"] in panel_dirs}
        selected = np.flatnonzero(misclassified(scores[[i for i, m in enumerate(models) if m["key"] in panel]],
                                                labels).any(axis=0))
        splits = {"validation": val}
        for split in ("train", "test"):
            try:
                splits[split] = split_metadata(args.cache_dir, split)
            except FileNotFoundError:
                pass
        context_rows = window_context(splits, scores[0], models[0]["key"])
        previous = []
        if args.previous:
            with open(args.previous) as f:
                previous = [int(os.path.basename(x).lstrip("v").split(".")[0]) for x in f.read().split()]
        payloads = window_payloads(args.cache_dir, "validation", selected)
        listed = build_review_page(models, scores, panel, val, payloads, args.out, context_rows, previous)
        print(f"{args.out}/index.html: {len(listed)} windows misclassified by {len(panel)} panel runs")
        return 0
    if args.command == "blind":
        _, scores = load_runs(args.panel, n)
        reviewed = np.flatnonzero(misclassified(scores, labels).any(axis=0))
        picked, source, population = blind_sample(n, reviewed, args.seed, args.n_random, args.n_errors)
        payloads = window_payloads(args.cache_dir, "validation", picked)
        build_blind_page(val, picked, source, payloads, args.out, args.key, population, args.seed)
        print(f"{args.out}/index.html: {len(picked)} windows; key in {args.key} (keep it out of the page)")
        return 0
    if args.command == "correct":
        verdicts = [r for path in args.verdicts for r in read_records(path)]
        new_labels, keep, counts = corrected_labels(labels, verdicts)
        os.makedirs(args.out, exist_ok=True)
        np.save(os.path.join(args.out, "validation_labels_corrected.npy"), new_labels)
        np.save(os.path.join(args.out, "validation_keep.npy"), keep)
        print(f"{counts['label']} labels flipped, {counts['model']} confirmed, {counts['other']} left out "
              f"-> {int(keep.sum())} windows")
        if args.runs:
            import train_openwhistle as tw

            models, scores = load_runs(args.runs, n)
            rows = []
            for model, s in zip(models, scores):
                before, after = metrics(labels, s), metrics(new_labels, s, keep)
                row = {"run": model["key"], "accuracy_original": before["accuracy"],
                       "accuracy_corrected": after["accuracy"], "precision_corrected": after["precision"],
                       "recall_corrected": after["recall"]}
                by_type = tw.accuracy_by_recording_type(new_labels[keep], s[keep], val["recording"][keep])
                row.update({f"accuracy_corrected_{k}": v for k, v in by_type.items()})
                rows.append(row)
            rows.sort(key=lambda r: -r["accuracy_corrected"])
            write_csv(os.path.join(args.out, "ranking.csv"), rows)
            for row in rows:
                print(f"  {row['run']:<30} {100 * row['accuracy_original']:6.2f}% -> {100 * row['accuracy_corrected']:6.2f}%")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
