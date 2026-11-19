import base64
import csv
import io
import json
import os
import re
import sys

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "Train_model"))

import label_review as lr  # noqa: E402

# (split, recording, labels of consecutive 0.4 s windows starting at 10 s)
RECORDINGS = [
    ("validation", "Exp_A_channel_0", [0, 1, 1, 1, 0, 0, 1, 0]),
    ("validation", "Exp_A_channel_1", [0, 1, 1, 0, 0, 0, 0, 0]),
    ("validation", "Exp_B_2021", [1, 1, 0, 0, 1, 0, 1, 1]),
    ("validation", "Exp_C_channel_0", [0, 1, 0, 1]),
    ("validation", "Exp_D_channel_0", [1, 0, 1, 0]),
    ("train", "Exp_D_channel_1", [1, 1, 0, 0]),
    ("test", "Exp_E_2020", [0, 1]),
]


def wav_bytes(seed):
    from scipy.io import wavfile

    buffer = io.BytesIO()
    wavfile.write(buffer, 96000, np.random.default_rng(seed).integers(-3000, 3000, 384).astype(np.int16))
    return buffer.getvalue()


def png_bytes(seed):
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.full((4, 4), seed % 256, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def cache(tmp_path):
    """A tiny OpenWhistle-CNN cache: the validation split in two parquet files of several row groups."""
    rows = {"train": [], "validation": [], "test": []}
    for split, recording, labels in RECORDINGS:
        for k, label in enumerate(labels):
            seed = len(rows[split]) + 1000 * ["train", "validation", "test"].index(split)
            rows[split].append({"audio": {"bytes": wav_bytes(seed), "path": None},
                                "spectrogram": {"bytes": png_bytes(seed), "path": None},
                                "label": label, "file_name": f"{recording}_{k}.wav", "recording": recording,
                                "onset": 10 + 0.4 * k, "offset": 10.4 + 0.4 * k})
    for split, split_rows in rows.items():
        parts = [split_rows[:13], split_rows[13:]] if split == "validation" else [split_rows]
        for p, part in enumerate(parts):
            pq.write_table(pa.Table.from_pylist(part), tmp_path / f"{split}-{p:05d}-of-{len(parts):05d}.parquet",
                           row_group_size=5)
    return tmp_path, rows


def fake_run(path, scores, params=None):
    path.mkdir(parents=True)
    np.save(path / lr.SCORES_FILE, np.asarray(scores, dtype=np.float32))
    if params:
        (path / "summary.json").write_text(json.dumps({"cost": {"params": params}}))
    return str(path)


def page_data(path):
    html = open(path, encoding="utf-8").read()
    return json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
                      .group(1).replace("<\\/", "</"))


def test_metadata_and_payloads_follow_the_dataset_order(cache):
    cache_dir, rows = cache
    val = lr.split_metadata(str(cache_dir), "validation")
    assert list(val["label"]) == [r["label"] for r in rows["validation"]]
    n = len(rows["validation"])
    payloads = lr.window_payloads(str(cache_dir), "validation", [0, 12, 13, n - 1])
    for i in (0, 12, 13, n - 1):  # across row groups and parquet files
        assert base64.b64decode(payloads[i]["wav"]) == rows["validation"][i]["audio"]["bytes"]
        assert base64.b64decode(payloads[i]["png"]) == rows["validation"][i]["spectrogram"]["bytes"]
    with pytest.raises(IndexError):
        lr.window_payloads(str(cache_dir), "validation", [n])
    assert lr.session_and_channel("Exp_07_Mar_2023_0945_channel_1") == ("Exp_07_Mar_2023_0945", "1")
    assert lr.session_and_channel("Exp_19_Apr_2021_1245pm") == ("Exp_19_Apr_2021_1245pm", None)


def test_error_overlap_finds_the_shared_core():
    labels = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([[0.1, 0.9, 0.8, 0.1, 0.2, 0.3],
                       [0.2, 0.4, 0.9, 0.2, 0.9, 0.3],
                       [0.3, 0.8, 0.7, 0.6, 0.9, 0.1]])
    result = lr.error_overlap(lr.misclassified(scores, labels), labels)
    assert list(result["core"]) == [0, 2]
    assert [r["windows"] for r in result["by_count"]] == [1, 3, 0, 2]
    assert result["by_count"][3]["labelled_whistle"] == 1 and result["by_count"][3]["labelled_noise"] == 1
    assert len(result["pairs"]) == 3 and all(0 <= p["jaccard"] <= p["overlap"] <= 1 for p in result["pairs"])


def test_window_context(cache):
    cache_dir, _ = cache
    splits = {s: lr.split_metadata(str(cache_dir), s) for s in ("train", "validation", "test")}
    scores = np.linspace(0, 1, len(splits["validation"]["label"]))
    rows = lr.window_context(splits, scores, "m")
    by_key = {(r["recording"], r["onset"]): r for r in rows}

    a = by_key[("Exp_A_channel_0", 10.4)]
    assert a["other_channel"] == "overlapping_window_found"
    assert a["other_channel_windows"].startswith("channel_1 10.4-10.8 s whistle m=")  # only the same time
    assert (a["previous_label"], a["next_label"]) == (0, 1)
    assert (a["whistle_run"], a["run_edge"]) == (3, 1)
    assert by_key[("Exp_A_channel_0", 10.8)]["run_edge"] == 0
    assert by_key[("Exp_A_channel_0", 12.4)]["whistle_run"] == 1
    assert by_key[("Exp_A_channel_0", 11.6)]["nearest_whistle_s"] == pytest.approx(0.4)

    d = by_key[("Exp_D_channel_0", 10.0)]
    assert d["other_channel_windows"] == "channel_1 10.0-10.4 s whistle (train)"  # found in another split
    assert by_key[("Exp_B_2021", 10.0)]["other_channel"] == "single_channel_recording"
    assert by_key[("Exp_C_channel_0", 10.0)]["other_channel"] == "other_channel_not_in_dataset"
    assert lr.other_channel_note(d)["found"] and "split train" in lr.other_channel_note(d)["text"]


def test_review_page_lists_the_panel_errors(cache, tmp_path):
    cache_dir, rows = cache
    labels = np.array([r["label"] for r in rows["validation"]])
    good = np.where(labels == 1, 0.9, 0.1)
    a, b = good.copy(), good.copy()
    a[[1, 4]] = 1 - a[[1, 4]]
    b[[4, 20]] = 1 - b[[4, 20]]
    run_a = fake_run(tmp_path / "runs" / "a", a, params=1234)
    run_b = fake_run(tmp_path / "runs" / "b", b)
    other = fake_run(tmp_path / "runs" / "c", np.full(len(labels), 0.6))
    previous = tmp_path / "previous.txt"
    previous.write_text("v00001\n")
    out = tmp_path / "page"
    assert lr.main(["--cache_dir", str(cache_dir), "page", other, run_a, "--panel", f"{run_a}=Model A", run_b,
                    "--previous", str(previous), "--out", str(out)]) == 0
    data = page_data(out / "index.html")
    assert [it["i"] for it in data["items"]] == [1, 4, 20]
    assert [it["r"] for it in data["items"]] == [1, 2, 2]
    assert [m["key"] for m in data["models"]] == ["a", "b", "c"] and data["models"][0]["name"] == "Model A"
    assert (out / "index.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert data["models"][0]["params"] == 1234 and data["models"][0]["acc"] == pytest.approx(1 - 2 / len(labels))
    for it in data["items"]:
        stored = json.loads((out / it["f"]).read_text())[it["id"]]
        assert base64.b64decode(stored["wav"]) == rows["validation"][it["i"]]["audio"]["bytes"]
        assert it["oc"]["text"]
    assert "__DATA__" not in (out / "index.html").read_text(encoding="utf-8")


def test_blind_page_hides_labels_and_draws_reproducibly(cache, tmp_path):
    cache_dir, rows = cache
    labels = np.array([r["label"] for r in rows["validation"]])
    scores = np.where(labels == 1, 0.9, 0.1)
    scores[[2, 5, 9]] = 1 - scores[[2, 5, 9]]
    run = fake_run(tmp_path / "run", scores)
    keys = []
    for k in range(2):
        out, key = tmp_path / f"blind{k}", tmp_path / f"key{k}.csv"
        lr.main(["--cache_dir", str(cache_dir), "blind", "--panel", run, "--seed", "5", "--n_random", "6",
                 "--n_errors", "2", "--out", str(out), "--key", str(key)])
        keys.append(list(csv.DictReader(open(key))))
    assert keys[0] == keys[1]  # same seed, same sample
    data = page_data(tmp_path / "blind0" / "index.html")
    assert [set(it) for it in data["items"]] == [{"id", "f"}] * 8  # no label, score or recording in the page
    sources = [r["source"] for r in keys[0]]
    assert sources.count("random_correct") == 6 and sources.count("reviewed_error") == 2
    assert all((int(r["index_in_split"]) in (2, 5, 9)) == (r["source"] == "reviewed_error") for r in keys[0])
    assert all(r["label"] == ("whistle" if labels[int(r["index_in_split"])] else "noise") for r in keys[0])
    meta = json.loads((tmp_path / "key0.json").read_text())
    assert meta["population"] == {"random_correct": len(labels) - 3, "reviewed_error": 3}
    with pytest.raises(ValueError):
        lr.blind_sample(10, [1, 2], 0, 9, 1)


def test_corrected_labels_and_ranking(cache, tmp_path):
    cache_dir, rows = cache
    labels = np.array([r["label"] for r in rows["validation"]])
    scores = np.where(labels == 1, 0.9, 0.1)
    scores[[0, 1, 2]] = 1 - scores[[0, 1, 2]]
    run = fake_run(tmp_path / "run", scores)
    folder = tmp_path / "db" / "verdicts"  # database export: one document per window
    folder.mkdir(parents=True)
    (folder / "v00000.json").write_text(json.dumps({"verdict": "label", "tags": []}))
    (folder / "v00001.json").write_text(json.dumps({"verdict": "model"}))
    export = tmp_path / "export.json"  # the page's own JSON export
    export.write_text(json.dumps([{"id": "v00002", "verdict": "unsure", "tags": [], "note": ""}]))
    out = tmp_path / "corrected"
    lr.main(["--cache_dir", str(cache_dir), "correct", "--verdicts", str(tmp_path / "db"), str(export),
             "--runs", run, "--out", str(out)])
    new = np.load(out / "validation_labels_corrected.npy")
    keep = np.load(out / "validation_keep.npy")
    assert new[0] == 1 - labels[0] and (new[1:] == labels[1:]).all()
    assert not keep[2] and keep.sum() == len(labels) - 1
    ranking = list(csv.DictReader(open(out / "ranking.csv")))
    assert float(ranking[0]["accuracy_original"]) == pytest.approx(1 - 3 / len(labels))
    assert float(ranking[0]["accuracy_corrected"]) == pytest.approx(1 - 1 / (len(labels) - 1))


def test_blind_agreement_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    key = [{"blind_id": f"b{k:03d}", "source": "random_correct" if k < 30 else "reviewed_error",
            "label": "whistle" if k % 2 else "noise"} for k in range(40)]
    first = {r["blind_id"]: r["label"] if rng.random() < 0.9 else rng.choice(["whistle", "noise", "unsure"])
             for r in key}
    second = {b: a if rng.random() < 0.8 else rng.choice(["whistle", "noise", "unsure"]) for b, a in first.items()}
    report = lr.blind_agreement(key, {"x": first, "y": second}, {"random_correct": 3000, "reviewed_error": 500})
    pair = report["pairs"][0]
    ids = sorted(first)
    assert pair["kappa_3_classes"] == pytest.approx(
        sklearn_metrics.cohen_kappa_score([first[b] for b in ids], [second[b] for b in ids]))
    entry = report["annotators"]["x"]["random_correct"]
    assert entry["definite"] + entry["unsure"] == 30
    low, high = entry["ci95"]
    assert low <= entry["rate"] <= high
    assert entry["implied_label_errors"][0] == pytest.approx(round(entry["rate"] * 3000, 1))
    assert lr.wilson_interval(0, 50)[0] == 0 and lr.wilson_interval(50, 50)[1] == 1
