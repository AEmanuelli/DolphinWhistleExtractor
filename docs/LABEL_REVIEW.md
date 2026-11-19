# Auditing the labels of OpenWhistle-CNN

The models of [TRAINING.md](TRAINING.md) do not fail at random on the validation split: whatever their size, they
get largely the same windows wrong. Some of these windows are hard, others may be mislabelled. This page describes
the tools used to tell them apart, in [`Train_model/label_review.py`](../Train_model/label_review.py):

1. measure how much the models share their errors (`overlap`);
2. describe the context of every window: the other channel at the same time, the neighbouring labels (`context`);
3. review the misclassified windows by hand, with their image, full-band spectrogram and audio (`page`);
4. review blind a random sample, to estimate the label error rate of the whole split and the agreement between
   annotators (`blind`, then `agreement`);
5. score every model again on the corrected labels (`correct`).

All commands read the dataset cache of `train_openwhistle.py prepare` (`--cache_dir`, default
`openwhistle_cnn_cache/`) and need `pyarrow` and `numpy`; `scores` also needs `torch` (or TensorFlow for a Keras
model).

## 1. Scores of the models

Every run trained with `train_openwhistle.py` saves the whistle score of every validation window at its best epoch
(`best_validation_scores.npy`). A run folder, or `RUN_DIR=Display name`, is what the other commands take.

```bash
python Train_model/label_review.py scores runs/old_sweep/*/      # runs trained before the scores were saved
python Train_model/label_review.py scores --keras models/base_MobileNetV2.keras --out runs/reference_base
```

## 2. Shared errors

```bash
python Train_model/label_review.py overlap runs/sweep/*/ --out runs/validation_error_core.npy
```

prints the number of windows misclassified by 0, 1, … all runs, the size of the core that every run gets wrong
next to its expected size if the runs erred independently, and how much the error sets of every pair of runs
overlap. `--out` saves the indices of the core windows.

## 3. Context of every window

```bash
python Train_model/label_review.py context --scores runs/sweep/mnv2_224 --out runs/validation_context.csv
```

One row per validation window:

- `other_channel`: the 2023–2024 recordings have two channels, `..._channel_0` and `..._channel_1`. The column says
  whether the other channel of the same session has windows at the same time in the dataset (in any split), and
  `other_channel_windows` lists them with their label (and score, for validation windows). Single-channel
  recordings, and sessions with one channel in the dataset, are marked as such;
- `previous_label`, `next_label`: the labels of the adjacent windows of the same recording, when they exist;
- `nearest_whistle_s`: the distance to the nearest other window labelled whistle in the recording;
- `whistle_run`, `run_edge`: for whistle windows, the length of the run of consecutive whistle windows, and whether
  the window is at an edge of it.

## 4. Review page

```bash
python Train_model/label_review.py page runs/sweep/*/ --panel runs/sweep/mnv2_224 runs/sweep/tiny32_112 --out review
python -m http.server -d review 8000      # then open http://localhost:8000
```

The page lists the validation windows misclassified by at least one **panel** run, and shows the scores of every
run given. For each window: the 224×224 image of the dataset (2–22 kHz), the 0–48 kHz spectrogram computed from the
audio, the audio at normal speed and slowed down 4 times (which brings the whole band into hearing), the other
channel (from `context`), and the scores. A verdict is recorded with one of three buttons:

| Verdict | Meaning | Effect in `correct` |
|---|---|---|
| Le modèle se trompe | the label is right | the window keeps its label |
| L'étiquette est fausse | the label is wrong | the label is flipped |
| Ambigu | cannot decide | the window is left out |

Tags (faint whistle, cut at the window edge, tonal noise, clicks…) and a note can be added. Filters select the
panel run, the type of error (missed whistle, false alarm), the recording and the review status. `--previous` takes
a text file of windows already reviewed (`v00510` or `510`, one per line): the others form a second series with its
own filter, so a second review can start where the first one stopped.

Verdicts are saved in the browser (`localStorage`); **Copier les verdicts** copies them as JSON. The page can also
be published as a [Claude artifact](https://claude.ai) with the `db` capability: the verdicts are then stored online,
one document per window (`verdicts/v00510`), and a database export is read by `correct` as a folder of JSON files.

## 5. Blind review

The review page shows the dataset label and the scores, and only covers the windows the models get wrong. Two
questions need a blind review instead:

- how many labels are wrong among the windows the models classify correctly (the rest of the split)?
- would another annotator reach the same verdicts?

```bash
python Train_model/label_review.py blind --panel runs/sweep/mnv2_224 runs/sweep/tiny32_112 \
    --seed <a private number> --n_random 300 --n_errors 100 --out blind --key runs/blind_key.csv
```

The page holds `n_random` windows drawn among those that no panel run misclassifies and `n_errors` drawn among the
reviewed errors, in a random order, numbered 1 to N, without label, score, recording or time. The annotator answers
**Sifflement** (a whistle is visible or audible in the 0.4 s, even short, faint or cut at the edge), **Pas de
sifflement** or **Incertain**, with optional tags and a note. The key linking each number to its window is written
to `--key` (and the seed and population sizes next to it, `blind_key.json`), never into the page.

Blinding relies on the annotators' good faith: the page shows the dataset's own images, which could be matched
against the public dataset. To keep it meaningful:

- keep the seed and the key private until every annotator has finished (the same seed and panel rebuild the same
  sample);
- give annotators only the blind page, not the review page;
- published as a Claude artifact, declare per-annotator rules so that an annotator cannot read the others' answers,
  and ask annotators to fill in their initials:

  ```json
  {"db": {"rules": [{"path": "answers", "read": "owner", "write": "owner"},
                    {"path": "answers/{self}", "read": "interact", "write": "interact"}]},
   "user": {}}
  ```

  Answers are stored in `answers/<annotator id>/w/<b001…>`; only the owner of the artifact reads them all.

Without the online store, each annotator copies their answers with **Copier mes réponses** and sends the JSON.

```bash
python Train_model/label_review.py agreement --key runs/blind_key.csv \
    --answers alice=answers_alice.json bob=db_export/answers/<id>/w --out runs/blind_agreement.json
```

reports, for each annotator and each part of the sample, how often a definite answer contradicts the dataset label,
with a 95% Wilson interval and the number of label errors it implies in the population that part was drawn from;
and for each pair of annotators the raw agreement and Cohen's kappa (with and without the "Incertain" answers).

## 6. Corrected labels

```bash
python Train_model/label_review.py correct --verdicts verdicts.json --runs runs/sweep*/*/ --out runs/corrected
```

writes `validation_labels_corrected.npy` (flipped where the verdict is "label") and `validation_keep.npy` (false for
the windows judged ambiguous or without verdict), and ranks the runs by accuracy on the corrected labels
(`ranking.csv`, with precision, recall and accuracy per recording type).

This ranking favours the panel runs: their errors were reviewed, while the errors that only other runs make were
not. Review the errors of every run compared, or rely on the blind random sample to bound the errors left in the
rest of the split, before drawing conclusions from small differences.
