# Reproducing the OpenWhistle whistle-detection CNN

This page explains how to reproduce the binary whistle detector of

> Mustun et al., *OpenWhistle: A Large-Scale Longitudinal Dataset and Benchmark of Bottlenose Dolphin
> Vocalizations*, [arXiv:2609.34839](https://arxiv.org/abs/2609.34839) (section 4.1 and appendix B).

and how this repository relates to it.

| Resource | Where |
|---|---|
| Official code (training, inference, sequences) | <https://github.com/sumbre-lab/OpenWhistle>, folder `cnn/` |
| Dataset (76,516 windows, audio + spectrogram) | <https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN> |
| Published checkpoint | <https://huggingface.co/dolphinteam/OpenWhistle-CNN-VGG16> (`model_vgg_final_best.pt`) |

## 1. What the paper does

**Windows and spectrograms** (`cnn/utils/model.py` of the official code):

| Step | Value |
|---|---|
| Sampling rate | 96 kHz (other rates are resampled) |
| Windows | 0.4 s, **non-overlapping** |
| STFT | Blackman window of 1024 samples, FFT size 1024, hop 512 |
| Scale | log-power (10·log10), min-max normalised over the whole spectrum |
| Band | 2–22 kHz, low frequencies at the bottom |
| Image | 224×224 (nearest-neighbour resize), replicated on 3 channels |
| Model input | pixels / 255, then ImageNet mean / std normalisation |

**Model**: VGG16 convolutional backbone initialised from ImageNet (torchvision `IMAGENET1K_V1`), whose classifier
is replaced by fully connected layers of 50 and 20 units (ReLU) and a 2-way output (noise / whistle). The whole
network is fine-tuned. The 50/20 head follows Korkmaz et al. 2023 (see section 5).

**Training** (`cnn/utils/config.py`): cross-entropy loss, Adam, learning rate 1e-5, batch size 4, at most 50 epochs,
early stopping with patience 10 on the validation loss, `ReduceLROnPlateau` (factor 0.5, patience 5, minimum 1e-6),
mixed precision on GPU, seed 7, no data augmentation. The checkpoint with the lowest validation loss is kept
(epoch 4 for the published model).

**Data**: balanced whistle / noise windows, split by recording session. Train 53,828 windows (195 sessions,
2021–2024), validation 5,980 (26 sessions, 2021–2024), test 16,708 (261 sessions, 2019–2020).

**Sequences** (`cnn/create_sequences_whistles.py`): positive windows less than 6 s apart are grouped; sequences
shorter than 2 s are dropped and runs longer than 20 s are split.

**Reported test result**: precision 96.52 %, recall 97.99 % (accuracy 97.23 %; TN 8059, FP 295, FN 168, TP 8186).

## 2. Reproduction pipeline

### Step 0 — environment

```bash
git clone https://github.com/sumbre-lab/OpenWhistle.git
cd OpenWhistle
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r cnn/requirements.txt
```

Python 3.12 is the version recommended by the authors. A GPU is needed for training (see step 2); on a
machine without one, install the CPU wheels of PyTorch first (see `cnn/README.md`).

### Step 1 — reproduce the published numbers (no training)

```bash
python cnn/train.py --test-only --no-wandb-enabled
```

This downloads the dataset and the published checkpoint, checks that no session is shared between splits, and
evaluates on the 16,708 test windows. Expected: accuracy ≈ 0.972, precision ≈ 0.965, recall ≈ 0.980. Reports
are written to `cnn/runs/reports/` (`test_summary.json`, confusion matrix, ROC).

The same check can be done from this repository, which has an independent implementation of the spectrogram
and of the model loading (see section 3):

```bash
pip install torch torchvision pyarrow
curl -L -o model_vgg_final_best.pt \
  https://huggingface.co/dolphinteam/OpenWhistle-CNN-VGG16/resolve/main/model_vgg_final_best.pt
python Train_model/evaluate_models.py --models model_vgg_final_best.pt
```

### Step 2 — retrain from ImageNet weights

```bash
python cnn/train.py --no-wandb-enabled
```

Defaults are the paper's recipe (section 1). One epoch is 53,828 windows at batch size 4, i.e. 13,457 optimisation
steps; the published run stopped improving after epoch 4 and early stopping needs 10 more epochs, so plan for
about 15 epochs, i.e. roughly 200,000 steps: a matter of hours on one GPU (my estimate, not measured) and not
practical on CPU. The best checkpoint and a `run_summary.json` are written under `cnn/runs/`.

Training reads the `spectrogram` column of the dataset by default. To also reproduce the image generation, train
from the audio instead (the images are recomputed with the parameters of section 1 and cached):

```bash
python cnn/train.py --no-wandb-enabled --train-input-source audio
```

### Step 3 — evaluate the retrained model

```bash
python cnn/train.py --test-only --no-wandb-enabled --checkpoint-path cnn/runs/models/<your checkpoint>.pt
```

Compare with section 1. A single retraining will not give exactly the published confusion matrix (GPU
non-determinism, mixed precision); the authors' learning-curve experiment (`cnn/learning_curve/`) repeats the
training with seeds 7, 17 and 27 to measure this variability, and is the right reference for "how close is close
enough":

```bash
python cnn/learning_curve/run_learning_curve.py --fractions 1.0 --seeds 7 17 27 --num-workers 8
```

### Step 4 — run the detector on recordings

With the official code:

```bash
python cnn/inference.py --recordings-dir /path/to/recordings --output-dir cnn/runs/inference --recursive
python cnn/create_sequences_whistles.py cnn/runs/inference --output-csv cnn/runs/inference/whistle_sequences.csv
```

or with this repository, which applies the same windows, spectrograms and sequence rule and also cuts the
sequences out of the recordings:

```bash
python Predict_and_extract/main.py --recordings /path/to/recordings --saving_folder out/ \
    --model_path model_vgg_final_best.pt --end_time -1 --extract
```

### Checklist of what must match

1. Splits: sessions disjoint between train, validation and test (the official loader refuses to run otherwise).
2. Spectrogram parameters of section 1, in particular 0.4 s windows, FFT 1024 / hop 512 and the 2–22 kHz band.
3. ImageNet normalisation of the inputs and ImageNet initialisation of the backbone.
4. Model selection on the validation loss only; the test split is used once, at the end.

## 3. What this repository does

Since version 2.0 the detection pipeline of this repository follows the paper by default:

- `Predict_and_extract/utils.py`: `PAPER_SPECTROGRAM` holds the parameters of section 1; `spectrogram_image`
  reproduces the images of the OpenWhistle-CNN dataset (checked in `tests/test_utils.py` on two windows of the
  dataset: identical up to one grey level on a few pixels, a float rounding effect between library versions).
- `Predict_and_extract/predict_and_extract_online.py`: non-overlapping 0.4 s windows; `TorchWhistleDetector` loads
  the published VGG16 checkpoint (weights only) and applies the ImageNet normalisation. The Keras models of
  `models/` are run on the same images (raw 0–255 pixels).
- `Predict_and_extract/process_predictions.py`: sequences with the 6 s / 2 s / 20 s rule.
- `--spectrogram legacy` restores the previous behaviour (0.8 s windows every 0.4 s, FFT 2048 with 20 % overlap,
  3–20 kHz), which is the set-up of Korkmaz et al. 2023 and of the older VGG `.h5` models.

## 4. Is there train / test leakage?

Two different "overlaps" must not be confused:

- the **STFT overlap** (hop 512 for a 1024-sample window, i.e. 50 %) is the overlap between FFT frames *inside*
  one spectrogram. It only sets the time resolution of the image and cannot leak anything between splits;
- the **overlap between windows** (a 0.8 s window every 0.4 s shares half of its audio with its neighbours) is
  what creates near-duplicate images. If such windows are split at random, the same piece of whistle ends up in
  both the training and the evaluation set and the score is inflated.

Checked on the published dataset (`artifacts/openwhistle_simple_metadata.csv`, 76,516 windows):

| Check | Result |
|---|---|
| Recordings shared between train / validation / test | 0 |
| Sessions shared (channels of one recording count as one session) | 0 |
| Calendar days shared between train and test, validation and test | 0 (test is 2019–2020, the rest 2021–2024) |
| Calendar days shared between train and validation | 14 (different sessions of the same day) |
| Windows overlapping in time across two splits | 0 |
| Windows overlapping another window of the same split | test 614 (3.7 %), train 192 (0.4 %), validation 28 (0.5 %) |

So the paper does not have the leak: windows do not overlap by construction and the splits are disjoint by
session, the test split being recorded in other years than the training data. Two minor remarks:

- 3.7 % of the test windows partly overlap another test window (manually annotated whistles that are not on the
  0.4 s grid). This is redundancy inside the test split, not leakage; it makes the test slightly less
  informative than its size suggests, which is why `evaluate_models.py` reports an interval obtained by
  resampling whole recordings.
- train and validation share 14 calendar days. This can only make the validation loss, hence early stopping,
  slightly optimistic; it does not touch the test figures.

By contrast, the notebooks `Train_model/fine-tune.ipynb` and `fine-tune2.ipynb` of this repository split
overlapping windows at random for validation, so their validation accuracies are inflated.

## 5. The other VGG article

The paper cites, for the 50/20 fully connected head:

> Korkmaz, Diamant, Danino and Testolin, *Automated detection of dolphin whistles with convolutional networks and
> transfer learning*, Frontiers in Artificial Intelligence 6:1099022, 2023,
> [doi:10.3389/frai.2023.1099022](https://doi.org/10.3389/frai.2023.1099022).

That article trains a VGG16 (ImageNet weights, fully connected layers of 50 and 20 units, softmax output, all
layers trainable) in Keras on recordings from the dolphin reef of Eilat:

| | Korkmaz et al. 2023 | OpenWhistle (2026) |
|---|---|---|
| Framework | Keras / TensorFlow | PyTorch |
| Windows | shifted by 0.4 s; 0.8 s long in the MATLAB script kept in this repository, hence overlapping | 0.4 s, non-overlapping |
| STFT | MATLAB, Blackman 2048, hop 0.8 × window | Blackman 1024, hop 512 |
| Band | 3–20 kHz | 2–22 kHz |
| Input scaling | pixels / 255 | pixels / 255, then ImageNet mean / std |
| Loss, optimiser | binary cross-entropy, Adam 1e-5, patience 15 | cross-entropy, Adam 1e-5, patience 10 |
| Train / test split | by date (24–30 July vs 13–15 July) | by session and year |
| Validation | random 5-fold of the training windows | separate sessions |
| Test accuracy | 92.3 % | 97.2 % |

`Train_model/Create_dataset/save_spectrogram.m`, `Download EilatDS.py`, the `model_vgg.h5` model mentioned in
`models/README.txt` and the `legacy` spectrogram settings of this repository all come from that first set-up.
Its test set is recorded on other days than its training set, so its test accuracy is not affected by window
overlap; only its cross-validation folds are (random folds over overlapping windows).
