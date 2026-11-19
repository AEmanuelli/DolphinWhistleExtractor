# Training a whistle detector on OpenWhistle-CNN

This page explains how to train a new whistle / noise classifier with this repository, on the public
[OpenWhistle-CNN](https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN) dataset of the OpenWhistle paper
([arXiv:2609.34839](https://arxiv.org/abs/2609.34839)), and how to compare it fairly with the existing models.
The script is [`Train_model/train_openwhistle.py`](../Train_model/train_openwhistle.py) (PyTorch).

To reproduce the paper's own VGG16 with the authors' code instead, see
[REPRODUCE_OPENWHISTLE_CNN.md](REPRODUCE_OPENWHISTLE_CNN.md).

## 1. Data

| Split | Windows | Sessions | Years | Used for |
|---|---|---|---|---|
| train | 53,828 | 195 | 2021–2024 | fitting the weights |
| validation | 5,980 | 26 | 2021–2024 | choosing the epoch and the configuration |
| test | 16,708 | 261 | 2019–2020 | the final figure, once, for the chosen model |

Windows are 0.4 s long, whistle and noise are balanced, and no session is shared between splits (the leakage checks
are in [REPRODUCE_OPENWHISTLE_CNN.md](REPRODUCE_OPENWHISTLE_CNN.md#4-is-there-train--test-leakage)). Each window
comes with its 224×224 spectrogram, which is exactly the image the detection pipeline computes with its default
settings (`--spectrogram paper`). A model trained on these images can therefore be used as is by `main.py`.

```bash
pip install torch torchvision pyarrow pillow scikit-learn
python Train_model/train_openwhistle.py prepare
```

`prepare` downloads the 17 parquet files (about 4 GB) into `openwhistle_cnn_cache/` (shared with
`evaluate_models.py`) and decodes the spectrograms once into uint8 `.npy` files (about 3.6 GB), which training then
reads from disk.

`prepare --frontends wide whiten pcen` also computes, from the audio of each window, the images of the other input
representations of [`Predict_and_extract/frontends.py`](../Predict_and_extract/frontends.py) (about 3.6 GB each, see
`--frontend` below).

## 2. One training run

```bash
python Train_model/train_openwhistle.py train --arch mobilenet_v2 --epochs 6 --lr 3e-4 \
    --time_shift 0.25 --time_masks 2 --freq_masks 2 --out runs/mnv2_aug
```

What a run does:

- **Model** (`--arch`, defined in [`Predict_and_extract/torch_models.py`](../Predict_and_extract/torch_models.py)):
  - an ImageNet-pretrained torchvision backbone whose last layer is replaced by 2 outputs (noise, whistle), the
    whole network being trained: `mobilenet_v2`, `mobilenet_v3_small`, `mobilenet_v3_large`, `efficientnet_b0`,
    `resnet18`, `resnet50`, `convnext_tiny`, or `vgg16_openwhistle` (the architecture of the paper);
  - or a small CNN for embedded devices, trained from scratch: `tinycnn_8`, `tinycnn_16`, `tinycnn_32`,
    `tinycnn_64` (a strided convolution, then 4 stages of depthwise-separable convolutions whose width doubles
    from 8 / 16 / 32 / 64 channels; 11 k to 546 k parameters);
  - or a small network from keyword spotting and sound event detection, also trained from scratch:
    `tinycnn_16_coord` and `tinycnn_32_coord` (the small CNN with a second input channel giving the frequency of
    each row), `bcresnet_1` to `bcresnet_8` (BC-ResNet, Kim et al. 2021: frequency-wise depthwise convolutions with
    sub-spectral normalisation, then time-wise convolutions broadcast over frequency; 8 k to 310 k parameters),
    `crnn_8` and `crnn_16` (4 convolution blocks, then a bidirectional GRU over time), `fdycrnn_8` and `fdycrnn_16`
    (the same CRNN with frequency dynamic convolutions, Nam et al. 2022: each frequency row mixes 4 kernels with its
    own attention weights).
- **Input**: the grayscale spectrogram replicated on 3 channels, scaled to 0–1 and normalised with the ImageNet
  mean / std, as in the paper. `--img_size` trains at a lower resolution (e.g. 112 or 64): the 224×224 images are
  resized bilinearly, and the pipeline does the same with the size stored in the checkpoint.
- **Input representation** (`--frontend`, after `prepare --frontends`): `paper` (default, the dataset's
  2–22 kHz spectrograms), or images computed from the audio over 2–48 kHz (224 bands, 1024-sample Blackman
  window): `wide` (log power), `whiten` (log power minus the median of each band over the window, then of each
  frame, which removes stationary tonal noise and broadband clicks) or `pcen` (per-channel energy normalisation,
  Wang et al. 2017). The checkpoint stores its front-end and the pipeline computes the same images (0.4 s windows
  at 96 kHz).
- **Optimisation**: AdamW, cross-entropy, linear warm-up then cosine decay of the learning rate, batch size 32.
  `--loss gce` uses the generalised cross-entropy (Zhang & Sabuncu 2018, `--gce_q`), which is less sensitive to
  wrong labels; `--ema 0.999` evaluates and saves an exponential moving average of the weights.
- **Label noise and dataset diagnostics**:
  - `--cv_folds 5 --cv_fold k` trains on 4/5 of the train recordings (folds balanced in windows, no recording
    split between folds) and saves the scores of the remaining fold (`oof_scores.npy`, `oof_indices.npy`). The 5
    runs together give every train window an out-of-fold score: a window whose score contradicts its label with
    high confidence is a candidate label error;
  - `--exclude_runs <the 5 fold runs> --exclude_threshold 0.9` trains without the train windows whose out-of-fold
    score contradicts the label by at least 0.9;
  - `--soft_source oof --soft_runs <the 5 fold runs> --soft_weight 0.5` mixes the hard labels with the out-of-fold
    scores as soft targets; `--soft_source teacher --soft_runs <run>` distils a trained model instead (same
    front-end), with `--distill_temperature`;
  - `--exclude_recordings channel_1` leaves out the train recordings whose name contains one of the given strings.
- **Augmentation** (optional, on the GPU): circular time shift (`--time_shift`, fraction of the window), small
  frequency shift (`--freq_shift`, pixels), SpecAugment time / frequency masks (`--time_masks`, `--freq_masks`,
  `--mask_width`), mixup (`--mixup`), label smoothing (`--label_smoothing`).
- **Model selection**: after each epoch the model is scored on the validation split; `best.pt` is the epoch with the
  lowest validation loss, as in the paper. Training stops early after `--patience` epochs without improvement.
- **Outputs** in `--out`: `best.pt`, `history.csv` (losses and validation metrics per epoch), `summary.json`
  (best epoch, validation accuracy per recording type: single-channel 2021 recordings, `channel_0` and
  `channel_1` of the 2023–2024 multichannel recordings, and the cost of the model: parameters, multiply-adds per
  image, median latency of one image on one CPU thread), and `best_validation_scores.npy` (the score of every
  validation window at the best epoch, so that runs can be compared window by window, or re-scored after a label
  correction, without recomputing them). An interrupted run resumes from its last finished epoch (`last.pt`) when
  relaunched with the same command.
- **Tracking** (optional): `--wandb_project <name>` also logs the configuration, the per-epoch metrics and the cost
  of the model to [Weights & Biases](https://wandb.ai) (`pip install wandb`, `wandb login`); an interrupted run
  continues the same W&B run.

All options: `python Train_model/train_openwhistle.py train --help`.

## 3. A sweep

A sweep is a JSON file listing runs (shared settings in `"base"`, one entry per run). The sweep used for the results
below is [`Train_model/sweeps/openwhistle_sweep.json`](../Train_model/sweeps/openwhistle_sweep.json):

```bash
python Train_model/train_openwhistle.py sweep Train_model/sweeps/openwhistle_sweep.json --out runs/sweep \
    --wandb_project openwhistle-cnn-sweep    # optional
```

Runs are trained one after the other; finished runs are skipped, so the command can be relaunched after an
interruption. `runs/sweep/sweep_results.csv` ranks the runs by **validation** accuracy (at their best-validation-loss
epoch), with their size and CPU latency, and the log names the smallest model reaching 95%, 96% and 97% validation
accuracy. A run that fails (for example out of GPU memory) is reported and skipped.

The sweep has two goals:

1. **The best detector**: beat `base_MobileNetV2` (97.9% on the test split) with ImageNet-pretrained backbones at
   224×224 (MobileNetV2, ResNet18, MobileNetV3-Large, EfficientNet-B0), all with time shifts and SpecAugment
   masks, plus one MobileNetV2 with mixup and label smoothing.
2. **The smallest detector above 95%**, for real-time inference on a Raspberry Pi or a smaller board: the small CNNs
   and MobileNetV3-Small at 64, 112 and 224 pixels, plus one run without augmentation to measure its effect.

Real time means classifying one 0.4 s window in less than 0.4 s, spectrogram included. All the models of the sweep
take a few milliseconds per image on one laptop CPU thread (table below); a Raspberry Pi 4 core is several times
slower, which still leaves a wide margin for the small models.

| Model | Parameters | Multiply-adds / image | 1 laptop CPU thread |
|---|---|---|---|
| `tinycnn_8` @ 112 px | 10,914 | 0.8 M | 1.2 ms |
| `tinycnn_16` @ 112 px | 38,210 | 2.2 M | 1.4 ms |
| `tinycnn_32` @ 112 px | 141,954 | 7.3 M | 1.9 ms |
| `tinycnn_64` @ 112 px | 546,050 | 25.7 M | 2.6 ms |
| `mobilenet_v3_small` @ 112 px | 1,519,906 | 15.9 M | 4.4 ms |
| `mobilenet_v3_small` @ 224 px | 1,519,906 | 55.5 M | 6.8 ms |
| `mobilenet_v2` @ 224 px | 2,226,434 | 299.5 M | 25.5 ms |
| `mobilenet_v3_large` @ 224 px | 4,204,594 | 215.3 M | 31.0 ms |
| `efficientnet_b0` @ 224 px | 4,010,110 | 384.5 M | 41.2 ms |
| `resnet18` @ 224 px | 11,177,538 | 1,813.6 M | 57.5 ms |

Choosing the small model: take the smallest one whose **validation** accuracy clears the target with some margin
(the test split is recorded in other years, 2019–2020, and can be harder), then check it once on the test split.

### Sweeps 2 and 3

The first sweep was stopped after 10 of its 15 runs, because its results pointed elsewhere: augmentation did not
help, and all the models failed on largely the same validation windows (154 windows are misclassified by all 9 models
compared, 126 of them labelled whistle). The tools used to check these windows by hand are described in
[LABEL_REVIEW.md](LABEL_REVIEW.md).

- [`openwhistle_sweep2_broad.json`](../Train_model/sweeps/openwhistle_sweep2_broad.json) was stopped after its first
  5 runs: a 5-fold cross-validation of `tinycnn_32` at 112 px over the train recordings, which gives every train
  window an out-of-fold score (validation accuracy of the 5 fold models: 94.5–94.8%; out-of-fold accuracy on the
  train split: 95.7%, with 1,183 train windows contradicted by a score of at least 0.9).
- [`openwhistle_sweep3_focused.json`](../Train_model/sweeps/openwhistle_sweep3_focused.json) uses these scores, and
  repeats the reference configuration with 3 seeds to measure the seed-to-seed spread.

### Results so far (validation split)

The test split has not been used yet. Validation accuracy at the best-validation-loss epoch; `tinycnn_32` at 112 px
without augmentation unless stated (142 k parameters, 7.3 M multiply-adds, about 2 ms per window on one laptop CPU
thread).

| Group | Run | Parameters | Validation accuracy |
|---|---|---|---|
| Sweep 1, best models | `mobilenet_v2` @ 224 px, time shifts + SpecAugment | 2.2 M | 95.65% |
| | `tinycnn_32`, no augmentation | 142 k | 95.30% |
| | `tinycnn_32`, time shifts + SpecAugment | 142 k | 94.18% |
| | `resnet18` @ 224 px, time shifts + SpecAugment | 11.2 M | 94.63% |
| | `tinycnn_8`, time shifts + SpecAugment | 11 k | 93.49% |
| Reference, 3 seeds | `tinycnn_32` | 142 k | 95.00 / 94.85 / 94.80% |
| Label noise | soft out-of-fold targets, 2 seeds | 142 k | 95.30 / 95.38% |
| | without the windows contradicted ≥ 0.9, 2 seeds | 142 k | 95.30 / 94.83% |
| | without the windows contradicted ≥ 0.7 | 142 k | 94.72% |
| | generalised cross-entropy | 142 k | 92.86% |
| Distillation | `tinycnn_32` taught by `mobilenet_v2` (below) | 142 k | 95.40% |
| | `bcresnet_2` taught by `mobilenet_v2` | 24 k | 93.46% |
| Large model | `mobilenet_v2` @ 224 px, no augmentation | 2.2 M | 95.22% |
| | `mobilenet_v2` @ 224 px, without the windows contradicted ≥ 0.9 (stopped during epoch 5) | 2.2 M | 95.10% |
| Architectures | `tinycnn_32_coord` | 142 k | 95.27% |
| | `fdycrnn_8` | 67 k | 94.15% |
| | `crnn_16` | 79 k | 93.61% |
| | `bcresnet_4` / `bcresnet_2` | 84 k / 24 k | 93.46 / 93.16% |
| Input representation | `whiten`, 2 seeds | 142 k | 93.60 / 93.56% |
| | `wide`, 2 seeds | 142 k | 92.66 / 92.36% |
| | `pcen`, 2 seeds | 142 k | 91.86 / 91.44% |
| Training recordings | without the `channel_1` recordings | 142 k | 89.46% |
| | single-channel 2021 recordings only | 142 k | 84.33% |

What these runs show so far:

- Seeds of the same configuration differ by 0.1 to 0.5 points (0.2 for the reference), so differences of a few
  tenths of a point are not meaningful.
- A 142 k-parameter CNN is within 0.25–0.35 points of the best MobileNetV2 (2.2 M parameters, 40 times more
  multiply-adds).
- The best small models come from handling label noise: soft out-of-fold targets or distillation reach 95.3–95.4%,
  about 0.5 points above the mean of the reference seeds (94.9%), which is at the edge of the seed spread.
  Generalised cross-entropy and aggressive cleaning (≥ 0.7) do not help.
- Neither the audio front-ends nor the architectures from keyword spotting and sound event detection beat the
  dataset's spectrogram with the small CNN.
- The validation errors are concentrated on the 2023–2024 multichannel recordings (about 98% accuracy on the
  single-channel 2021 recordings, about 94% on `channel_1`), yet training without them costs 5 to 11 points: they
  are needed.

## 4. Testing the chosen model

Only once a configuration has been chosen on the validation split:

```bash
python Train_model/train_openwhistle.py test runs/sweep/<best run>/best.pt
```

This prints accuracy, precision, recall, specificity and ROC AUC on the 16,708 test windows, with a 95% interval of
the accuracy obtained by resampling whole recordings, i.e. the same figures as the "Model evaluation" table of the
README (`Train_model/evaluate_models.py --models runs/sweep/<best run>/best.pt` gives the same numbers through the
detection pipeline's own loader).

Picking the best of several runs on the test split, or tuning the decision threshold on it, would make the test
figure optimistic: do it on the validation split.

## 5. Using the trained model

```bash
python Predict_and_extract/main.py --recordings /path/to/recordings --saving_folder out/ --model_path runs/sweep/<best run>/best.pt
```

The checkpoint stores its architecture and normalisation; the pipeline loads it without executing code
(`torch.load(..., weights_only=True)`). Do not use `--image_norm`: PyTorch models normalise their inputs themselves.

## 6. Hardware notes

Training time is dominated by the GPU. Measured on a laptop GeForce 940MX (4 GB, 2016), training speed alone
(validation adds a little per epoch):

| Model | Batch | Images / s | One epoch (53,828 windows) |
|---|---|---|---|
| `tinycnn_16` @ 112 px | 128 | ~600 (data loading bound) | ~1.5 min |
| `tinycnn_64` @ 112 px | 128 | 255 | ~3.5 min |
| `mobilenet_v3_small` @ 112 px | 64 | 240 | ~4 min |
| `tinycnn_32` @ 224 px | 64 | 132 | ~7 min |
| `mobilenet_v3_small` @ 224 px | 32 | 73 | ~12 min |
| `mobilenet_v2` @ 160 px | 32 | 38 | ~24 min |
| `resnet18` @ 224 px | 32 | 27 | ~33 min |
| `mobilenet_v3_large` @ 224 px | 32 | 26 | ~34 min |
| `mobilenet_v2` @ 224 px | 32 | 20 | ~45 min |
| `efficientnet_b0` @ 224 px | 32 | 13 | ~68 min |

Another job sharing the GPU roughly halves these figures. The training and validation images are loaded in RAM
(about 3 GB): random reads in the `.npy` files on a hard disk would be far slower than the GPU.

A recent GPU is much faster. Without a GPU, use `--train_fraction` for quick tests only. Old GPUs such as the 940MX
(compute capability 5.0) work with PyTorch wheels whose architecture list includes `sm_50`
(`python -c "import torch; print(torch.cuda.get_arch_list())"`).
