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
    from 8 / 16 / 32 / 64 channels; 11 k to 546 k parameters).
- **Input**: the grayscale spectrogram replicated on 3 channels, scaled to 0–1 and normalised with the ImageNet
  mean / std, as in the paper. `--img_size` trains at a lower resolution (e.g. 112 or 64): the 224×224 images are
  resized bilinearly, and the pipeline does the same with the size stored in the checkpoint.
- **Optimisation**: AdamW, cross-entropy, linear warm-up then cosine decay of the learning rate, batch size 32.
- **Augmentation** (optional, on the GPU): circular time shift (`--time_shift`, fraction of the window), small
  frequency shift (`--freq_shift`, pixels), SpecAugment time / frequency masks (`--time_masks`, `--freq_masks`,
  `--mask_width`), mixup (`--mixup`), label smoothing (`--label_smoothing`).
- **Model selection**: after each epoch the model is scored on the validation split; `best.pt` is the epoch with the
  lowest validation loss, as in the paper. Training stops early after `--patience` epochs without improvement.
- **Outputs** in `--out`: `best.pt`, `history.csv` (losses and validation metrics per epoch), `summary.json`
  (best epoch, and the cost of the model: parameters, multiply-adds per image, median latency of one image on one
  CPU thread). An interrupted run resumes from its last finished epoch (`last.pt`) when relaunched with the same
  command.
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
