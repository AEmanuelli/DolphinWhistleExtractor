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

- **Model**: an ImageNet-pretrained backbone from torchvision (`mobilenet_v2`, `mobilenet_v3_large`,
  `efficientnet_b0`, `resnet18`, `resnet50`, `convnext_tiny`, or `vgg16_openwhistle`, the architecture of the
  paper) whose last layer is replaced by 2 outputs (noise, whistle). The whole network is trained.
- **Input**: the grayscale spectrogram replicated on 3 channels, scaled to 0–1 and normalised with the ImageNet
  mean / std, as in the paper.
- **Optimisation**: AdamW, cross-entropy, linear warm-up then cosine decay of the learning rate, batch size 32.
- **Augmentation** (optional, on the GPU): circular time shift (`--time_shift`, fraction of the window), small
  frequency shift (`--freq_shift`, pixels), SpecAugment time / frequency masks (`--time_masks`, `--freq_masks`,
  `--mask_width`), mixup (`--mixup`), label smoothing (`--label_smoothing`).
- **Model selection**: after each epoch the model is scored on the validation split; `best.pt` is the epoch with the
  lowest validation loss, as in the paper. Training stops early after `--patience` epochs without improvement.
- **Outputs** in `--out`: `best.pt`, `history.csv` (losses and validation metrics per epoch), `summary.json`.
  An interrupted run resumes from its last finished epoch (`last.pt`) when relaunched with the same command.

All options: `python Train_model/train_openwhistle.py train --help`.

## 3. A sweep

A sweep is a JSON file listing runs (shared settings in `"base"`, one entry per run). The sweep used for the results
below is [`Train_model/sweeps/openwhistle_sweep.json`](../Train_model/sweeps/openwhistle_sweep.json):

```bash
python Train_model/train_openwhistle.py sweep Train_model/sweeps/openwhistle_sweep.json --out runs/sweep
```

Runs are trained one after the other; finished runs are skipped, so the command can be relaunched after an
interruption. `runs/sweep/sweep_results.csv` ranks the runs by **validation** accuracy (at their best-validation-loss
epoch). A run that fails (for example out of GPU memory) is reported and skipped.

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

Training time is dominated by the GPU. Measured on a laptop GeForce 940MX (4 GB, 2016):

<!-- filled in after the sweep -->

A recent GPU is much faster. Without a GPU, use `--train_fraction` for quick tests only. Old GPUs such as the 940MX
(compute capability 5.0) work with PyTorch wheels whose architecture list includes `sm_50`
(`python -c "import torch; print(torch.cuda.get_arch_list())"`).
