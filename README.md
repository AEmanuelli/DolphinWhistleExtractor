# Dolphin Whistle Detection

A comprehensive tool for detecting and extracting dolphin whistles from audio and video recordings using deep neural networks.

The detection follows the whistle-detection pipeline of the OpenWhistle paper
([arXiv:2609.34839](https://arxiv.org/abs/2609.34839)): non-overlapping 0.4 s windows, 2–22 kHz spectrograms,
whistle sequences of 2–20 s. See [docs/REPRODUCE_OPENWHISTLE_CNN.md](docs/REPRODUCE_OPENWHISTLE_CNN.md) for the
details and for how to reproduce the paper's CNN.

## 🌟 Features

- Automated detection of dolphin whistles using deep learning
- User-friendly GUI
- Command-line interface for advanced users
- WAV and FLAC recordings
- Multi-threaded processing, resumable runs
- Results exported as CSV
- Automatic extraction of the detected whistle segments

## 📋 Prerequisites

- Python 3.9 – 3.11
- `tkinter` for the GUIs (see below)
- `ffmpeg` only if you extract video clips

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AEmanuelli/DolphinWhistleExtractor.git
cd DolphinWhistleExtractor
```

2. Set up a virtual environment (recommended):
```bash
# Using conda
conda create -n DolphinWhistleExtractor python=3.9
conda activate DolphinWhistleExtractor
conda install tk
```
```bash
# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install tkinter (only needed with venv):
   - Ubuntu: `sudo apt-get install python3-tk`
   - Fedora: `sudo dnf install python3-tkinter`
   - macOS: `brew install python-tk`

4. Install dependencies:
```bash
pip install -r requirements.txt
```
   (or use the "Install Dependencies" button of the inference GUI)

## 💻 Usage

### GUI (recommended)

```bash
python Predict_and_extract/GUIs/GUI_inference.py
```

In the GUI:
- Select the model (defaults to `models/base_MobileNetV2.keras`)
- Choose the input folder containing the recordings (`.wav` or `.flac`)
- Select the output folder for results
- Configure processing parameters (start/end times, batch size, frequency cutoffs CLF/CHF, image normalization)
- Click "Start Processing"

### Command line

```bash
cd Predict_and_extract

# Detect whistles and save the positive spectrograms
python main.py --recordings /path/to/recordings --saving_folder /path/to/output --save_p

# Whole recordings (default: first 30 min), then extract the detected whistle sequences as .wav clips
python main.py --recordings /path/to/recordings --saving_folder /path/to/output --end_time -1 --extract

# Use the VGG16 detector published with the OpenWhistle paper (needs: pip install torch torchvision)
curl -L -o model_vgg_final_best.pt https://huggingface.co/dolphinteam/OpenWhistle-CNN-VGG16/resolve/main/model_vgg_final_best.pt
python main.py --recordings /path/to/recordings --saving_folder /path/to/output --model_path model_vgg_final_best.pt

# All options
python main.py --help
```

Main options:

| Option | Default | Meaning |
|---|---|---|
| `--model_path` | `models/base_MobileNetV2.keras` | Keras model (`.keras` / `.h5`) or PyTorch checkpoint of the paper's VGG16 (`.pt`) |
| `--spectrogram` | `paper` | `paper`: 0.4 s non-overlapping windows, FFT 1024 / hop 512, 2–22 kHz. `legacy`: 0.8 s windows every 0.4 s, FFT 2048 with 20 % overlap, 3–20 kHz (for models trained on the older images) |
| `--start_time` / `--end_time` | `0` / `1800` | Portion of each recording to analyse, in seconds (`--end_time -1`: whole file) |
| `--CLF` / `--CHF` | `2` / `22` | Frequency band (kHz) shown in the spectrograms (`3` / `20` with `--spectrogram legacy`) |
| `--threshold` | `0.5` | Minimum score for a window to count as a whistle |
| `--image_norm` | off | Divide images by 255. Only for models trained on normalized images: **not** the bundled ones |
| `--save_p` / `--save` | off | Save positive / all spectrogram images |
| `--extract` | off | Cut the detected whistle sequences out of the recordings (audio clips) |
| `--max_gap` / `--min_duration` / `--max_duration` | `6` / `2` / `20` | Sequence rule for `--extract`, in seconds (see below) |
| `--specific_files` | – | Text file listing the recordings to process (one name per line) |

The last `--recordings` / `--saving_folder` used are remembered in `~/.predict_extract_config.json`.

### Output

```
<saving_folder>/
├── <recording>/
│   ├── <recording>.wav_predictions.csv   # file_name, initial_point, finish_point, confidence
│   ├── positive/                         # spectrograms of the detected windows (--save_p)
│   ├── all/                              # every spectrogram (--save)
│   └── pas_d_extraits/                   # note explaining why nothing was extracted (--extract)
└── extracted_audio/<recording>/extrait_<start>_<end>.wav   # with --extract
```

A recording that already has a predictions CSV is skipped, so an interrupted run can simply be restarted.
A recording that fails (e.g. corrupted file) gets no CSV and is retried on the next run.

## 🔧 How it works

1. **Detection** (`predict_and_extract_online.py`): each recording is resampled to 96 kHz if needed and cut into
   non-overlapping 0.4 s windows. Every window becomes a 224×224 grayscale image of its log-power spectrogram
   (Blackman window of 1024 samples, hop 512, 2–22 kHz, min-max normalised; `utils.py`) and is classified as
   whistle or noise. A detection is reported in the CSV as the window `[initial_point, finish_point]`.
2. **Extraction** (`process_predictions.py`): detections less than 6 s apart are grouped into a sequence;
   sequences shorter than 2 s are dropped and runs longer than 20 s are split. Each sequence is cut out of the
   recording; audio clips keep the original sampling rate and bit depth.
   Video clips (with or without the hydrophone audio) are available from Python via
   `process_prediction_files_in_folder(..., audio_only=False)`.

These are the settings of the OpenWhistle paper and of its [official code](https://github.com/sumbre-lab/OpenWhistle);
the images are the same as those of the
[OpenWhistle-CNN](https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN) dataset.
`--spectrogram legacy` gives the previous behaviour of this repository (0.8 s windows every 0.4 s, 3–20 kHz).

## 📁 Project Structure

```
├── models/                         # pre-trained Keras models (see models/README.txt)
├── Predict_and_extract/
│   ├── main.py                     # command-line entry point
│   ├── predict_and_extract_online.py   # detection pipeline
│   ├── process_predictions.py      # extraction of detected whistles (audio / video)
│   ├── utils.py                    # audio loading, spectrogram images, interval helpers
│   ├── count_whistles.py           # count whistle groups in a predictions CSV
│   ├── GUIs/                       # inference GUI + model testing / comparison GUIs
│   ├── Extraction_with_kaggle/     # notebooks & scripts to run the pipeline on Kaggle
│   └── ...                         # research scripts (see note below)
├── Train_model/                    # dataset creation, training scripts / notebooks, evaluate_models.py
├── Models_vs_years_comparison/     # comparison of the models across recording years
├── docs/                           # how to reproduce the OpenWhistle CNN, leakage checks
├── tests/                          # pytest suite
└── requirements.txt
```

> **Note** — `maintenance.py`, `Show_newly_detected_stuff.py`, `prepare_audio.py`, `test.py`, `predict_online.py`
> and the `Train_model/` scripts are research scripts kept for reference: they contain paths specific to the
> original analysis machines and must be adapted before use.

## 📊 Model evaluation

Models are evaluated on the **test split of the public
[OpenWhistle-CNN](https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN) benchmark**: 16,708 windows of 0.4 s
(8,354 whistle / 8,354 noise) from 261 recordings of 2019-2020; no recording, session or day is shared with its
train / validation splits (2021-2024). Threshold 0.5; the 95% interval of the accuracy resamples whole recordings.

| Model | Accuracy (95% CI) | Precision | Recall | Specificity | ROC AUC |
|---|---|---|---|---|---|
| `base_MobileNetV2` (default) | **97.9%** (97.5-98.2) | 98.0% | 97.7% | 98.0% | 0.993 |
| VGG16 of the OpenWhistle paper (`model_vgg_final_best.pt`) | 97.2% (96.9-97.5) | 96.5% | 98.0% | 96.5% | 0.988 |
| `model_finetuned_vgg.h5` (not in the repository, with `--image_norm`) | 96.4% (95.9-96.9) | 94.5% | 98.7% | 94.2% | 0.992 |
| `base_EfficientNetB0` | 95.0% (94.5-95.4) | 97.5% | 92.2% | 97.7% | 0.978 |
| `base_mobilenetv2_best_val_loss` | 94.5% (94.0-94.9) | 95.8% | 93.0% | 95.9% | 0.972 |

Reproduce with:

```bash
pip install pyarrow            # in addition to requirements.txt
python Train_model/evaluate_models.py                                  # the Keras models of models/
python Train_model/evaluate_models.py --models model_vgg_final_best.pt # the paper's VGG16 (needs torch, torchvision)
python Train_model/evaluate_models.py --source pipeline                # spectrograms recomputed from the audio
python Train_model/evaluate_models.py --models models/model_finetuned_vgg.h5 --scale div255  # the older Keras VGG
```

Notes:

- The row of the paper's VGG16 reproduces the published confusion matrix exactly (TP 8186, FP 295, FN 168, TN 8059,
  i.e. precision 96.52%, recall 97.99%).
- The detection pipeline computes the same images as the benchmark, so these figures are also those of
  `main.py`: on a recording made of 1,500 test windows put end to end, the command line takes the same decision
  as the image-level evaluation for every window.
- **Do not use `--image_norm` with the bundled Keras models.** They expect raw 0-255 images; with images divided by
  255 the three of them predict "noise" for every window (recall 0%, accuracy 50%). The option is ignored for the
  paper's VGG16, which normalises its inputs itself.
- `model_finetuned_vgg.h5` (Keras VGG16 with a 50/20 head, see `models/README.txt`) is the opposite: it needs
  `--image_norm`. On raw 0-255 images it predicts "whistle" for almost every window (accuracy 50.0%, ROC AUC 0.50).
  At threshold 0.5 it leans towards "whistle" (483 false positives for 112 misses); its ROC AUC is on par with
  `base_MobileNetV2`, and a higher threshold (0.9-0.95) gives about 97.6% on this test split. That threshold was
  read off the test split itself, so the figure is optimistic; pick a threshold on the validation split instead.
  Fed with the `legacy` STFT and 3-20 kHz band (on the benchmark's 0.4 s windows) it drops to 87.8%: use the
  default `paper` spectrograms.
- The benchmark tells nothing about which data the bundled Keras models were trained on; if a model was trained on
  recordings of 2019-2020, its figures above are optimistic. The paper's VGG16 was trained on 2021-2024 only.
- With the settings used by this repository before version 2.0 (`--spectrogram legacy`: 0.8 s windows every 0.4 s,
  20% STFT overlap, 3-20 kHz) the bundled models see images they were not trained on; on synthetic 0.8 s windows
  `base_MobileNetV2` dropped to about 94%.

Train / test leakage checks and the comparison with the earlier VGG article (Korkmaz et al. 2023) are in
[docs/REPRODUCE_OPENWHISTLE_CNN.md](docs/REPRODUCE_OPENWHISTLE_CNN.md).

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests use a synthetic recording and a fake model, so they run in a few seconds without TensorFlow.

## 📚 Additional Resources

- [Kaggle](https://www.kaggle.com/code/alexisemanuelli/whistles-detection-transfer-learning-0-95)
- [Model Fine-tuning Guide](https://www.kaggle.com/alexisemanuelli/fine-tune)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 📧 Contact

alexis.emanuelli@psl.eu