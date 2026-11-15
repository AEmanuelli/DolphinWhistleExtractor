# Dolphin Whistle Detection

A comprehensive tool for detecting and extracting dolphin whistles from audio and video recordings using deep neural networks.

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

# Whole recordings (default: first 30 min), then extract the detected whistles as .wav clips
python main.py --recordings /path/to/recordings --saving_folder /path/to/output --end_time -1 --extract

# All options
python main.py --help
```

Main options:

| Option | Default | Meaning |
|---|---|---|
| `--model_path` | `models/base_MobileNetV2.keras` | Keras model used for detection |
| `--start_time` / `--end_time` | `0` / `1800` | Portion of each recording to analyse, in seconds (`--end_time -1`: whole file) |
| `--CLF` / `--CHF` | `3` / `20` | Frequency band (kHz) shown in the spectrograms |
| `--threshold` | `0.5` | Minimum score for a window to count as a whistle |
| `--image_norm` | off | Divide images by 255. Only for models trained on normalized images: **not** the bundled ones |
| `--save_p` / `--save` | off | Save positive / all spectrogram images |
| `--extract` | off | Cut the detected whistles out of the recordings (audio clips) |
| `--specific_files` | – | Text file listing the recordings to process (one name per line) |

The last `--recordings` / `--saving_folder` used are remembered in `~/.predict_extract_config.json`.

### Output

```
<saving_folder>/
├── <recording>/
│   ├── <recording>.wav_predictions.csv   # file_name, initial_point, finish_point, confidence
│   ├── positive/                         # spectrograms of the detected windows (--save_p)
│   └── pas_d_extraits/                   # note explaining why nothing was extracted (--extract)
└── extracted_audio/<recording>/extrait_<start>_<end>.wav   # with --extract
```

A recording that already has a predictions CSV is skipped, so an interrupted run can simply be restarted.
A recording that fails (e.g. corrupted file) gets no CSV and is retried on the next run.

## 🔧 How it works

1. **Detection** (`predict_and_extract_online.py`): each recording is cut into 0.8 s windows, one every 0.4 s.
   Every window becomes a grayscale spectrogram image (`utils.py`) classified by the CNN.
   In the CSV a detection is reported as the 0.4 s step `[initial_point, finish_point]` at which its window starts.
2. **Extraction** (`process_predictions.py`): detections less than 3 s apart are merged; merged segments longer
   than 5 s are cut out of the recording. Audio clips keep the original sampling rate and bit depth.
   Video clips (with or without the hydrophone audio) are available from Python via
   `process_prediction_files_in_folder(..., audio_only=False)`.

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
├── tests/                          # pytest suite
└── requirements.txt
```

> **Note** — `maintenance.py`, `Show_newly_detected_stuff.py`, `prepare_audio.py`, `test.py`, `predict_online.py`
> and the `Train_model/` scripts are research scripts kept for reference: they contain paths specific to the
> original analysis machines and must be adapted before use.

## 📊 Model evaluation

The bundled models were evaluated on the **test split of the public
[OpenWhistle-CNN](https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN) benchmark**: 16,708 windows of 0.4 s
(8,354 whistle / 8,354 noise) from 261 recordings of 2019-2020, session-disjoint from its train/validation splits.
Threshold 0.5, raw 0-255 images; the 95% interval of the accuracy resamples whole recordings.

| Model | Accuracy (95% CI) | Recall | Specificity | ROC AUC |
|---|---|---|---|---|
| `base_MobileNetV2` (default) | **97.9%** (97.5-98.2) | 97.7% | 98.0% | 0.993 |
| `base_EfficientNetB0` | 95.0% (94.5-95.4) | 92.2% | 97.7% | 0.978 |
| `base_mobilenetv2_best_val_loss` | 94.5% (94.0-94.9) | 93.0% | 95.9% | 0.972 |

Reproduce with:

```bash
pip install pyarrow            # in addition to requirements.txt
python Train_model/evaluate_models.py                     # spectrograms shipped with the dataset
python Train_model/evaluate_models.py --source pipeline   # spectrograms recomputed by this repository
```

What the evaluation showed:

- **Do not use `--image_norm` with the bundled models.** They expect raw 0-255 images; with images divided by 255
  the three of them predict "noise" for every window (recall 0%, accuracy 50%).
- **The scores depend on how the spectrogram is computed.** Feeding the same 0.4 s windows through this repository's
  spectrogram code (`--source pipeline`) gives the figures below. The dataset images are matched by an STFT overlap
  of 80%; the detection pipeline uses 20% (as the original MATLAB script), on 0.8 s windows.

  | Spectrogram fed to the model | `base_MobileNetV2` | `base_EfficientNetB0` | `base_mobilenetv2_best_val_loss` |
  |---|---|---|---|
  | Shipped with the dataset (0.4 s) | 97.9% | 95.0% | 94.5% |
  | Repository code, 0.4 s window, 80% overlap | 97.6% | 95.3% | 95.3% |
  | Repository code, 0.4 s window, 20% overlap | 66.1% | 81.8% | 88.5% |
  | Pipeline setting: 0.8 s window, 20% overlap\* | 94.3% | 93.8% | 91.0% |
  | 0.8 s window, 80% overlap\* | 96.5% | 89.7% | 90.1% |

  \* The benchmark has no 0.8 s windows: these rows use 4,000 synthetic windows made of a test window followed or
  preceded by a noise window of the same recording, so they are indicative only.

  In short, the detection pipeline (0.8 s windows) is expected to score a few points below the benchmark figures,
  and the models work best on 0.4 s windows rendered with 80% overlap, the format of the benchmark images.
- The benchmark tells nothing about which data each bundled model was trained on; if a model was trained on
  recordings of 2019-2020, its figures above are optimistic.

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