#!/usr/bin/env python3
"""
Dolphin Whistle Detection and Extraction Tool

Detects dolphin whistles in audio recordings (.wav / .flac) with a pre-trained
neural network and, optionally, extracts the detected segments as audio clips.
By default the processing follows the OpenWhistle paper (arXiv:2609.34839):
non-overlapping 0.4 s windows, 2-22 kHz spectrograms, sequences of 2-20 s.

Examples:
    # Detect whistles in a folder of recordings (model defaults to models/base_MobileNetV2.keras)
    python main.py --recordings /path/to/recordings --saving_folder /path/to/output --save_p

    # Whole recordings, then extract the detected whistle sequences as .wav clips
    python main.py --recordings rec/ --saving_folder out/ --end_time -1 --extract

    # The VGG16 detector published with the paper (PyTorch checkpoint)
    python main.py --recordings rec/ --saving_folder out/ --model_path model_vgg_final_best.pt

    # Only process the files listed (one name per line) in a text file
    python main.py --recordings rec/ --saving_folder out/ --specific_files file_list.txt

The last used --recordings / --saving_folder are remembered in
~/.predict_extract_config.json and used as defaults on the next run.
"""

import argparse
import json
import logging
import os
import sys

from predict_and_extract_online import DEFAULT_MODEL_PATH, process_predict_extract
from process_predictions import DEFAULT_MAX_DURATION, DEFAULT_MAX_GAP, DEFAULT_MIN_DURATION
from utils import SPECTROGRAM_PRESETS

CONFIG_PATH = os.path.expanduser("~/.predict_extract_config.json")


def read_file_list(file_path):
    """Read a list of file names (one per line, blank lines ignored) from a text file."""
    try:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except OSError as e:
        print(f"Error reading file list '{file_path}': {e}")
        sys.exit(1)


def load_config(config_path=CONFIG_PATH):
    """Load the saved configuration (last used folders); empty dict if missing or invalid."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config if isinstance(config, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(config, config_path=CONFIG_PATH):
    """Remember the folders used for the next run (best effort)."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    except OSError as e:
        print(f"Warning: could not save configuration to '{config_path}': {e}")


def build_parser(config=None):
    """Build the command-line parser; ``config`` provides defaults for the folders."""
    config = config or {}
    parser = argparse.ArgumentParser(
        description="Detect dolphin whistles in audio recordings and extract the detected segments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and path parameters
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH,
                        help="Keras model (.keras / .h5) or PyTorch checkpoint of the paper's VGG16 (.pt)")
    parser.add_argument('--recordings', default=config.get("recordings"),
                        required=not config.get("recordings"),
                        help='Path to recordings folder')
    parser.add_argument('--saving_folder', default=config.get("saving_folder"),
                        required=not config.get("saving_folder"),
                        help='Path to the output folder (one sub-folder per recording)')

    # Processing parameters
    parser.add_argument('--start_time', type=float, default=0,
                        help='Start time for processing (seconds)')
    parser.add_argument('--end_time', type=float, default=1800,
                        help='End time for processing (seconds); negative value: entire recording')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for model inference')
    parser.add_argument('--max_workers', type=int, default=0,
                        help='Number of recordings processed in parallel (0: automatic)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Minimum model score for a window to count as a whistle')

    # Frequency filtering parameters
    parser.add_argument('--spectrogram', choices=sorted(SPECTROGRAM_PRESETS), default='paper',
                        help='"paper": non-overlapping 0.4 s windows, 1024-point FFT with hop 512, 2-22 kHz '
                             '(arXiv:2609.34839). "legacy": 0.8 s windows every 0.4 s, 2048-point FFT, 3-20 kHz '
                             '(for models trained on the older images)')
    parser.add_argument('--CLF', type=float, default=None,
                        help='Cut low frequency in kHz (default: 2 for "paper", 3 for "legacy")')
    parser.add_argument('--CHF', type=float, default=None,
                        help='Cut high frequency in kHz (default: 22 for "paper", 20 for "legacy")')

    # File selection parameter
    parser.add_argument('--specific_files',
                        help='Path to a text file listing the files to process (one per line)')

    # Output options
    parser.add_argument('--save', action='store_true',
                        help='Save all spectrogram images')
    parser.add_argument('--save_p', action='store_true',
                        help='Save the spectrogram images of positive (detected) windows')
    parser.add_argument('--extract', action='store_true',
                        help='After detection, extract the detected whistles as audio clips')
    parser.add_argument('--extract_folder',
                        help='Folder receiving the audio clips (default: <saving_folder>/extracted_audio)')
    parser.add_argument('--max_gap', type=float, default=DEFAULT_MAX_GAP,
                        help='--extract: detections closer than this many seconds form one sequence')
    parser.add_argument('--min_duration', type=float, default=DEFAULT_MIN_DURATION,
                        help='--extract: sequences shorter than this many seconds are dropped')
    parser.add_argument('--max_duration', type=float, default=DEFAULT_MAX_DURATION,
                        help='--extract: longer runs of detections are split into sequences of at most this many seconds')

    # Processing options
    parser.add_argument('--image_norm', action='store_true',
                        help='Normalize spectrograms by dividing by 255 (only for models trained on '
                             'normalized images; the models bundled in models/ need raw images)')

    parser.add_argument('--version', action='version', version='%(prog)s 2.0.0')
    return parser


def main(argv=None):
    """Parse command-line arguments and run the detection (and extraction) pipeline."""
    parser = build_parser(load_config())
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Validate key parameters
    if not os.path.exists(args.model_path):
        parser.error(f"Model file not found at '{args.model_path}'")
    if not os.path.isdir(args.recordings):
        parser.error(f"Recordings folder not found at '{args.recordings}'")
    preset = SPECTROGRAM_PRESETS[args.spectrogram]
    low = preset.cut_low_frequency if args.CLF is None else args.CLF
    high = preset.cut_high_frequency if args.CHF is None else args.CHF
    if low >= high:
        parser.error("--CLF must be lower than --CHF")
    if args.max_duration < args.min_duration:
        parser.error("--max_duration must not be lower than --min_duration")

    specific_files = None
    if args.specific_files:
        if not os.path.exists(args.specific_files):
            parser.error(f"Specific files list not found at '{args.specific_files}'")
        specific_files = read_file_list(args.specific_files)
        print(f"Processing {len(specific_files)} files from list: {args.specific_files}")

    os.makedirs(args.saving_folder, exist_ok=True)
    save_config({"recordings": os.path.abspath(args.recordings),
                 "saving_folder": os.path.abspath(args.saving_folder)})

    results = process_predict_extract(
        recording_folder_path=args.recordings,
        saving_folder=args.saving_folder,
        cut_low_freq=args.CLF,
        cut_high_freq=args.CHF,
        image_normalize=args.image_norm,
        start_time=args.start_time,
        end_time=args.end_time if args.end_time >= 0 else None,
        batch_size=args.batch_size,
        save=args.save,
        save_positives=args.save_p,
        model_path=args.model_path,
        binary_threshold=args.threshold,
        max_workers=args.max_workers,
        specific_files=specific_files,
        spectrogram=args.spectrogram,
    )

    failed = sorted(name for name, count in results.items() if count is None)
    detections = sum(count or 0 for count in results.values())
    print(f"Processed {len(results) - len(failed)} file(s): {detections} positive window(s).")

    if args.extract:
        from process_predictions import process_prediction_files_in_folder
        process_prediction_files_in_folder(
            args.saving_folder, args.recordings, max_workers=args.max_workers or 4,
            audio_only=True, exit=args.extract_folder, max_gap=args.max_gap,
            min_duration=args.min_duration, max_duration=args.max_duration)
        print("Extraction completed.")

    if failed:
        print(f"{len(failed)} file(s) failed and will be retried on the next run: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
