import argparse
import json
import os

from predict_and_extract_online import process_predict_extract


def read_file_list(file_path):
    """Read a list of files from a text file."""
    with open(file_path, 'r') as file:
        return file.read().splitlines()


def load_config(config_path):
    """Load JSON configuration if it exists; otherwise, prompt for input and save it."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    recordings = input("Enter recordings folder path for first-time setup: ")
    saving_folder = input("Enter saving folder path for first-time setup: ")
    config = {"recordings": recordings, "saving_folder": saving_folder}
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return config


def main():
    # Default parameters
    default_model_path = "DNN_whistle_detection/models/model_vgg.h5"
    default_root = "/media/DOLPHIN_ALEXIS/Analyses_alexis/2023_analysed/"
    config_path = os.path.expanduser("~/.predict_extract_config.json")
    config = load_config(config_path)
    default_recordings = config.get("recordings", "")
    default_saving_folder = config.get("saving_folder", "")
    default_start_time = 0
    default_end_time = None
    default_batch_size = 64
    default_max_workers = 8
    default_CLF = 3
    default_CHF = 20

    # Set up command-line arguments
    parser = argparse.ArgumentParser(
        description="Process predictions and extract segments from recordings."
    )
    parser.add_argument('--model_path', default=default_model_path, help='Path to the model')
    parser.add_argument('--root', default=default_root, help='Path to the root directory')
    parser.add_argument('--recordings', default=default_recordings, help='Path to recordings folder')
    parser.add_argument('--saving_folder', default=default_saving_folder, help='Path to saving folder')
    parser.add_argument('--start_time', type=int, default=default_start_time, help='Start time')
    parser.add_argument('--end_time', type=int, default=default_end_time, help='End time')
    parser.add_argument('--batch_size', type=int, default=default_batch_size, help='Batch size')
    parser.add_argument('--save', action='store_true', help='Flag to save output')
    parser.add_argument('--save_p', action='store_true', help='Flag to save positive outputs')
    parser.add_argument('--max_workers', type=int, default=default_max_workers, help='Maximum number of workers')
    parser.add_argument('--specific_files', help='Path to a file containing list of files to process')
    parser.add_argument('--CLF', type=int, default=default_CLF, help='Cut low frequency')
    parser.add_argument('--CHF', type=int, default=default_CHF, help='Cut high frequency')
    parser.add_argument('--image_norm', action='store_true', help='Normalize image by dividing by 255')
    args = parser.parse_args()

    specific_files = read_file_list(args.specific_files) if args.specific_files else None

    # Process predictions and extract segments
    process_predict_extract(
        recording_folder_path=args.recordings,
        saving_folder=args.saving_folder,
        CLF=args.CLF,
        CHF=args.CHF,
        image_norm=args.image_norm,
        start_time=args.start_time,
        end_time=args.end_time,
        batch_size=args.batch_size,
        save=args.save,
        save_p=args.save_p,
        model_path=args.model_path,
        max_workers=args.max_workers,
        specific_files=specific_files
    )


if __name__ == "__main__":
    main()
