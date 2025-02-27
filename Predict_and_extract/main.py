#!/usr/bin/env python3
"""
Dolphin Whistle Detection and Extraction Tool

This script processes audio recordings to detect and extract dolphin whistle segments
using a pre-trained deep neural network model. It supports batch processing with
configurable parameters and multi-threaded execution.

Usage:
    ./dolphin_whistle_detector.py [OPTIONS]

Examples:
    # Basic usage with default parameters
    ./dolphin_whistle_detector.py
    
    # Process specific files with custom frequency range
    ./dolphin_whistle_detector.py --specific_files file_list.txt --CLF 5 --CHF 25 --save_p
"""

import argparse
import json
import os
import sys
from predict_and_extract_online import process_predict_extract


def read_file_list(file_path):
    """
    Read a list of files from a text file.
    
    Args:
        file_path (str): Path to the text file containing the list of files.
        
    Returns:
        list: A list of file paths.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read().splitlines()
    except (IOError, FileNotFoundError) as e:
        print(f"Error reading file list '{file_path}': {e}")
        sys.exit(1)


def load_config(config_path):
    """
    Load JSON configuration if it exists; otherwise, use default values and save.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"Configuration loaded from: {config_path}")
                return config
        except json.JSONDecodeError:
            print(f"Error: Configuration file at '{config_path}' is corrupted or not valid JSON.")
            print("Using default configuration and overwriting the file.")
            # Fall through to use default config and overwrite
        except IOError as e:
            print(f"Error reading configuration file '{config_path}': {e}")
            print("Using default configuration.")
            # Fall through to use default config

    # Configuration file doesn't exist or had an error - use default values
    default_recordings_folder = "recordings_default"  # Define your default path
    default_saving_folder = "saving_folder_default"    # Define your default path

    config = {"recordings": default_recordings_folder, "saving_folder": default_saving_folder}

    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Default configuration saved to: {config_path}")
        return config
    except IOError as e:
        print(f"Error writing configuration file '{config_path}': {e}")
        print("Using default configuration in memory, but configuration was not saved to disk.")
        return config # Return the config in memory even if saving failed


def main():
    """
    Main function that parses command-line arguments and executes the processing pipeline.
    """
    # Default parameters
    default_model_path = "DNN_whistle_detection/models/model_vgg.h5"
    default_root = "/media/DOLPHIN_ALEXIS/Analyses_alexis/2023_analysed/"
    config_path = os.path.expanduser("~/.predict_extract_config.json")
    
    # Load saved configuration
    config = load_config(config_path)
    default_recordings = config.get("recordings", "")
    default_saving_folder = config.get("saving_folder", "")
    
    # Other default values
    default_start_time = 0
    default_end_time = None
    default_batch_size = 64
    default_max_workers = 8
    default_CLF = 3  # Cut low frequency
    default_CHF = 20  # Cut high frequency

    # Set up command-line arguments
    parser = argparse.ArgumentParser(
        description="Process predictions and extract segments from dolphin audio recordings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and path parameters
    parser.add_argument('--model_path', default=default_model_path, 
                        help='Path to the trained model file')
    parser.add_argument('--root', default=default_root, 
                        help='Path to the root directory for relative paths')
    parser.add_argument('--recordings', default=default_recordings, 
                        help='Path to recordings folder')
    parser.add_argument('--saving_folder', default=default_saving_folder, 
                        help='Path to saving folder for extracted segments')
    
    # Processing parameters
    parser.add_argument('--start_time', type=int, default=default_start_time, 
                        help='Start time for processing (seconds)')
    parser.add_argument('--end_time', type=int, default=default_end_time, 
                        help='End time for processing (seconds, None for entire recording)')
    parser.add_argument('--batch_size', type=int, default=default_batch_size, 
                        help='Batch size for model inference')
    parser.add_argument('--max_workers', type=int, default=default_max_workers, 
                        help='Maximum number of parallel workers')
    
    # Frequency filtering parameters
    parser.add_argument('--CLF', type=int, default=default_CLF, 
                        help='Cut low frequency (kHz)')
    parser.add_argument('--CHF', type=int, default=default_CHF, 
                        help='Cut high frequency (kHz)')
    
    # File selection parameter
    parser.add_argument('--specific_files', 
                        help='Path to a file containing list of specific files to process')
    
    # Output options
    parser.add_argument('--save', action='store_true', 
                        help='Flag to save all processed segments')
    parser.add_argument('--save_p', action='store_true', 
                        help='Flag to save only positive (detected) segments')
    
    # Processing options
    parser.add_argument('--image_norm', action='store_true', 
                        help='Normalize spectrograms by dividing by 255')
    
    # Add version info
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    args = parser.parse_args()
    
    # Validate key parameters
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        sys.exit(1)
        
    if not os.path.exists(args.recordings):
        print(f"Error: Recordings folder not found at '{args.recordings}'")
        sys.exit(1)
        
    # Create saving folder if it doesn't exist
    if not os.path.exists(args.saving_folder):
        try:
            os.makedirs(args.saving_folder)
            print(f"Created saving folder: {args.saving_folder}")
        except OSError as e:
            print(f"Error creating saving folder '{args.saving_folder}': {e}")
            sys.exit(1)
    
    # Read specific files list if provided
    specific_files = None
    if args.specific_files:
        if not os.path.exists(args.specific_files):
            print(f"Error: Specific files list not found at '{args.specific_files}'")
            sys.exit(1)
        specific_files = read_file_list(args.specific_files)
        print(f"Processing {len(specific_files)} files from list: {args.specific_files}")
    
    try:
        # Process predictions and extract segments
        print(f"Starting whistle detection with {args.max_workers} workers...")
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
        print("Processing completed successfully.")
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()