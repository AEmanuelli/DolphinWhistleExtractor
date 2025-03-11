import warnings
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import concurrent.futures
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

from utils import process_audio_file, save_csv, transform_file_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and debug messages


@dataclass
class ProcessingConfig:
    """Configuration for audio processing and prediction."""
    batch_duration: float
    batch_size: int
    cut_low_frequency: int
    cut_high_frequency: int
    image_normalize: bool
    save_positive_examples: bool
    binary_threshold: float = 0.5  # Threshold for binary classifier


def prepare_image_batch(images: List[np.ndarray], start_time: float, 
                         image_normalize: bool) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, float, float]]]:
    """
    Prepare a batch of images for model prediction.
    
    Args:
        images: List of spectrograms
        start_time: Start time of the first image
        image_normalize: Whether to normalize image values
        
    Returns:
        Tuple of processed images ready for prediction and time information
    """
    image_batch = []
    time_batch = []
    
    for idx, image in enumerate(images):
        original_image = image.copy()
        image_start_time = round(start_time + idx * 0.4, 2)
        image_end_time = round(image_start_time + 0.4, 2)
        
        # Resize and preprocess image for model
        processed = cv2.resize(image, (224, 224))
        processed = np.expand_dims(processed, axis=0)
        processed = preprocess_input(processed)
        if image_normalize:
            processed = processed / 255
            
        image_batch.append(processed)
        time_batch.append((original_image, image_start_time, image_end_time))
        
    return image_batch, time_batch


def save_positive_prediction(image: np.ndarray, start_time: float, end_time: float, 
                             saving_folder: Path) -> None:
    """
    Save a positive prediction image to disk.
    
    Args:
        image: The spectrogram image
        start_time: Start time of the spectrogram
        end_time: End time of the spectrogram
        saving_folder: Folder to save the image
    """
    saving_positive = saving_folder / "positive"
    saving_positive.mkdir(exist_ok=True)
    
    image_name = saving_positive / f"{start_time}-{end_time}.jpg"
    cv2.imwrite(str(image_name), image)


def is_positive_prediction(prediction: np.ndarray, binary_threshold: float) -> Tuple[bool, float]:
    """
    Determine if a prediction is positive and return confidence score.
    Handles both binary and categorical outputs.
    
    Args:
        prediction: Model prediction output
        binary_threshold: Threshold for binary classification
        
    Returns:
        Tuple of (is_positive, confidence_score)
    """
    # Check prediction shape to determine if binary or categorical
    if prediction.ndim == 0 or (prediction.ndim == 1 and len(prediction) == 1):
        # Binary classifier with single output
        score = float(prediction.item() if prediction.ndim == 0 else prediction[0])
        return score >= binary_threshold, score
    elif prediction.ndim == 1 and len(prediction) == 2:
        # Categorical classifier with two classes
        score = float(prediction[1])
        return prediction[1] > prediction[0], score
    else:
        # Multi-class classifier, use argmax
        positive_class_idx = 1  # Assuming class 1 is the positive class
        class_idx = np.argmax(prediction)
        score = float(prediction[positive_class_idx])
        return class_idx == positive_class_idx, score


def process_and_predict(file_path: str, config: ProcessingConfig, 
                        start_time: float, end_time: Optional[float], 
                        model: tf.keras.Model, saving_folder: Path) -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    Process an audio file, extract batches of spectrograms, and predict the presence of whistles.

    Args:
        file_path: Path to the audio file
        config: Processing configuration
        start_time: Start time in seconds for processing
        end_time: End time in seconds (None for end of file)
        model: Loaded prediction model
        saving_folder: Folder to save results

    Returns:
        Tuple of (record_names, positive_start_times, positive_end_times, confidence_scores)
    """
    try:
        file_name = Path(file_path).name
        transformed_file_name = transform_file_name(file_name)
        fs, x = wavfile.read(file_path)
        
        # Determine processing length
        N = len(x)
        if end_time is not None:
            N = min(N, int(end_time * fs))
            
        total_duration = (N / fs) - start_time
        num_batches = int(np.ceil(total_duration / config.batch_duration))
        
        # Initialize result containers
        record_names = []
        positive_initial = []
        positive_finish = []
        class_1_scores = []
        
        for batch in tqdm(range(num_batches), desc=f"Processing {transformed_file_name}", 
                          leave=False, colour='blue'):
            batch_start = batch * config.batch_duration + start_time
            
            # Extract spectrograms from audio segment
            images = process_audio_file(
                file_path, str(saving_folder), 
                batch_size=config.batch_size, 
                start_time=batch_start, 
                end_time=end_time,
                cut_low_frequency=config.cut_low_frequency, 
                cut_high_frequency=config.cut_high_frequency
            )
            
            if not images:
                continue
                
            # Prepare images for model prediction
            image_batch, time_batch = prepare_image_batch(
                images, batch_start, config.image_normalize
            )
            
            if not image_batch:
                continue
                
            # Stack images and make batch prediction
            stacked_batch = np.vstack(image_batch)
            predictions = model.predict(stacked_batch, verbose=0)
            
            # Process prediction results
            for idx, prediction in enumerate(predictions):
                orig_image, image_start_time, image_end_time = time_batch[idx]
                
                # Check if positive prediction using the general function
                is_positive, confidence = is_positive_prediction(prediction, config.binary_threshold)
                
                if is_positive:
                    record_names.append(file_name)
                    positive_initial.append(image_start_time)
                    positive_finish.append(image_end_time)
                    class_1_scores.append(confidence)
                    
                    # Save positive example if requested
                    if config.save_positive_examples:
                        save_positive_prediction(
                            orig_image, image_start_time, image_end_time, saving_folder
                        )
        
        return record_names, positive_initial, positive_finish, class_1_scores
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return [], [], [], []


def process_single_file(file_name: str, recording_folder: Path, saving_folder: Path, 
                        config: ProcessingConfig, start_time: float, end_time: Optional[float], 
                        model: tf.keras.Model, pbar: tqdm) -> None:
    """
    Process a single audio file for whistle detection.
    
    Args:
        file_name: Name of the audio file
        recording_folder: Folder containing audio files
        saving_folder: Folder to save results
        config: Processing configuration
        start_time: Start time for processing
        end_time: End time for processing (None for full file)
        model: Loaded prediction model
        pbar: Progress bar
    """
    try:
        # Prepare file paths
        file_stem = Path(file_name).stem
        file_path = recording_folder / file_name
        file_saving_folder = saving_folder / file_stem
        prediction_path = file_saving_folder / f"{file_stem}.wav_predictions.csv"
        
        # Create output directory
        file_saving_folder.mkdir(exist_ok=True)
        
        # Skip processing if already done or not a WAV file
        if not file_name.lower().endswith(".wav") or prediction_path.exists():
            logger.info(f"Skipping {file_name}: Already processed or not a WAV file")
            return
            
        logger.info(f"Processing: {file_stem}")
        
        # Process audio and get predictions
        record_names, positive_initial, positive_finish, class_1_scores = process_and_predict(
            str(file_path), config, start_time, end_time, model, file_saving_folder
        )
        
        # Save results if any positive detections found
        if record_names:
            save_csv(record_names, positive_initial, positive_finish, class_1_scores, str(prediction_path))
            logger.info(f"Saved {len(record_names)} detections for {file_name}")
        else:
            logger.info(f"No detections found in {file_name}")
            
    except Exception as e:
        logger.error(f"Error processing {file_name}: {str(e)}")
    finally:
        pbar.update(1)


def detect_model_output_type(model: tf.keras.Model) -> str:
    """
    Detect the output type of the model by examining its architecture.
    
    Args:
        model: The loaded Keras model
        
    Returns:
        String indicating the output type: 'binary', 'categorical', or 'unknown'
    """
    try:
        # Get the last layer of the model
        last_layer = model.layers[-1]
        
        # Check activation function
        if hasattr(last_layer, 'activation'):
            activation_name = last_layer.activation.__name__
            
            # Check output shape
            output_shape = last_layer.output_shape
            
            # Binary classification typically has sigmoid activation with shape (None, 1)
            if activation_name == 'sigmoid' and output_shape[-1] == 1:
                return 'binary'
            
            # Categorical classification typically has softmax activation with shape (None, n)
            elif activation_name == 'softmax':
                return 'categorical'
        
        # Check by running a test prediction
        dummy_input = tf.zeros((1,) + model.input_shape[1:])
        output = model.predict(dummy_input, verbose=0)
        
        if output.shape[-1] == 1:
            return 'binary'
        elif output.shape[-1] > 1:
            return 'categorical'
        
        return 'unknown'
        
    except Exception as e:
        logger.warning(f"Could not detect model type: {str(e)}")
        return 'unknown'


def process_predict_extract(recording_folder_path: str, saving_folder: str, 
                            cut_low_freq: int = 3, cut_high_freq: int = 20, 
                            image_normalize: bool = False, start_time: float = 0, 
                            end_time: Optional[float] = 1800, batch_size: int = 50,
                            save: bool = False, save_positives: bool = True, 
                            model_path: str = "models/model_vgg.h5", 
                            binary_threshold: float = 0.5,
                            max_workers: int = 16, specific_files: Optional[List[str]] = None) -> None:
    """
    Process and extract predictions from multiple audio files.
    
    Args:
        recording_folder_path: Path to folder containing audio files
        saving_folder: Path to save results
        cut_low_freq: Low frequency cutoff for spectrograms
        cut_high_freq: High frequency cutoff for spectrograms
        image_normalize: Whether to normalize images
        start_time: Start time for processing audio files
        end_time: End time for processing (None for full files)
        batch_size: Number of spectrograms per batch
        save: Whether to save intermediate results
        save_positives: Whether to save positive detection spectrograms
        model_path: Path to the trained model
        binary_threshold: Threshold for binary classification (0-1)
        max_workers: Maximum number of concurrent workers
        specific_files: List of specific files to process (None for all files)
    """
    # Convert paths to Path objects
    recording_folder = Path(recording_folder_path)
    saving_folder = Path(saving_folder)
    saving_folder.mkdir(exist_ok=True)
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Detect model type
        model_type = detect_model_output_type(model)
        logger.info(f"Detected model type: {model_type}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return
    
    # Create processing configuration
    config = ProcessingConfig(
        batch_duration=batch_size * 0.4,
        batch_size=batch_size,
        cut_low_frequency=cut_low_freq,
        cut_high_frequency=cut_high_freq,
        image_normalize=image_normalize,
        save_positive_examples=save_positives,
        binary_threshold=binary_threshold
    )
    
    # Get files to process
    if specific_files:
        files_to_process = sorted([f for f in specific_files if (recording_folder / f).exists()])
    else:
        files_to_process = sorted(
            [f for f in os.listdir(recording_folder_path) 
             if f.lower().endswith('.wav') and not (saving_folder / f.split('.')[0] / f"{f.split('.')[0]}.wav_predictions.csv").exists()]
        )
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Process files with thread pool
    with tqdm(total=len(files_to_process), desc="Processing files", position=0, leave=True) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for file_name in files_to_process:
                future = executor.submit(
                    process_single_file, 
                    file_name, 
                    recording_folder, 
                    saving_folder,
                    config,
                    start_time,
                    end_time,
                    model,
                    pbar
                )
                futures.append(future)
            
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)
    
    logger.info(f"Completed processing {len(files_to_process)} files")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio files for whistle detection")
    parser.add_argument("--input", required=True, help="Folder containing audio files")
    parser.add_argument("--output", required=True, help="Folder to save results")
    parser.add_argument("--model", default="models/model_vgg.h5", help="Path to model file")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker threads")
    parser.add_argument("--low-freq", type=int, default=3, help="Low frequency cutoff")
    parser.add_argument("--high-freq", type=int, default=20, help="High frequency cutoff")
    parser.add_argument("--normalize", action="store_true", help="Normalize images")
    parser.add_argument("--save-positives", action="store_true", help="Save positive detections")
    parser.add_argument("--binary-threshold", type=float, default=0.5, 
                        help="Threshold for binary classification (0-1)")
    
    args = parser.parse_args()
    
    process_predict_extract(
        recording_folder_path=args.input,
        saving_folder=args.output,
        cut_low_freq=args.low_freq,
        cut_high_freq=args.high_freq,
        image_normalize=args.normalize,
        batch_size=args.batch_size,
        save_positives=args.save_positives,
        model_path=args.model,
        binary_threshold=args.binary_threshold,
        max_workers=args.workers
    )