import warnings
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import cv2
import concurrent.futures
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import time

# TensorFlow optimizations
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Configure TensorFlow to use GPU memory growth
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Using GPU: {physical_devices[0].name}")
    except Exception as e:
        print(f"GPU memory configuration error: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages


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
    image_size: Tuple[int, int] = (224, 224)  # Standard image size for most models


# Create optimized model prediction function
@tf.function(reduce_retracing=True)
def predict_optimized(model, images):
    """Optimized model prediction function using TensorFlow's graph execution."""
    return model(images, training=False)



def prepare_image_batch(images: List[np.ndarray], start_time: float, 
                                  config: ProcessingConfig) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float, float]]]:
    """
    Vectorized implementation of image batch preparation for massive performance improvement.
    
    Args:
        images: List of spectrograms
        start_time: Start time of the first image
        config: Processing configuration
        
    Returns:
        Tuple of processed images ready for prediction and time information
    """
    if not images:
        return np.array([]), []
    
    # Pre-allocate arrays with exact shape needed
    batch_size = len(images)
    
    # Stack images first for vectorized operations
    # Use numpy's stack to avoid Python loop
    original_images = [img.copy() for img in images]
    
    # Calculate all start/end times at once using vectorized operations
    image_start_times = start_time + np.arange(batch_size) * 0.4
    image_end_times = image_start_times + 0.4
    
    # Round for consistent output
    image_start_times = np.round(image_start_times, 2)
    image_end_times = np.round(image_end_times, 2)
    
    # Create time info list
    time_info = [(original_images[i], image_start_times[i], image_end_times[i]) 
                 for i in range(batch_size)]
    
    # Process all images at once when possible
    # Convert to numpy array for vectorized operations
    processed_images = []
    
    # Use OpenCV's batch processing if available
    if hasattr(cv2, 'resize_batch'):
        # OpenCV 4.7+ has batch processing
        processed_images = cv2.resize_batch(images, config.image_size)
    else:
        # Fall back to list comprehension which is still faster than for loop
        processed_images = np.array([cv2.resize(img, config.image_size) for img in images])
    
    # Add channel dimension if needed - vectorized operation
    if processed_images.shape[-1] != 3:
        if len(processed_images.shape) == 3:  # [batch, height, width]
            processed_images = np.repeat(processed_images[:, :, :, np.newaxis], 3, axis=3)
        else:  # Handle single image case
            processed_images = np.repeat(processed_images[:, :, np.newaxis], 3, axis=2)
    
    # Normalize in a single vectorized operation if required
    if config.image_normalize:
        processed_images = processed_images / 255.0
    
    # Ensure correct dtype for TensorFlow
    return processed_images.astype(np.float32), time_info




    
def save_positive_prediction(image: np.ndarray, start_time: float, end_time: float, 
                             saving_folder: Path) -> None:
    """Save a positive prediction image to disk."""
    saving_positive = saving_folder / "positive"
    saving_positive.mkdir(exist_ok=True)
    
    image_name = saving_positive / f"{start_time}-{end_time}.jpg"
    cv2.imwrite(str(image_name), image)


def is_positive_prediction(prediction: np.ndarray, binary_threshold: float) -> Tuple[bool, float]:
    """
    Determine if a prediction is positive and return confidence score.
    Uses vectorized operations for speed.
    """
    # Simplified logic with fewer conditional branches
    if prediction.size == 1:
        # Binary classifier with single output
        score = float(prediction.item())
        return score >= binary_threshold, score
    elif prediction.size == 2:
        # Categorical classifier with two classes
        score = float(prediction[1])
        return score > 0.5, score
    else:
        # Multi-class classifier
        positive_class_idx = 1  # Assuming class 1 is positive class
        score = float(prediction[positive_class_idx]) 
        return np.argmax(prediction) == positive_class_idx, score


def process_and_predict(file_path: str, config: ProcessingConfig, 
                                start_time: float, end_time: Optional[float], 
                                model: tf.keras.Model, saving_folder: Path,
                                model_type: str = 'unknown') -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    Process an audio file with batched prediction and memory-efficient processing.
    
    Optimizations:
    - Uses memory mapping for audio files
    - Processes audio in streaming chunks
    - Uses TensorFlow's dataset API for optimized data pipeline
    - Implements vectorized prediction processing
    """
    try:
        start_time_proc = time.time()
        file_name = Path(file_path).name
        
        # Import utility functions - moved outside loop for better performance
        from utils import process_audio_file
        
        # Memory-mapped file reading for large audio files
        with open(file_path, 'rb') as fid:
            import io
            import mmap
            # Memory map the file for faster access
            mm = mmap.mmap(fid.fileno(), 0, access=mmap.ACCESS_READ)
            fid_mm = io.BytesIO(mm)
            
            # Use scipy to read the memory-mapped file
            fs, x = wavfile.read(fid_mm)
        
        # Calculate processing parameters more efficiently
        N = len(x)
        if end_time is not None:
            N = min(N, int(end_time * fs))
            
        total_duration = (N / fs) - start_time
        num_batches = int(np.ceil(total_duration / config.batch_duration))
        
        # Pre-allocate results with reasonable initial capacity
        estimated_positives = max(10, num_batches * config.batch_size // 10)
        record_names = []
        positive_initial = []
        positive_finish = []
        class_1_scores = []
        
        # Create TensorFlow dataset for batch processing
        def generate_batches():
            for batch in range(num_batches):
                batch_start = batch * config.batch_duration + start_time
                yield batch_start
        
        # Create and optimize TensorFlow dataset
        batch_dataset = tf.data.Dataset.from_generator(
            generate_batches,
            output_types=tf.float32,
            output_shapes=()
        ).prefetch(tf.data.AUTOTUNE)
        
        # Process batches with optimized pipeline
        for batch_start in tqdm(batch_dataset, total=num_batches, 
                              desc=f"Processing {file_name}", leave=False, colour='blue'):
            batch_start = float(batch_start.numpy())
            
            # Process audio batch and get images
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
            
            # Use our optimized vectorized image preparation
            image_batch, time_batch = prepare_image_batch(images, batch_start, config)
            
            if len(image_batch) == 0:
                continue
            
            # Use TensorFlow's efficient batch prediction
            # Convert to TensorFlow tensor first for better GPU utilization
            tf_batch = tf.convert_to_tensor(image_batch, dtype=tf.float32)
            predictions = predict_optimized(model, tf_batch).numpy()
            
            # Vectorized prediction processing
            # Process predictions in bulk where possible
            if predictions.shape[1] == 1:  # Binary output
                # Reshape to 1D array for easier comparison
                scores = predictions.reshape(-1)
                # Find positive indices in one operation
                positive_indices = np.where(scores >= config.binary_threshold)[0]
                confidences = scores[positive_indices]
            elif predictions.shape[1] == 2:  # Two-class output
                # Class 1 probabilities
                scores = predictions[:, 1]
                # Find positive indices in one operation
                positive_indices = np.where(scores >= config.binary_threshold)[0]
                confidences = scores[positive_indices]
            else:  # Multi-class
                # Find max class per prediction
                max_classes = np.argmax(predictions, axis=1)
                # Find where class 1 is the max
                positive_indices = np.where(max_classes == 1)[0]
                confidences = predictions[positive_indices, 1]
            
            # Bulk process the positive predictions
            for idx in positive_indices:
                orig_image, image_start_time, image_end_time = time_batch[idx]
                
                record_names.append(file_name)
                positive_initial.append(image_start_time)
                positive_finish.append(image_end_time)
                class_1_scores.append(float(confidences[np.where(positive_indices == idx)[0][0]]))
                
                # Save positive example if requested
                if config.save_positive_examples:
                    save_positive_prediction(
                        orig_image, image_start_time, image_end_time, saving_folder
                    )
            
            # Explicitly clean up memory
            del images, image_batch, predictions, tf_batch
            
        elapsed = time.time() - start_time_proc
        logger.info(f"Processed {file_name} in {elapsed:.2f}s with {len(record_names)} detections")
        return record_names, positive_initial, positive_finish, class_1_scores
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return [], [], [], []

def process_single_file(file_name: str, recording_folder: Path, saving_folder: Path, 
                        config: ProcessingConfig, start_time: float, end_time: Optional[float], 
                        model: tf.keras.Model, model_type: str, pbar: tqdm) -> None:
    """Process a single audio file for whistle detection."""
    try:
        # Prepare file paths
        file_stem = Path(file_name).stem
        file_path = recording_folder / file_name
        file_saving_folder = saving_folder / file_stem
        prediction_path = file_saving_folder / f"{file_stem}.wav_predictions.csv"
        
        # Skip processing if already done or not a WAV file
        if not file_name.lower().endswith(".wav"):
            pbar.update(1)
            return
            
        if prediction_path.exists():
            logger.info(f"Skipping {file_name}: Already processed")
            pbar.update(1)
            return
        
        # Create output directory - do this once per file
        file_saving_folder.mkdir(exist_ok=True, parents=True)
            
        logger.info(f"Processing: {file_stem}")
        
        # Process audio and get predictions
        record_names, positive_initial, positive_finish, class_1_scores = process_and_predict(
            str(file_path), config, start_time, end_time, model, file_saving_folder, model_type
        )
        
        # Import here to avoid circular imports
        from utils import save_csv
        
        # Save results if any positive detections found
        if record_names:
            save_csv(record_names, positive_initial, positive_finish, class_1_scores, str(prediction_path))
            logger.info(f"Saved {len(record_names)} detections for {file_name}")
        else:
            # Create empty file to mark as processed
            with open(prediction_path, 'w') as f:
                f.write("record_name,initial,finish,confidence\n")
            logger.info(f"No detections found in {file_name}")
            
    except Exception as e:
        logger.error(f"Error processing {file_name}: {str(e)}")
    finally:
        pbar.update(1)


def detect_model_output_type(model: tf.keras.Model) -> str:
    """Detect the output type of the model using a unified approach."""
    try:
        # Use a single dummy input for all detection approaches
        dummy_input = np.zeros((1,) + model.input_shape[1:], dtype=np.float32)
        
        # Run prediction once
        output = model.predict(dummy_input, verbose=0)
        
        if output.ndim == 1 or (output.ndim == 2 and output.shape[1] == 1):
            return 'binary'
        elif output.ndim == 2 and output.shape[1] > 1:
            return 'categorical'
        
        # As a fallback, check the last layer
        last_layer = model.layers[-1]
        if hasattr(last_layer, 'activation'):
            activation_name = last_layer.activation.__name__
            if activation_name == 'sigmoid':
                return 'binary'
            elif activation_name == 'softmax':
                return 'categorical'
        
        return 'unknown'
        
    except Exception as e:
        logger.warning(f"Could not detect model type: {str(e)}")
        return 'unknown'


def get_optimal_batch_size(file_count: int) -> int:
    """Determine optimal batch size based on available system resources."""
    import psutil
    
    # Get available memory in GB
    available_memory = psutil.virtual_memory().available / (1024**3)
    
    # Get CPU and GPU info
    cpu_count = os.cpu_count() or 8
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    
    # Base batch size on available memory and compute resources
    if gpu_available:
        # GPU-based batch sizing
        if available_memory > 16:
            return 64
        elif available_memory > 8:
            return 32
        else:
            return 16
    else:
        # CPU-based batch sizing
        if available_memory > 32:
            return 128
        elif available_memory > 16:
            return 64
        elif available_memory > 8:
            return 32
        else:
            return 16


def get_optimal_worker_count(file_count: int) -> int:
    """Determine optimal worker count based on system resources and workload."""
    import psutil
    
    # Get system information
    cpu_count = os.cpu_count() or 8
    available_memory = psutil.virtual_memory().available / (1024**3)
    
    # Calculate based on workload and resources
    if file_count < 5:
        # For small workloads, use fewer workers
        return max(1, min(file_count, cpu_count // 2))
    elif available_memory < 4:
        # Limited memory systems
        return max(1, cpu_count // 4)
    elif available_memory < 8:
        # Medium memory systems
        return max(1, cpu_count // 2)
    else:
        # High memory systems
        return max(1, int(cpu_count * 0.75))

def process_predict_extract(recording_folder_path: str, saving_folder: str, 
                            cut_low_freq: int = 3, cut_high_freq: int = 20, 
                            image_normalize: bool = False, start_time: float = 0, 
                            end_time: Optional[float] = 1800, batch_size: int = 50,
                            save: bool = False, save_positives: bool = True, 
                            model_path: str = "models/model_vgg.h5", 
                            binary_threshold: float = 0.5,
                            max_workers: Optional[int] = None, 
                            specific_files: Optional[List[str]] = None) -> None:
    """
    Process and extract predictions from multiple audio files with optimized performance.
    """
    # Convert paths to Path objects
    recording_folder = Path(recording_folder_path)
    saving_folder = Path(saving_folder)
    saving_folder.mkdir(exist_ok=True, parents=True)
    
    # Start timer
    start_time_total = time.time()
    
    # Determine files to process
    if specific_files:
        files_to_process = sorted([f for f in specific_files if (recording_folder / f).exists() and f.lower().endswith('.wav')])
    else:
        files_to_process = sorted(
            [f for f in os.listdir(recording_folder_path) 
             if f.lower().endswith('.wav') and not (saving_folder / f.split('.')[0] / f"{f.split('.')[0]}.wav_predictions.csv").exists()]
        )
    
    # Skip processing if no files found
    if not files_to_process:
        logger.info("No files to process.")
        return
        
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Optimize batch size based on available resources if not specified
    if batch_size <= 0:
        batch_size = get_optimal_batch_size(len(files_to_process))
        logger.info(f"Automatically selected batch size: {batch_size}")
    
    # Optimize worker count based on system resources if not specified
    if max_workers is None or max_workers <= 0:
        max_workers = get_optimal_worker_count(len(files_to_process))
        logger.info(f"Automatically selected worker count: {max_workers}")
    
    # Load and optimize model - do this ONCE before processing
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Load model with optimizations
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Set model to inference mode
        model.trainable = False
        
        # Detect model type
        model_type = detect_model_output_type(model)
        logger.info(f"Detected model type: {model_type}")
        
        # Warm up the model with a dummy prediction
        dummy_input = np.zeros((1,) + model.input_shape[1:], dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
        
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
        binary_threshold=binary_threshold,
        image_size=(224, 224)  # Standard size for most models
    )
    
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
                    model_type,
                    pbar
                )
                futures.append(future)
            
            # Wait for all tasks to complete (or handle exceptions)
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Get result or exception
                except Exception as e:
                    logger.error(f"Thread error: {str(e)}")
    
    elapsed = time.time() - start_time_total
    logger.info(f"Completed processing {len(files_to_process)} files in {elapsed:.2f}s")

import cProfile
import pstats
import io
# from predict_and_extract_online import process_predict_extract

# Define test parameters - update these paths

# # Profile function
# def profile_worker_function():
#     recording_folder_path = "/home/emanuelli/Téléchargements"
#     saving_folder = "/home/emanuelli/Téléchargements/profilejihugyf="
#     model_path = "models/base_MobileNetV2.keras"

#     # Load model once outside profiling
#     model = tf.keras.models.load_model(model_path, compile=False)
#     model.trainable = False
#     model_type = detect_model_output_type(model)
    
#     # Create configuration
#     config = ProcessingConfig(
#         batch_duration=50 * 0.4,
#         batch_size=50,
#         cut_low_frequency=3,
#         cut_high_frequency=20,
#         image_normalize=False,
#         save_positive_examples=True,
#         binary_threshold=0.5,
#         image_size=(224, 224)
#     )
    
#     # Get a single file
#     recording_folder = Path(recording_folder_path)
#     saving_folder = Path(saving_folder)
#     saving_folder.mkdir(exist_ok=True, parents=True)
    
#     # Find first WAV file
#     for i, file_name in enumerate(os.listdir(recording_folder_path)):
#         if file_name.lower().endswith('.wav'):
#             if i == 1:
#                 # Create dummy progress bar
#                 dummy_pbar = tqdm(total=1, disable=False)
                
#                 # Profile just the worker function
#                 process_single_file(
#                     file_name=file_name,
#                     recording_folder=recording_folder,
#                     saving_folder=saving_folder,
#                     config=config,
#                     start_time=0,
#                     end_time=None,  # Profile first 5 minutes only
#                     model=model,
#                     model_type=model_type,
#                     pbar=dummy_pbar
#                 )
#                 break

# # Run profiler
# profiler = cProfile.Profile()
# profiler.enable()
# profile_worker_function()
# profiler.disable()

# # Save and print results
# profiler.dump_stats('processing_profile.prof')
# stats = pstats.Stats(profiler).sort_stats('cumulative')
# stats.print_stats(20)  # Top 20 time-consuming functions

# print("\nProfile data saved to 'processing_profile.prof'")
# print("To visualize: pip install snakeviz && snakeviz processing_profile.prof")
