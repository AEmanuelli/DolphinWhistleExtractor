"""Whistle detection: turn recordings into spectrogram windows and classify them.

For every recording found in a folder, the audio is cut into overlapping
0.8 s windows (one every 0.4 s), each window is converted to a spectrogram
image and classified by a Keras model. Detections are written to
``<saving_folder>/<recording>/<recording>.wav_predictions.csv`` and, optionally,
the positive spectrograms are saved next to it in a ``positive`` sub-folder.

TensorFlow is imported lazily (when a model is loaded) so that this module can
be imported cheaply, e.g. by the GUI before the dependencies are installed.
"""
import concurrent.futures
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:  # works both as a script directory import and as a package import
    from utils import AUDIO_EXTENSIONS, WINDOW_HOP, audio_to_spectrogram_images, load_audio, save_csv
except ImportError:  # pragma: no cover
    from .utils import AUDIO_EXTENSIONS, WINDOW_HOP, audio_to_spectrogram_images, load_audio, save_csv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = str(REPO_ROOT / "models" / "base_MobileNetV2.keras")
PREDICTION_COLUMNS = ("file_name", "initial_point", "finish_point", "confidence")


@dataclass
class ProcessingConfig:
    """Configuration for audio processing and prediction."""
    batch_duration: float
    batch_size: int
    cut_low_frequency: int
    cut_high_frequency: int
    image_normalize: bool
    save_positive_examples: bool
    binary_threshold: float = 0.5  # Threshold on the "whistle" score
    image_size: Tuple[int, int] = (224, 224)  # (width, height) expected by the model
    save_all_images: bool = False  # Also save every spectrogram, not only the positives


class WhistleDetector:
    """Thin wrapper around a Keras classifier used for inference only."""

    def __init__(self, model_path: str):
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # must be set before importing TensorFlow
        import tensorflow as tf

        for device in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except (RuntimeError, ValueError) as e:  # already initialised
                logger.debug("GPU memory configuration skipped: %s", e)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading model from %s", model_path)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.trainable = False

        input_shape = self.model.input_shape
        height, width = input_shape[1], input_shape[2]
        self.image_size = (width or 224, height or 224)  # (width, height), OpenCV convention

        # Fixed signature: one traced graph whatever the batch size, safe to call from threads.
        spec = tf.TensorSpec([None, self.image_size[1], self.image_size[0], input_shape[3] or 3], tf.float32)
        self._infer = tf.function(lambda images: self.model(images, training=False), input_signature=[spec])
        # Warm up (builds the graph once, before worker threads start)
        self.predict(np.zeros((1, self.image_size[1], self.image_size[0], input_shape[3] or 3), dtype=np.float32))

    def predict(self, image_batch: np.ndarray) -> np.ndarray:
        """Return the raw model output for a float32 batch of images."""
        return np.asarray(self._infer(image_batch))


def prepare_image_batch(images: List[np.ndarray], start_time: float,
                        config: ProcessingConfig) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float, float]]]:
    """
    Resize spectrograms to the model input size and attach their time stamps.

    Args:
        images: List of spectrogram images (consecutive windows, ``WINDOW_HOP`` apart)
        start_time: Start time (s) of the first image
        config: Processing configuration

    Returns:
        (batch, time_info): float32 array of shape (n, height, width, 3) ready for
        prediction, and a list of (original image, start time, end time).
    """
    if not images:
        return np.array([]), []

    starts = np.round(start_time + np.arange(len(images)) * WINDOW_HOP, 2)
    ends = np.round(starts + WINDOW_HOP, 2)
    time_info = [(image, float(start), float(end)) for image, start, end in zip(images, starts, ends)]

    batch = np.array([cv2.resize(img, config.image_size) for img in images])
    if batch.ndim == 3:  # grayscale images: add the channel dimension
        batch = np.repeat(batch[..., np.newaxis], 3, axis=3)

    batch = batch.astype(np.float32)
    if config.image_normalize:
        batch /= 255.0
    return batch, time_info


def positive_scores(predictions: np.ndarray, binary_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpret raw model outputs.

    Supports single-output (sigmoid) models, two-class models (class 1 = whistle)
    and multi-class models (positive when class 1 is the arg-max).

    Returns:
        (is_positive, scores): boolean mask and "whistle" score for every image.
    """
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]

    if predictions.shape[1] == 1:
        scores = predictions[:, 0]
        return scores >= binary_threshold, scores
    scores = predictions[:, 1]
    if predictions.shape[1] == 2:
        return scores >= binary_threshold, scores
    return np.argmax(predictions, axis=1) == 1, scores


def save_positive_prediction(image: np.ndarray, start_time: float, end_time: float,
                             saving_folder: Path) -> None:
    """Save a positive prediction image to disk."""
    saving_positive = Path(saving_folder) / "positive"
    saving_positive.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(saving_positive / f"{start_time}-{end_time}.jpg"), image)


def process_and_predict(file_path: str, config: ProcessingConfig,
                        start_time: float, end_time: Optional[float],
                        detector, saving_folder: Path
                        ) -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    Run the detector over one recording.

    The audio file is loaded once, then processed batch by batch so that only
    ``config.batch_size`` spectrograms are held in memory at a time.

    Args:
        detector: object exposing ``predict(batch) -> np.ndarray`` (see WhistleDetector).

    Returns:
        (record_names, initial_points, finish_points, confidences) of the positive windows.
    """
    started = time.time()
    file_name = Path(file_path).name
    fs, x = load_audio(file_path)

    n_samples = len(x)
    if end_time is not None:
        n_samples = min(n_samples, int(end_time * fs))
    total_duration = max(0.0, n_samples / fs - start_time)
    num_batches = int(np.ceil(total_duration / config.batch_duration))

    record_names: List[str] = []
    positive_initial: List[float] = []
    positive_finish: List[float] = []
    class_1_scores: List[float] = []

    for batch_index in range(num_batches):
        batch_start = batch_index * config.batch_duration + start_time
        images = audio_to_spectrogram_images(
            x, fs, file_name=Path(file_path).stem, saving_folder=str(saving_folder),
            batch_size=config.batch_size, start_time=batch_start, end_time=end_time,
            save=config.save_all_images,
            cut_low_frequency=config.cut_low_frequency,
            cut_high_frequency=config.cut_high_frequency,
        )
        if not images:
            continue

        image_batch, time_batch = prepare_image_batch(images, batch_start, config)
        is_positive, scores = positive_scores(detector.predict(image_batch), config.binary_threshold)

        for idx in np.flatnonzero(is_positive):
            orig_image, image_start_time, image_end_time = time_batch[idx]
            record_names.append(file_name)
            positive_initial.append(image_start_time)
            positive_finish.append(image_end_time)
            class_1_scores.append(float(scores[idx]))
            if config.save_positive_examples:
                save_positive_prediction(orig_image, image_start_time, image_end_time, saving_folder)

    logger.info("Processed %s in %.2fs with %d detections", file_name, time.time() - started, len(record_names))
    return record_names, positive_initial, positive_finish, class_1_scores


def prediction_csv_path(saving_folder: Path, file_name: str) -> Path:
    """Path of the predictions CSV written for a given recording."""
    stem = Path(file_name).stem
    return Path(saving_folder) / stem / f"{stem}.wav_predictions.csv"


def process_single_file(file_name: str, recording_folder: Path, saving_folder: Path,
                        config: ProcessingConfig, start_time: float, end_time: Optional[float],
                        detector) -> Optional[int]:
    """
    Detect whistles in one recording and write its predictions CSV.

    Returns:
        Number of detections, or None if the file was skipped (already processed).

    The CSV is only written when processing succeeded, so a file that failed is
    retried on the next run instead of being recorded as "no whistle".
    """
    prediction_path = prediction_csv_path(saving_folder, file_name)
    if prediction_path.exists():
        logger.info("Skipping %s: already processed", file_name)
        return None

    file_saving_folder = prediction_path.parent
    file_saving_folder.mkdir(exist_ok=True, parents=True)

    record_names, positive_initial, positive_finish, class_1_scores = process_and_predict(
        str(Path(recording_folder) / file_name), config, start_time, end_time, detector, file_saving_folder
    )

    # Written even when empty (header only): marks the recording as processed.
    save_csv(record_names, positive_initial, positive_finish, class_1_scores, str(prediction_path))
    return len(record_names)


def get_optimal_worker_count(file_count: int) -> int:
    """Number of worker threads: one per file, capped by CPU count (at most 8)."""
    return max(1, min(file_count, os.cpu_count() or 1, 8))


def list_files_to_process(recording_folder: Path, saving_folder: Path,
                          specific_files: Optional[List[str]] = None) -> List[str]:
    """Audio files of ``recording_folder`` that do not have a predictions CSV yet."""
    if specific_files:
        candidates = [f.strip() for f in specific_files if f.strip()]
        missing = [f for f in candidates if not (recording_folder / f).is_file()]
        for f in missing:
            logger.warning("Listed file not found in %s: %s", recording_folder, f)
        candidates = [f for f in candidates if f not in missing]
    else:
        candidates = [f for f in os.listdir(recording_folder) if (recording_folder / f).is_file()]

    return sorted(
        f for f in candidates
        if f.lower().endswith(AUDIO_EXTENSIONS) and not prediction_csv_path(saving_folder, f).exists()
    )


def process_predict_extract(recording_folder_path: str, saving_folder: str,
                            cut_low_freq: int = 3, cut_high_freq: int = 20,
                            image_normalize: bool = False, start_time: float = 0,
                            end_time: Optional[float] = 1800, batch_size: int = 50,
                            save: bool = False, save_positives: bool = True,
                            model_path: str = DEFAULT_MODEL_PATH,
                            binary_threshold: float = 0.5,
                            max_workers: Optional[int] = None,
                            specific_files: Optional[List[str]] = None,
                            progress_callback: Optional[Callable[[float, str], None]] = None,
                            ) -> Dict[str, Optional[int]]:
    """
    Detect whistles in every recording (.wav / .flac) of a folder.

    Args:
        recording_folder_path: Folder containing the recordings.
        saving_folder: Folder where one sub-folder per recording is created.
        cut_low_freq / cut_high_freq: Frequency band (kHz) shown in the spectrograms.
        image_normalize: Divide images by 255 before prediction (depends on how the model was trained).
        start_time / end_time: Portion of each recording to analyse, in seconds (end_time=None: whole file).
        batch_size: Number of spectrograms per inference batch.
        save: Save every spectrogram image.
        save_positives: Save the spectrogram images classified as whistles.
        model_path: Keras model file.
        binary_threshold: Minimum score for a window to count as a whistle.
        max_workers: Number of recordings processed in parallel (None or <= 0: automatic).
        specific_files: Only process these file names (relative to the recordings folder).
        progress_callback: Called as ``progress_callback(percent, status_text)`` after each file.

    Returns:
        Dict mapping each processed file name to its number of detections
        (None when the file failed; it will be retried on the next run).

    Raises:
        FileNotFoundError: If the recordings folder or the model does not exist.
    """
    recording_folder = Path(recording_folder_path)
    saving_folder = Path(saving_folder)
    if not recording_folder.is_dir():
        raise FileNotFoundError(f"Recordings folder not found: {recording_folder}")
    saving_folder.mkdir(exist_ok=True, parents=True)

    started = time.time()
    files_to_process = list_files_to_process(recording_folder, saving_folder, specific_files)
    if not files_to_process:
        logger.info("No files to process.")
        return {}
    logger.info("Found %d files to process", len(files_to_process))

    if batch_size is None or batch_size <= 0:
        batch_size = 50
    if max_workers is None or max_workers <= 0:
        max_workers = get_optimal_worker_count(len(files_to_process))
        logger.info("Automatically selected worker count: %d", max_workers)

    detector = WhistleDetector(model_path)

    config = ProcessingConfig(
        batch_duration=batch_size * WINDOW_HOP,
        batch_size=batch_size,
        cut_low_frequency=cut_low_freq,
        cut_high_frequency=cut_high_freq,
        image_normalize=image_normalize,
        save_positive_examples=save_positives,
        binary_threshold=binary_threshold,
        image_size=detector.image_size,
        save_all_images=save,
    )

    results: Dict[str, Optional[int]] = {}
    lock = threading.Lock()

    with tqdm(total=len(files_to_process), desc="Processing files") as pbar, \
            concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_file, file_name, recording_folder, saving_folder,
                            config, start_time, end_time, detector): file_name
            for file_name in files_to_process
        }
        for future in concurrent.futures.as_completed(futures):
            file_name = futures[future]
            try:
                results[file_name] = future.result()
                status = f"{file_name}: {results[file_name] or 0} detections"
            except Exception as e:  # keep going with the other recordings
                results[file_name] = None
                status = f"{file_name}: failed ({e})"
                logger.error("Error processing %s: %s", file_name, e)
            with lock:
                pbar.update(1)
                if progress_callback is not None:
                    progress_callback(100.0 * pbar.n / len(files_to_process), status)

    failed = [name for name, count in results.items() if count is None]
    logger.info("Completed processing %d files in %.2fs (%d failed)",
                len(files_to_process), time.time() - started, len(failed))
    return results
