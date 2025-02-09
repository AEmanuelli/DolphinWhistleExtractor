# =============================================================================
#********************* IMPORTS
# =============================================================================
import warnings
import sys
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm 
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import concurrent.futures
from utils import *
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ignorer les messages d'information et de débogage de TensorFlow


# =============================================================================
#********************* FUNCTIONS
# =============================================================================

def process_and_predict(file_path, batch_duration, start_time, end_time, batch_size, model, save_p, saving_folder_file, CLF, CHF, image_norm):
    """
    Process an audio file, extract batches of audio data, and predict the presence of a specific class using a given model.

    Args:
        file_path (str): The path to the audio file.
        batch_duration (float): The duration of each batch in seconds.
        start_time (float): The start time in seconds from which to process the audio file.
        end_time (float): The end time in seconds until which to process the audio file. If None, process until the end of the file.
        batch_size (int): The number of audio samples in each batch.
        model (object): The model used for prediction.
        save_p (bool): Whether to save positive predictions as images.
        saving_folder_file (str): The path to the folder where positive prediction images will be saved.

    Returns:
        tuple: A tuple containing the following lists:
            - record_names (list): The names of the audio files.
            - positive_initial (list): The initial times of positive predictions.
            - positive_finish (list): The finish times of positive predictions.
            - class_1_scores (list): The scores of positive predictions.
    """
    file_name = os.path.basename(file_path)
    transformed_file_name = transform_file_name(file_name)
    fs, x = wavfile.read(file_path)
    N = len(x)

    if end_time is not None:
        N = min(N, int(end_time * fs))

    total_duration = (N / fs) - start_time
    record_names = []
    positive_initial = []
    positive_finish = []
    class_1_scores = []
    num_batches = int(np.ceil(total_duration / batch_duration))

    for batch in tqdm(range(num_batches), desc=f"Batches for {transformed_file_name}", leave=False, colour='blue'):
        start = batch * batch_duration + start_time
        images = process_audio_file(file_path, saving_folder_file, batch_size=batch_size, start_time=start, end_time=end_time, cut_low_frequency=CLF, cut_high_frequency=CHF)
        saving_positive = os.path.join(saving_folder_file, "positive")
        
        image_batch = []
        time_batch = []

        for idx, image in enumerate(images):
            im_cop = image
            image_start_time = round(start + idx * 0.4, 2)
            image_end_time = round(image_start_time + 0.4, 2)

            image = cv2.resize(image, (224, 224))
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            if image_norm : 
                image = image/255
            image_batch.append(image)
            time_batch.append((im_cop, image_start_time, image_end_time))

        if image_batch:
            image_batch = np.vstack(image_batch)
            predictions = model.predict(image_batch, verbose=0)
            for idx, prediction in enumerate(predictions):
                im_cop, image_start_time, image_end_time = time_batch[idx]
                if prediction[1] > prediction[0]:
                    record_names.append(file_name)
                    positive_initial.append(image_start_time)
                    positive_finish.append(image_end_time)
                    class_1_scores.append(prediction[1])
                    if save_p:
                        if not os.path.exists(saving_positive):
                            os.makedirs(saving_positive)
                        image_name = os.path.join(saving_positive, f"{image_start_time}-{image_end_time}.jpg")
                        cv2.imwrite(image_name, im_cop)

    return record_names, positive_initial, positive_finish, class_1_scores

def process_predict_extract_worker(file_name, recording_folder_path, saving_folder, start_time, end_time, CLF, CHF, image_norm,
                                batch_size, save_p, model, pbar):
    """
    Process and predict whistle detection for a given audio file.

    Args:
        file_name (str): The name of the audio file.
        recording_folder_path (str): The path to the folder containing the audio file.
        saving_folder (str): The path to the folder where the prediction results will be saved.
        start_time (float): The start time of the segment to process and predict.
        end_time (float): The end time of the segment to process and predict.
        batch_size (int): The batch size for processing the audio file.
        save_p (bool): Flag indicating whether to save the positive predictions.
        model: The trained model for whistle detection.
        pbar: The progress bar object for tracking the processing progress.

    Returns:
        None
    """
    # pbar.set_postfix(file=file_name)
    date_and_channel = os.path.splitext(file_name)[0]
    
    print("Processing:", date_and_channel) 
    saving_folder_file = os.path.join(saving_folder, f"{date_and_channel}")
    os.makedirs(saving_folder_file, exist_ok=True)
    prediction_file_path = os.path.join(saving_folder_file, f"{date_and_channel}.wav_predictions.csv")

    file_path = os.path.join(recording_folder_path, file_name)

    if not file_name.lower().endswith(".wav") or (os.path.exists(prediction_file_path)) :#or ("channel_2" in file_path): #and os.path.exists(saving_positive)):
        print(f"Non-audio or channel 2 or already predicted : {file_name}. Skipping processing.")
        return

    batch_duration = batch_size * 0.4
    record_names, positive_initial, positive_finish, class_1_scores = process_and_predict(file_path, batch_duration, start_time, end_time, batch_size, model, save_p, saving_folder_file, CLF=CLF, CHF=CHF, image_norm=image_norm)
    save_csv(record_names, positive_initial, positive_finish, class_1_scores, prediction_file_path)    
    # process_prediction_file(prediction_file_path, file_name, recording_folder_path)
    pbar.update()

def process_predict_extract(recording_folder_path, saving_folder, CLF = 3, CHF = 20, image_norm = False, start_time=0, end_time=1800, batch_size=50, 
                            save=False, save_p=True, model_path="models/model_vgg.h5", max_workers = 16, specific_files = None):
    """
    Process and extract predictions from audio files in a given folder.

    Args:
        recording_folder_path (str): The path to the folder containing the audio files.
        saving_folder (str): The path to the folder where the predictions will be saved.
        start_time (int, optional): The start time in seconds for processing the audio files. Defaults to 0.
        end_time (int, optional): The end time in seconds for processing the audio files. Defaults to 1800.
        batch_size (int, optional): The batch size for processing the audio files. Defaults to 50.
        save (bool, optional): Flag indicating whether to save the predictions. Defaults to False.
        save_p (bool, optional): Flag indicating whether to save the processed audio files. Defaults to True.
        model_path (str, optional): The path to the trained model file. Defaults to "models/model_vgg.h5".
        max_workers (int, optional): The maximum number of worker threads to use for processing. Defaults to 16.
        specific_files (list, optional): A list of specific file names to process. Defaults to None.

    Returns:
        None
    """
    files = os.listdir(recording_folder_path)
    sorted_files = sorted(files, key=lambda x: os.path.getctime(os.path.join(recording_folder_path, x)), reverse=False)
    
    if specific_files:
        sorted_files = sorted(specific_files, key=lambda x: os.path.getctime(os.path.join(recording_folder_path, x)), reverse=False)


    mask_count = 0  # Compteur pour les fichiers filtrés par le masque
    model = tf.keras.models.load_model(model_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        with tqdm(total=len(files), desc="Files that are not going to be processed right now : ", position=0, leave=True, colour='green') as pbar:
            for file_name in sorted_files:
                file_path = os.path.join(recording_folder_path, file_name)
                prediction_file_path = os.path.join(saving_folder, f"{file_name}_predictions.csv")
                mask = (os.path.isdir(file_path) or 
                            not file_name.lower().endswith('.wav') or 
                            os.path.exists(prediction_file_path))
                    
                if mask:
                    mask_count += 1
                    pbar.update(1)  # Incrémenter la barre de progression pour les fichiers filtrés
                    continue
                
                future = executor.submit(process_predict_extract_worker, file_name, recording_folder_path, 
                                         saving_folder, start_time, end_time, CLF, CHF, image_norm, batch_size, save_p, model, pbar)
                future.add_done_callback(lambda _: pbar.update(1))  # Mettre à jour la barre de progression lorsque le thread termine
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                future.result()
