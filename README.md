# DNN whistle detection

This project is designed for the extraction, prediction, and analysis of audio signals, particularly focusing on whistle sounds. The project structure is organized into several modules that handle different aspects of the workflow, from data preparation and model training to prediction and comparison.

## Folder Structure

```
.
├── __init__.py
├── models/
├── Models_vs_years_comparison
│   ├── figs/
│   └── models_vs_years_predictions.ipynb
├── Predict_and_extract
│   ├── Extraction_with_kaggle
│   │   ├── classify_again.ipynb
│   │   ├── utility_script (extraction_using_kaggle).py
│   │   └── Whistle_pipeline_(extraction_with_kaggle).ipynb
│   ├── __init__.py
│   ├── main.py
│   ├── maintenance.py
│   ├── pipeline.ipynb
│   ├── predict_and_extract_online.py
│   ├── predict_online.py
│   ├── process_predictions.py
│   ├── Show_newly_detected_stuff.py
│   ├── utils.py
│   └── vidéoaudio.py
├── README.md
├── requirements.txt
└── Train_model
    ├── Create_dataset
    │   ├── AllWhistlesSubClustering_final.csv
    │   ├── create_batch_data.py
    │   ├── DNN précision - CSV_EVAL.csv
    │   ├── dolphin_signal_train.csv
    │   ├── save_spectrogram.m
    │   └── timings
    │       ├── negatives.csv
    │       └── positives.csv
    ├── csv2dataset.py
    ├── dataset2csv.py
    ├── fine-tune2.ipynb
    ├── fine-tune.ipynb
    ├── __init__.py
    └── Train.py
```

## Contents

### Root Directory
- **`__init__.py`**: Initialization file for the package.
- **`README.md`**: This file, providing an overview and details about the project structure and usage.
- **`requirements.txt`**: Lists all the dependencies required to run the code in this folder.

### Models Directory
- **`models/`**: Directory intended for storing trained machine learning models.

### Models_vs_years_comparison 
- **`figs/`**: Directory for storing figures and plots generated during model comparison.
- **`models_vs_years_predictions.ipynb`**: Jupyter notebook for comparing different model predictions across different years.

### Predict_and_extract
- **`Extraction_with_kaggle`**: Directory for whistke extraction using kaggle (to accelerate global extraction)
  - **`create_batch_data.py`**: This script recursively scans all files in the specified source directory. It groups the files into batches where the total size does not exceed 80 GB and creates ZIP archives for each batch. The relative directory structure is preserved within each ZIP file, and the archives are stored in the specified destination directory.
  - **`classify_again.ipynb`**: Jupyter notebook for classifying images again using a model.
  - **`utility_script (extraction_using_kaggle).py`**: Python script containing utility functions for extraction.
  - **`Whistle_pipeline_(extraction_with_kaggle).ipynb`**: Jupyter notebook for the whistle extraction pipeline using Kaggle.
- **`__init__.py`**: Initialization file for the `Predict_and_extract` module.
- **`main.py`**: Main script to run the prediction and extraction pipeline.
- **`maintenance.py`**: Script for maintaining and updating the dataset and models.
- **`pipeline.ipynb`**: Jupyter notebook doing what main.py does but without GPU .
- **`predict_and_extract_online.py`**: 
    - Defines the `process_and_predict` function, which takes a file path and performs audio processing and prediction on it. It extracts audio features, preprocesses them, and uses a pre-trained model to make predictions. It also saves positive predictions as images if specified.
    - Defines the `process_predict_extract_worker` function, which is called by the `process_predict_extract` function. It processes a single file by calling the `process_and_predict` function and saves the predictions in a CSV file.
    - Defines the `process_predict_extract` function, which is the main function of the script. It takes a folder path containing audio files, iterates over the files, and processes them in parallel using multiple threads. It calls the `process_predict_extract_worker` function for each file.
- **`predict_online.py`**: I don't remember exactly what this does.
- **`process_predictions.py`**: Script for processing and analyzing the predictions. ie process the csv file previously produced. This can be done for video files, video and audio (audio = True) or only audio for Wav2vec stuff, (audio_only = True)
- **`Show_newly_detected_stuff.py`**: Script that contains the entire pipeline, but that is used to compare the outputs of new models with respect to what has previously been extracted (it saves the images that were not previously labelled as positive in a folder that has the name of th model, inside Analyses_Alexis/Newly_extracted_whistles)
- **`utils.py`**: Utility functions used across the `Predict_and_extract` module.
- **`vidéoaudio.py`**: Script for extracting video segments and also their associated audio.

### Train_model
- **`Create_dataset`**:
  - **`AllWhistlesSubClustering_final.csv`**: CSV file used for the dataset creation
  - **`DNN précision - CSV_EVAL.csv`**: CSV file containing evaluation metrics of the DNN model.
  - **`dolphin_signal_train.csv`**: CSV file containing training data for dolphin signals.
  - **`save_spectrogram.m`**: MATLAB script for saving spectrograms.
  - **`timings`**:
    - **`negatives.csv`**: CSV file containing negative samples.
    - **`positives.csv`**: CSV file containing positive samples.
- **`csv2dataset.py`**: Script to extract dataset from CSV data.
- **`dataset2csv.py`**: Script to convert datasets back to CSV format.
- **`fine-tune2.ipynb`**: Jupyter notebook for fine-tuning the model (version 2).
- **`fine-tune.ipynb`**: Jupyter notebook for fine-tuning the model. [https://www.kaggle.com/alexisemanuelli/fine-tune]
- **`__init__.py`**: Initialization file for the `Train_model` module.
- **`Train.py`**: Script to train the machine learning models.

also cf [https://www.kaggle.com/code/alexisemanuelli/whistles-detection-transfer-learning-0-95]

## Getting Started

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/AEmanuelli/Dolphins.git

   cd Dolphins/DNN_whistle_detection/
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### NOTES 


The main code you want to look at is:

- `main.py`: This file is used to run the entire pipeline.

- `predict_and_extract_online.py`: This file handles the extraction using a neural network and produces a CSV file.

- `process_predictions.py`: This file takes the previously generated CSV and extracts the desired segments of audiovisual modality.

#### Summary of `predict_and_extract_online.py` script:

- `process_and_predict` function: Takes in the path to an audio file, batch duration, start and end times, batch size, model, save flag, and saving folder file path. It processes the audio file, extracts batches of audio data, and predicts the presence of the specific class using the model. It returns a tuple containing lists of record names, positive initial times, positive finish times, and class 1 scores.
- `process_predict_extract_worker` function: Worker function that processes and predicts whistle detection for a given audio file. It takes in the file name, recording folder path, saving folder path, start and end times, batch size, save flag, model, and a progress bar object. It creates a saving folder for the positive predictions, processes the audio file, and saves the prediction results in a CSV file.
- `process_predict_extract` function: Main function that processes and extracts predictions from audio files in a given folder. It takes in the recording folder path, saving folder path, start and end times, batch size, save flag, save_p flag, model path, maximum number of worker threads, and a list of specific file names to process. It loads the model, sorts the files in the recording folder, and uses a thread pool executor to process the audio files in parallel.

#### Summary of `process_predictions.py` script:

- `audioextraits` function: Extracts audio clips based on given intervals and saves them as WAV files.
- `transform_file_name` function: Transforms a file name using regular expressions.
- `extraire_extraits_video` function: Extracts video clips based on given intervals and saves them as MP4 files.
- `process_non_empty_file` function:
- `handle_empty_file` function: Handles an empty prediction file by creating a text file indicating no whistles were detected.
- `handle_missing_file` function: Handles a missing prediction file by creating a text file indicating no CSV file was found.
- `process_prediction_file` function: Processes a non-empty file based on given parameters. It reads intervals from a CSV file, merges them with a threshold, and checks if it should process audio only or both audio and video. Depending on the flags, it either extracts audio or video or both.
- `process_folder` function: Processes a folder by finding the corresponding prediction file and calling the `process_prediction_file` function.
- `process_prediction_files_in_folder` function: Processes all prediction files in a given folder by using multithreading to process each folder concurrently.
