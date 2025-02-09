import os
import re
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal.windows import blackman
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import moviepy.editor as mp
import os
from tqdm import tqdm

import librosa
import numpy as np

#%%
# =============================================================================
#********************* FUNCTIONS 
# =============================================================================

def name_saving_folder(base_folder):
    """
    Obtient le chemin absolu du dossier de sauvegarde en fonction du dossier de base fourni.

    Args:
        base_folder (str): Le chemin du dossier de base à partir duquel le dossier de sauvegarde doit être généré.

    Returns:
        str: Le chemin absolu du dossier de sauvegarde, en tenant compte du numéro de dossier incrémenté le cas échéant.
    """
    # Obtenir le chemin complet du dossier avec le suffixe "_last"
    saving_folder = os.path.join(os.path.dirname(base_folder), f"{os.path.basename(base_folder)}_last")

    # Vérifier si un autre dossier porte le même nom
    if os.path.exists(saving_folder):
        # Renommer le dossier existant pour remplacer "last" par le numéro correct
        existing_folders = [f for f in os.listdir(os.path.dirname(base_folder)) if f.startswith(os.path.basename(base_folder))]
        if existing_folders:
            # Filtrer les noms de dossiers qui ont un numéro valide
            valid_folders = [f for f in existing_folders if f.split('_')[-1].isdigit()]
            if valid_folders:
                # Obtenir le numéro le plus élevé parmi les dossiers valides
                latest_folder = max(valid_folders, key=lambda f: int(f.split('_')[-1]))
                folder_number = int(latest_folder.split('_')[-1]) + 1
            else:
                folder_number = 1
        else:
            folder_number = 1

        # Renommer le dossier existant
        os.rename(saving_folder, os.path.join(os.path.dirname(base_folder), f"{os.path.basename(base_folder)}_{folder_number}"))

    # Retourner le chemin absolu du dossier avec le suffixe "_last"
    return saving_folder

def count_lines_in_csv(csv_file_path):
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        return sum(1 for row in reader)

#*********************  Whistle to video 
def convertir_texte_en_csv(fichier_texte, fichier_csv, delimiteur="\t", skip_lines = 1):
    """
    Convertit un fichier texte en fichier CSV en traitant une ligne sur deux.

    Args:
        fichier_texte (str): Chemin vers le fichier texte d'entrée.
        fichier_csv (str): Chemin vers le fichier CSV de sortie.
        delimiteur (str, optional): Délimiteur utilisé dans le fichier texte. Par défaut, '\t' (tabulation).
    """
    # Ouvrir le fichier texte en mode lecture
    with open(fichier_texte, 'r') as file:
        lignes = file.readlines()  # Lire toutes les lignes du fichier texte

    # Ouvrir le fichier CSV en mode écriture
    with open(fichier_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Parcourir chaque ligne du fichier texte, en commençant par la deuxième ligne (indice 1)
        for i in range(skip_lines, len(lignes), skip_lines):
            ligne = lignes[i]  # Sélectionner la ligne correspondante
            # Séparer les données en colonnes en utilisant le délimiteur
            colonnes = ligne.strip().split(delimiteur)
            # Écrire les colonnes dans le fichier CSV
            writer.writerow(colonnes)

    print("Conversion terminée avec succès.")

def lire_csv_extraits(nom_fichier):
    intervals = []
    with open(nom_fichier, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Passer à la deuxième ligne pour ignorer le titre des colonnes
        for row in reader:
            try:
                # Convertir les valeurs en nombres flottants et les ajouter à la liste
                debut = float(row[0])
                fin = float(row[1])
                intervals.append((debut, fin))
            except ValueError:
                # Gérer les erreurs de conversion en nombres flottants dans la première colonne
                #print("Erreur de conversion en nombre flottant. La première colonne sera ignorée.")
                debut = float(row[1])
                fin = float(row[2])  # Commencer à partir de la deuxième colonne
                intervals.append((debut, fin))
    return intervals

def fusionner_intervalles(intervalles, hwindow=4):
    # Trier les intervalles par début
    intervalles.sort(key=lambda x: x[0])
    
    # Initialiser la liste résultante
    intervalles_fusionnes = []
    
    # Parcourir les intervalles
    for intervalle in intervalles:
        debut, fin = intervalle
        
        # Arrondir la borne initiale à la seconde inférieure et convertir en entier
        debut_arrondi = int(debut) - hwindow
        
        # Arrondir la borne finale à la seconde supérieure et convertir en entier
        fin_arrondi = int(fin + 0.9999) + hwindow
        
        # Si la liste résultante est vide ou si l'intervalle ne se chevauche pas avec le dernier intervalle fusionné
        if not intervalles_fusionnes or debut_arrondi > intervalles_fusionnes[-1][1]:
            intervalles_fusionnes.append((debut_arrondi, fin_arrondi))
        else:
            # Fusionner l'intervalle avec le dernier intervalle fusionné
            intervalles_fusionnes[-1] = (intervalles_fusionnes[-1][0], max(intervalles_fusionnes[-1][1], fin_arrondi))
    
    # Ajuster les bornes du premier et du dernier intervalle fusionné si nécessaire
    if intervalles_fusionnes:
        premier_intervalle = intervalles_fusionnes[0]
        dernier_intervalle = intervalles_fusionnes[-1]
        
        if premier_intervalle[0] < 0:
            premier_intervalle = (0, premier_intervalle[1])
        
        # if dernier_intervalle[1] > 1800:
        #     dernier_intervalle = (dernier_intervalle[0], 1800)
        
        intervalles_fusionnes[0] = premier_intervalle
        intervalles_fusionnes[-1] = dernier_intervalle
    
    return intervalles_fusionnes


# FUNCTION TO EXTRACT AUDIO CLIPS FROM A CSV FILE, DEFINED WITH CHIARA FOR WAV2VEC PROJECT
def fusionner_intervalles_avec_seuil(intervalles, fusion_threshold=3, duration_threshold=3):
    """
    Fusionne les intervalles qui se chevauchent avec un seuil de fusion et une durée minimale.

    Args:
        intervalles (list): Une liste d'intervalles représentés par des tuples (début, fin).
        fusion_threshold (int, optional): Le seuil de fusion pour déterminer si deux intervalles peuvent être fusionnés. 
            Les intervalles dont la différence entre le début du deuxième intervalle et la fin du premier intervalle est inférieure ou égale à ce seuil seront fusionnés. 
            Par défaut, fusion_threshold est de 3.
        duration_threshold (int, optional): La durée minimale requise pour qu'un intervalle fusionné soit ajouté à la liste résultante. 
            Les intervalles dont la différence entre le début et la fin est supérieure à ce seuil seront ajoutés. 
            Par défaut, duration_threshold est de 3.

    Returns:
        list: Une liste d'intervalles fusionnés qui satisfont les critères de fusion et de durée minimale.
    """
    # Trier les intervalles par début
    intervalles.sort(key=lambda x: x[0])
    
    # Initialiser la liste résultante
    intervalles_fusionnes = []
    
    # Initialiser le premier intervalle
    if not intervalles:
        return intervalles_fusionnes
    
    current_debut, current_fin = intervalles[0]
    
    # Parcourir les intervalles
    for intervalle in intervalles[1:]:
        debut, fin = intervalle
        
        # Vérifier si l'intervalle actuel peut être fusionné avec le précédent
        if debut - current_fin <= fusion_threshold:
            current_fin = max(current_fin, fin)
        else:
            if current_fin - current_debut > duration_threshold:
                intervalles_fusionnes.append((max(0, current_debut), current_fin))
            current_debut = debut
            current_fin = fin
    
    # Ajouter le dernier intervalle s'il dure plus de duration_threshold 
    if current_fin - current_debut > duration_threshold:
        intervalles_fusionnes.append((max(0, current_debut), current_fin))
    
    return intervalles_fusionnes


def trouver_fichier_video(fichier_csv, dossier_videos="/media/DOLPHIN/2023/"):
    """
    Finds a video file that corresponds to a given CSV file based on their date and time.

    Args:
        fichier_csv (str): The name of the CSV file.
        dossier_videos (str, optional): The directory where the video files are located. Defaults to "/media/DOLPHIN/2023/".

    Returns:
        str: The path of the video file that corresponds to the given CSV file.
            Returns None if no corresponding video file is found.
    """
    # Extraire la date et l'heure du nom de fichier CSV
    elements_nom_csv = fichier_csv.split("_")
    date_heure_csv = elements_nom_csv[1] + "_" + elements_nom_csv[2] + "_" + elements_nom_csv[3] + "_" + elements_nom_csv[4]  # Format: "Aug_2023_0845"

    # Parcourir tous les fichiers vidéo dans le dossier
    for fichier_video in os.listdir(dossier_videos):
        if fichier_video.endswith(".mp4"):
            try:
                # Extraire la date et l'heure du nom de fichier vidéo
                elements_nom_video = fichier_video.split("_")
                date_heure_video = elements_nom_video[1] + "_" + elements_nom_video[2] + "_" + elements_nom_video[3] + "_" + elements_nom_video[4]  # Format: "Aug_2023_1645"
            except IndexError:
                # print(f"{fichier_video} n'a pas le bon format.")
                continue
            # Comparer les dates et heures pour trouver une correspondance
            if date_heure_csv == date_heure_video:
                return os.path.join(dossier_videos, fichier_video)

    print(f"Aucun fichier vidéo correspondant trouvé pour le fichier CSV {fichier_csv}")
    return None

#********************* Audio to Spectrogram

def spectrogram_f(samples, sample_rate, stride_ms = 5.0, 
                window_ms = 10.0, max_freq = np.inf, min_freq = 0):
    """
    Compute a spectrogram with consecutive Fourier transforms.

    Parameters
    ----------
    samples : waveform samples.
    sample_rate : acquisition frequency.
    stride_ms : overlap stride used in ms. The default is 5.0.
    window_ms : window used in ms. The default is 10.0.
    max_freq : frequency maximum. The default is np.inf.
    min_freq : frequency minimum. The default is 0.

    Returns
    -------
    specgram
    freqs 
    times

    """
    
    # Get number of points for each window and stride
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)
    

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape = nshape, strides = nstrides)
    
    times = np.linspace(0, (len(samples)+truncate_size)/sample_rate, nshape[1])
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Select the spectogram window  
    ind_max = np.where(freqs <= max_freq)[0][-1] + 1
    ind_min = np.where(freqs >= min_freq)[0][0]
    freqs = freqs[ind_min:ind_max]
    
    # specgram = np.log(fft[ind_min:ind_max, :] + eps)
    specgram = fft[ind_min:ind_max, :]
        
    return specgram, freqs, times

def wav_to_spec(recording_path, stride_ms = 5.0, window_ms = 10.0, max_freq = np.inf, min_freq = 0, cut=None):
    """
    Function to get spectrogram from recording path directly. More effecient way to use the memory.

    Parameters
    ----------
    recording_path
    stride_ms : overlap stride used in ms. The default is 5.0.
    window_ms : window used in ms. The default is 10.0.
    max_freq : frequency maximum. The default is np.inf.
    min_freq : frequency minimum. The default is 0.
    cut : optional. Used for selecting a portion of the recording.
          Use a tuple (s_start, s_end) where s_start and s_end are 
          the beginning and the end of the section in seconds.

    Returns
    -------
    specgram
    freqs 
    times

    """
    
    sample_rate, samples = wavfile.read(recording_path)
    try:
        samples = samples[:,0]
    except IndexError:
        pass
    
    specgram, freqs, times = spectrogram_f(samples, sample_rate, stride_ms=5.0, window_ms=10.0, 
                                         max_freq=max_freq, min_freq=min_freq)
    
    if cut is not None:
        s_start, s_end = cut
        specgram = specgram[:,(times > s_start)&(times < s_end)]
        times = times[(times > s_start)&(times < s_end)]
    
    return specgram, freqs, times

def wav_to_spec_librosa(recording_path, stride_ms=5.0, window_ms=10.0, max_freq=np.inf, min_freq=0, cut=None):
    """
    Function to get spectrogram from recording path directly. More efficient way to use memory.

    Parameters
    ----------
    recording_path : str
        Path to the WAV file.
    stride_ms : float, optional
        Overlap stride used in milliseconds. The default is 5.0.
    window_ms : float, optional
        Window used in milliseconds. The default is 10.0.
    max_freq : float, optional
        Maximum frequency. The default is np.inf.
    min_freq : float, optional
        Minimum frequency. The default is 0.
    cut : tuple, optional
        Used for selecting a portion of the recording.
        Use a tuple (s_start, s_end) where s_start and s_end are 
        the beginning and the end of the section in seconds.

    Returns
    -------
    specgram : np.ndarray
        Spectrogram of the audio.
    freqs : np.ndarray
        Frequencies.
    times : np.ndarray
        Time axis.

    """
    
    # Load audio file
    samples, sample_rate = librosa.load(recording_path, sr=None, mono=True)
    
    # Compute spectrogram
    specgram = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_fft=int(sample_rate * window_ms / 1000),
                                              hop_length=int(sample_rate * stride_ms / 1000),
                                              fmax=max_freq, fmin=min_freq)
    specgram_db = librosa.power_to_db(specgram, ref=np.max)
    
    # Compute time axis
    times = np.arange(specgram.shape[1]) * (window_ms - stride_ms) / 1000.0
    
    # Compute frequency axis
    freqs = librosa.core.mel_frequencies(n_mels=specgram.shape[0], fmin=min_freq, fmax=max_freq)
    
    # Apply cut if specified
    if cut is not None:
        s_start, s_end = cut
        start_frame = int(s_start * sample_rate / (window_ms - stride_ms))
        end_frame = int(s_end * sample_rate / (window_ms - stride_ms))
        specgram_db = specgram_db[:, start_frame:end_frame]
        times = times[start_frame:end_frame]
    
    return specgram_db, freqs, times

def plot_spectogram(specgram, freqs, times, ms=False, log=True, eps= 1e-14):
    """
    Plot the spectrogram.

    Parameters
    ----------
    specgram
    freqs
    times
    ms : Plot with the time in seconds. The default is False.
    log : The default is True.
    eps : The default is 1e-14.

    Returns
    -------
    ax

    """
    if ms :
        fig, ax = plt.subplots(figsize=(10, 5))
        if log :
            ax.pcolormesh(times, freqs, np.log(specgram + eps), shading='auto', cmap='viridis')
        else:
            ax.pcolormesh(times, freqs, specgram, shading='auto', cmap='viridis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [ms]')
        # plt.xticks(np.arange(0, times[-1]+0.1, step=0.1))
        plt.yticks(np.arange(freqs[0],freqs[-1], step=1000))
        plt.title('Spectrogram')
    else :
        fig, ax = plt.subplots(figsize=(10, 5))
        if log :
            ax.pcolormesh(np.arange(0, specgram.shape[1]), freqs, np.log(specgram + eps), shading='auto', cmap='viridis')
        else:
            ax.pcolormesh(np.arange(0, specgram.shape[1]), freqs, specgram, shading='auto', cmap='viridis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time')
        # plt.xticks(np.arange(0, specgram.shape[1], step=50))
        plt.yticks(np.arange(freqs[0],freqs[-1], step=1000))
        plt.title('Spectrogram')

    return ax

def transform_file_name(file_name):
    # Utilisation d'une expression régulière pour extraire les parties nécessaires du nom de fichier
    match = re.match(r'Exp_(\d{2})_(\w{3})_(\d{4})_(\d{4})_channel_(\d)', file_name)
    if match:
        day = match.group(1)
        month = match.group(2)
        year = match.group(3)[2:]  # Récupération des deux derniers chiffres de l'année
        time = match.group(4)
        channel = match.group(5)
        transformed_name = f"{day}_{month}_{year}_{time}_c{channel}"
        return transformed_name
    else:
        return None

def prepare_csv_data(file_path, record_names, positive_initial, positive_finish):
    part = file_path.split('wav-')

    name = part[0] + "wav"
    record_names.append(name)
    
    ini = part[1].replace(".jpg", "")
    ini = float(ini)
    positive_initial.append(ini)
    
    fin = ini + 0.8
    fin = round(fin, 1)
    positive_finish.append(fin)
    
    return record_names, positive_initial, positive_finish

def save_csv(record_names, positive_initial, positive_finish, class_1_scores, csv_path):
    df = {'file_name': record_names,
    'initial_point': positive_initial,
    'finish_point': positive_finish,
    'confidence': class_1_scores}

    df = pd.DataFrame(df)
    
    df.to_csv(csv_path, index=False)

def process_audio_file(file_path, saving_folder="./images", batch_size=50, start_time=0, end_time=None, save=False, wlen=2048,
                       nfft=2048, sliding_w=0.4, cut_low_frequency=3, cut_high_frequency=20, target_width_px=903,
                       target_height_px=677):
    """
    Process an audio file and generate spectrogram images.

    Parameters:
    - file_path (str): Path to the audio file.
    - saving_folder (str): Path to the folder where the spectrogram images will be saved. Default is "./images".
    - batch_size (int): Number of spectrogram images to generate. Default is 50.
    - start_time (float): Start time in seconds for processing the audio file. Default is 0.
    - end_time (float): End time in seconds for processing the audio file. Default is None.
    - save (bool): Whether to save the spectrogram images. Default is False.
    - wlen (int): Length of the window for spectrogram calculation. Default is 2048.
    - nfft (int): Number of points for FFT calculation. Default is 2048.
    - sliding_w (float): Sliding window size in seconds. Default is 0.4.
    - cut_low_frequency (int): Lower frequency limit for the spectrogram plot. Default is 3.
    - cut_high_frequency (int): Upper frequency limit for the spectrogram plot. Default is 20.
    - target_width_px (int): Width of the spectrogram image in pixels. Default is 903.
    - target_height_px (int): Height of the spectrogram image in pixels. Default is 677.

    Returns:
    - images (list): List of spectrogram images as numpy arrays.

    Raises:
    - FileNotFoundError: If the audio file is not found.
    """

    try:
        # Load sound recording
        fs, x = wavfile.read(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    
    # Create the saving folder if it doesn't exist
    if save and not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    
    # Calculate the spectrogram parameters
    hop = round(0.8 * wlen)  # window hop size
    win = blackman(wlen, sym=False)

    images = []
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    N = len(x)  # signal length

    if end_time is not None:
        N = min(N, int(end_time * fs))

    low = int(start_time * fs)
    
    samples_per_slice = int(sliding_w * fs)

    for _ in range(batch_size):
        if low + samples_per_slice > N:  # Check if the slice exceeds the signal length
            break
        x_w = x[low:low + samples_per_slice]
        
        # Calculate the spectrogram
        f, t, Sxx = spectrogram(x_w, fs, nperseg=wlen, noverlap=hop, nfft=nfft, window=win)
        Sxx = 20 * np.log10(np.abs(Sxx)+1e-14)  # Convert to dB

        # Create the spectrogram plot
        fig, ax = plt.subplots()
        ax.pcolormesh(t, f / 1000, Sxx, cmap='gray')
        ax.set_ylim(cut_low_frequency, cut_high_frequency)

        ax.set_axis_off()  # Turn off axis
        fig.set_size_inches(target_width_px / plt.rcParams['figure.dpi'], target_height_px / plt.rcParams['figure.dpi'])
        
        # Adjust margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
      
        # Save the spectrogram as a JPG image without borders
        if save:
            image_name = os.path.join(saving_folder, f"{file_name}-{low/fs}.jpg")
            fig.savefig(image_name, dpi=plt.rcParams['figure.dpi'])  # Save without borders

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

        low += samples_per_slice

    plt.close('all')  # Close all figures to release memory

    return images

# def process_audio_file_alternative(file_path, saving_folder="./images", batch_size=50, start_time=0, end_time=None, save=False, wlen=2048,
#                                    nfft=2048, sliding_w=0.4, cut_low_frequency=3, cut_high_frequency=25, target_width_px=1920,
#                                    target_height_px=1080):
#     try:
#         # Load sound recording
#         x, fs = librosa.load(file_path, sr=None)
#         print("zboub")
#     except FileNotFoundError:
#         raise FileNotFoundError(f"File {file_path} not found.")
    
#     # Create the saving folder if it doesn't exist
#     if save and not os.path.exists(saving_folder):
#         os.makedirs(saving_folder)
    
#     # Calculate the spectrogram parameters
#     hop_length = int(sliding_w * fs)
#     n_mels = 128  # Number of Mel bands
    
#     images = []
#     file_name = os.path.splitext(os.path.basename(file_path))[0]
#     N = len(x)  # signal length

#     if end_time is not None:
#         N = min(N, int(end_time * fs))

#     low = int(start_time * fs)
    
#     for _ in range(batch_size):
#         if low + hop_length > N:  # Check if the slice exceeds the signal length
#             break
#         x_w = x[low:low + hop_length]
        
#         # Calculate the Mel spectrogram
#         Sxx = librosa.feature.melspectrogram(y=x_w, sr=fs, n_fft=nfft, hop_length=hop_length, n_mels=n_mels, fmin=cut_low_frequency, fmax=cut_high_frequency)
#         Sxx = librosa.power_to_db(Sxx, ref=np.max)  # Convert to dB

#         # Create the spectrogram plot
#         fig, ax = plt.subplots()
#         librosa.display.specshow(Sxx, sr=fs, hop_length=hop_length, cmap='inferno')
#         ax.set_ylim(cut_low_frequency, cut_high_frequency)
        
#         ax.axis('off')  # Turn off axis
        
#         fig.set_size_inches(target_width_px / plt.rcParams['figure.dpi'], target_height_px / plt.rcParams['figure.dpi'])

#         # Save the spectrogram as a JPG image without borders
#         if save:
#             image_name = os.path.join(saving_folder, f"{file_name}-{low/fs}.jpg")
#             fig.savefig(image_name, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight', pad_inches=0)  # Save without borders

#         fig.canvas.draw()
#         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         images.append(image)

#         low += hop_length

#         plt.close(fig)  # Close figure to release memory

#     return images



#%%
# =============================================================================
#********************* UTILISATION
# =============================================================================


fichier_texte = "/media/DOLPHIN_ALEXIS/temp_alexis/Label_01_Aug_2023_0845_channel_1.txt"
fichier_csv = "/media/DOLPHIN_ALEXIS/temp_alexis/Label_01_Aug_2023_0845_channel_1.csv"


#********************* txt vers csv
# # Spécifiez le chemin vers le fichier texte d'entrée et le fichier CSV de sortie

# convertir_texte_en_csv(fichier_texte, fichier_csv, skip_lines=1)





# #********************* csv vers intervalles 
# nom_fichier = "/media/DOLPHIN_ALEXIS/temp_alexis/labels_channel_1.csv"
# intervalles = lire_csv_extraits(nom_fichier)
# intervalles_fusionnes = fusionner_intervalles(intervalles, hwindow = 5)
# print(intervalles_fusionnes)


# #********************* intervalles vers extraits
# fichier_video = "/media/DOLPHIN_ALEXIS/temp_alexis/1_08/Exp_01_Aug_2023_1645_cam_all.mp4"  # Chemin vers le fichier vidéo d'entrée
# dossier_sortie = "./extraits/channel_1"
# extraire_extraits_video(intervalles_fusionnes, fichier_video, dossier_sortie)


# import matlab.engine
# eng = matlab.engine.start_matlab()
# # Run a MATLAB script
# eng.eval("/users/zfne/emanuell/Documents/GitHub/Dolphins/DNN_whistle_detection/save_spectrogram.m", nargout=0)

# # Call a MATLAB function
# result = eng.my_matlab_function(arg1, arg2)


        
        

