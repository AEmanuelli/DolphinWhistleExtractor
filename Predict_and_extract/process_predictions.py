# =============================================================================
#********************* IMPORTS
# =============================================================================
import os
from concurrent.futures import ThreadPoolExecutor
from utils import *
import re
from tqdm import tqdm 
# =============================================================================
#********************* FUNCTIONS
# =============================================================================


from moviepy.editor import AudioFileClip


def audioextraits(intervalles, fichier_audio, dossier_sortie_audio):

    # Chargement de l'audio
    audio = AudioFileClip(fichier_audio)
    filename = os.path.splitext(os.path.basename(fichier_audio))[0]
    
    # Calculer le nombre total d'extraits à générer
    total_extraits = len(intervalles)
    
    # Afficher une barre de progression
    with tqdm(total=total_extraits, desc=f'Extraction pour {filename}') as pbar:
        # Parcourir les intervalles
        for i, intervalle in enumerate(intervalles):
            debut, fin = intervalle
             
            nom_sortie = f'extrait_{debut}_{fin}.wav'  # Nom de sortie basé sur l'intervalle
            
            chemin_sortie = os.path.join(dossier_sortie_audio, filename, nom_sortie)  # Chemin complet de sortie

            dossier_du_fichier = os.path.join(dossier_sortie_audio, filename) 
            os.makedirs(dossier_du_fichier, exist_ok=True)
            if not os.path.exists(chemin_sortie):
                # Extraire l'extrait correspondant à l'intervalle
                extrait_audio = audio.subclip(debut, fin)
                # Sauvegarder l'extrait audio
                extrait_audio.write_audiofile(chemin_sortie, codec='pcm_s16le', verbose=False)
                print(f'ok : {chemin_sortie}')
                
            else:
                print(f"L'extrait {nom_sortie} existe déjà.")
            pbar.update(1)
        
    # Libérer la mémoire en supprimant l'objet AudioFileClip
    audio.close()

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

def extraire_extraits_video(intervalles, fichier_video, dossier_sortie_video):
    # Chargement de la vidéo
    video = mp.VideoFileClip(fichier_video)
    
    # Calculer le nombre total d'extraits à générer
    total_extraits = len(intervalles)
    
    # Afficher une barre de progression
    with tqdm(total=total_extraits, desc=f'Extraction pour {fichier_video}') as pbar:
        # Parcourir les intervalles
        for i, intervalle in enumerate(intervalles):
            debut, fin = intervalle
            nom_sortie = f'extrait_{debut}_{fin}.mp4'  # Nom de sortie basé sur l'intervalle

            chemin_sortie = os.path.join(dossier_sortie_video, nom_sortie)  # Chemin complet de sortie
            if not os.path.exists(chemin_sortie):
                # Extraire l'extrait correspondant à l'intervalle
                extrait = video.subclip(debut, fin)
                # Sauvegarder l'extrait vidéo
                extrait.write_videofile(chemin_sortie, verbose=False)
            else : 
                print("zbzbz")
            pbar.update(1)
        
    # Libérer la mémoire en supprimant l'objet VideoFileClip
    video.close()

def process_non_empty_file(prediction_file_path, folder_name, recording_folder_path, exit, folder_path, audio = True, audio_only = True):
    intervalles = lire_csv_extraits(prediction_file_path)
    
    # print(intervalles_fusionnes)
    # print(folder_name)
    
    if audio_only :
            intervalles_fusionnes = fusionner_intervalles_avec_seuil(intervalles, duration_threshold=5, fusion_threshold=3)
            fichier_audio = os.path.join(recording_folder_path, folder_name + ".wav")
            dossier_sortie_audio = exit
            os.makedirs(dossier_sortie_audio, exist_ok=True)
            audioextraits(intervalles_fusionnes, fichier_audio, dossier_sortie_audio)
    
    else : 
        intervalles_fusionnes = fusionner_intervalles(intervalles, hwindow=5)
        fichier_video = trouver_fichier_video(folder_name, recording_folder_path)
        if fichier_video:
            dossier_sortie_video = os.path.join(folder_path, "extraits")

            # # A supprimer après le premier run 
            # import shutil
            # dossier_sortie_video_a_supprimer = os.path.join(item_path, "pas_d_extraits")
            # shutil.rmtree(dossier_sortie_video_a_supprimer) if os.path.exists(dossier_sortie_video_a_supprimer) else None
            # ####

            if audio : 
                from vidéoaudio import vidéoetaudio
                fichier_audio = os.path.join(recording_folder_path, folder_name + ".wav")
                dossier_sortie_video_et_audio = dossier_sortie_video + "_avec_audio"
                os.makedirs(dossier_sortie_video_et_audio, exist_ok=True)
                vidéoetaudio(intervalles_fusionnes, fichier_video, fichier_audio, dossier_sortie_video_et_audio)
            else :
                os.makedirs(dossier_sortie_video, exist_ok=True)

                extraire_extraits_video(intervalles_fusionnes, fichier_video, dossier_sortie_video)
    
        else : 
            dossier_sortie_video = os.path.join(folder_path, "pas_d_extraits")
            os.makedirs(dossier_sortie_video, exist_ok=True)
            txt_file_path = os.path.join(dossier_sortie_video, f"No_Video_found.txt")
            with open(txt_file_path, 'w') as txt_file:
                txt_file.write(f"No video found for {folder_name}")
            print(f"Missing video file for {folder_name}. No video extraction can be performed.")

def handle_empty_file(folder_path, folder_name):
    dossier_sortie_video = os.path.join(folder_path, "pas_d_extraits")
    t_file_name = transform_file_name(folder_name)

    # A supprimer après le premier run 
    # import shutil
    # dossier_sortie_video_a_supprimer = os.path.join(folder_path, "extraits")
    # shutil.rmtree(dossier_sortie_video_a_supprimer) if (os.path.exists(dossier_sortie_video_a_supprimer) and not os.listdir(dossier_sortie_video_a_supprimer)) else None
    ####

    os.makedirs(dossier_sortie_video, exist_ok=True)
    txt_file_path = os.path.join(dossier_sortie_video, f"No_whistles_detected.txt")
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(f"No whistles detected in {t_file_name} ")
    print(f"Empty CSV file for {t_file_name}. No video extraction will be performed. A message has been saved to {txt_file_path}.")

def handle_missing_file(folder_path, folder_name):
    t_file_name = transform_file_name(folder_name)
    dossier_sortie_video = os.path.join(folder_path, "pas_d_extraits")
    os.makedirs(dossier_sortie_video, exist_ok=True)
    txt_file_path = os.path.join(dossier_sortie_video, f"No_CSV_found.txt")
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(f"No CSV found in {t_file_name}")
    print(f"Missing CSV file for {t_file_name}. No video extraction will be performed.")

def process_prediction_file(prediction_file_path, folder_name, recording_folder_path, folder_path, exit, audio, audio_only = True):
    t_file_name = transform_file_name(folder_name)
    print(f"processing : {t_file_name}")
    empty = False 
    if os.path.exists(prediction_file_path):
        # Check if the file is empty
        with open(prediction_file_path, 'r') as file:
            lines = file.readlines()
            if len(lines) <= 1:
                empty = True
    else:
        # File is missing
        handle_missing_file(folder_path, folder_name)

    if not empty:
        # File exists and is not empty
        process_non_empty_file(prediction_file_path, folder_name, recording_folder_path, folder_path=folder_path, exit = exit, audio = audio, audio_only = audio_only)
    else:
        # File exists but is empty
        handle_empty_file(folder_path, folder_name)

def process_folder(root, folder_name, recording_folder_path, folder_path, exit, audio = True, audio_only = True):
    csv_file_name = folder_name + ".wav_predictions.csv"
    # print(csv_file_name, csv_file_path)
    prediction_file_path = os.path.join(root, folder_name, csv_file_name)
    print("Prediction file path:", prediction_file_path)
    if os.path.exists(prediction_file_path): #s'assure de l'existence du ficheir csv
        process_prediction_file(prediction_file_path, folder_name, recording_folder_path, folder_path, audio = audio, audio_only = audio_only, exit = exit)

def process_prediction_files_in_folder(root, recording_folder_path, max_workers=8, audio = True, audio_only = True, exit = "/media/DOLPHIN/Analyses_alexis/WAVextracts"):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for folder_name in tqdm(reversed(os.listdir(root)), leave=False):
            folder_path = os.path.join(root, folder_name)
            if os.path.isdir(folder_path):
                executor.submit(process_folder, root, folder_name, recording_folder_path, folder_path, audio = audio, audio_only = audio_only, exit = exit)
# def process_prediction_files_in_folder(folder_path, recording_folder_path="/media/DOLPHIN_ALEXIS1/2023", max_workers = 16):
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for root, _, files in os.walk(folder_path):
#             for file_name in files:
#                 if file_name.endswith(".csv"):
#                     prediction_file_path = os.path.join(root, file_name)
#                     docname = "_".join(os.path.splitext(file_name)[0].split("_")[:7])
#                     extract_folder_path = os.path.join(root, "extraits")
#                     executor.submit(process_prediction_file, prediction_file_path, file_name, recording_folder_path, extract_folder_path)
# root = "/media/DOLPHIN/Analyses_alexis/2023_analysed/"
# recording_folder_path="/media/DOLPHIN/2023/"
# for folder_name in tqdm(reversed(os.listdir(root)), leave=False):
#             folder_path = os.path.join(root, folder_name)
#             if os.path.isdir(folder_path):
#                 process_folder(root, folder_name, recording_folder_path, folder_path, audio = True, audio_only = True, exit = "/media/DOLPHIN/Analyses_alexis/WAVextracts")