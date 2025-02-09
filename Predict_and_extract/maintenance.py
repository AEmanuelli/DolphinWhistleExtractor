from tqdm import tqdm
import os
import shutil
# from predict_online_parallel import process_prediction_files_in_folderpr

# =============================================================================
#********************* MERGE
# =============================================================================
                    
def move_csv_files(folder1, folder2, folder3):
    for root, dirs, files in os.walk(folder3):
        for file in files:
            if file.endswith(".csv"):
                try:
                    video_file_wav = "_".join(file.split("_")[0:7])
                    source_file_path = os.path.join(root, file)
                    video_file = os.path.splitext(video_file_wav)[0]
                    video_target_dir = os.path.join(folder1, video_file)
                    video_target_dir_wav = os.path.join(folder1, video_file_wav)

                    # Check if the target directories exist with or without .wav extension
                    if os.path.exists(video_target_dir):
                        target_dir = video_target_dir
                    elif os.path.exists(video_target_dir_wav):
                        target_dir = video_target_dir_wav
                    else:
                        raise FileNotFoundError("No matching video directory found.")

                    # Move the CSV file to the target directory
                    target_csv_path = os.path.join(target_dir, file)
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.move(source_file_path, target_csv_path)
                    print(f"Moved CSV file to folder: {target_csv_path}")

                except IndexError:
                    print(f"Error: No matching video file found for {file}.")
                except Exception as e:
                    print(f"An error occurred: {e}")


# Chemins des dossiers
dossier_2023_extraits = "/users/zfne/emanuell/Documents/GitHub/Dolphins/DNN_whistle_detection/2023"
dossier_extraits = "/users/zfne/emanuell/Documents/GitHub/Dolphins/DNN_whistle_detection/extraits"
dossier_predictions = "/users/zfne/emanuell/Documents/GitHub/Dolphins/DNN_whistle_detection/predictions"

# move_csv_files(dossier_2023_extraits, dossier_extraits, dossier_predictions)
# print("Fusion des dossiers terminée avec succès !")



# =============================================================================
#********************* Juste extraits
# =============================================================================


import os

def rename_folders(directory):
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            if os.path.isdir(os.path.join(root, d)):  # Vérifie si c'est un dossier
                old_name = os.path.join(root, d)
                new_name = os.path.join(root, d.rstrip('.wav'))  # Supprime '.wav' s'il est présent
                if old_name != new_name:
                    os.rename(old_name, new_name)
                    print(f"Renamed folder: {old_name} to {new_name}")

# Exemple d'utilisation
directory = "/users/zfne/emanuell/Documents/GitHub/Dolphins/DNN_whistle_detection/2023"  # Remplacez par le chemin absolu ou relatif du dossier
# rename_folders(directory)

def rename_subfolders(parent_folder):
    for root, dirs, files in os.walk(parent_folder):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            subdirs = next(os.walk(dir_path))[1]  # Liste des sous-dossiers
            
            for subdir_name in subdirs:
                subdir_path = os.path.join(dir_path, subdir_name)
                
                # Vérifier si le nom du sous-dossier commence par les 10 premiers caractères du nom du dossier parent
                if subdir_name.startswith(os.path.basename(dir_path)[:10]):
                    new_name = "extraits"
                    new_subdir_path = os.path.join(dir_path, new_name)
                    
                    # Renommer le sous-dossier interne
                    os.rename(subdir_path, new_subdir_path)
                    print(f"Sous-dossier renommé : {subdir_path} -> {new_subdir_path}")
        break

# Chemin du dossier parent
# parent_folder = "/users/zfne/emanuell/Documents/GitHub/Dolphins/DNN_whistle_detection/2023"

# Appel de la fonction pour renommer les sous-dossiers
# rename_subfolders(parent_folder)
# Liste pour stocker les noms des dossiers de niveau 1 ne contenant pas de sous-dossier nommé "positive"


# parent_folder = "/media/DOLPHIN_ALEXIS1/Analyses_alexis/2023_analysed/"
parent_folder = "/media/DOLPHIN/Analyses_alexis/2023_analysed/"


folders_without_positive_subfolder = []
i=0
for directory in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, directory)
    if os.path.isdir(subfolder_path):  # Vérifie si c'est un dossier
        
        if all(subdir not in os.listdir(subfolder_path) for subdir in ["extraits", "pas_d_extraits"]):
            folders_without_positive_subfolder.append(subfolder_path)
        # if directory+".wav" in os.listdir(subfolder_path) : 
        #     import shutil
        #     mab = os.path.join(subfolder_path,directory+".wav" )
        #     print(mab)
        #     shutil.rmtree(mab)
            ####


print("fichiers dont le csv n'a pas été analysé", len(folders_without_positive_subfolder))
for i, folder in enumerate(folders_without_positive_subfolder):
    print(folder)
    # shutil.rmtree(folder)
    if i>10:
        break



import os
from moviepy.editor import VideoFileClip

def is_corrupted_mp4(file_path):
    try:
        clip = VideoFileClip(file_path)
        return False
    except Exception as e:
        # print(f"Erreur lors de la lecture de {file_path}: {e}")
        return True

def find_corrupted_mp4_files(folder_path, output_file):
    with open(output_file, 'a') as f:
        for folder in tqdm(os.listdir(folder_path)):
            folder_full_path = os.path.join(folder_path, folder)
            if os.path.isdir(folder_full_path):
                for sub_folder in os.listdir(folder_full_path):
                    sub_folder_full_path = os.path.join(folder_full_path, sub_folder)
                    if os.path.isdir(sub_folder_full_path) and sub_folder == "extraits":
                        for video in os.listdir(sub_folder_full_path):
                            video_path = os.path.join(sub_folder_full_path, video)
                            if is_corrupted_mp4(video_path):
                                f.write(video_path + '\n')

import os

parent_folder = "/media/DOLPHIN/Analyses_alexis/2023_analysed/"
num_folders_with_extracts_subfolder = 0

for directory in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, directory)
    if os.path.isdir(subfolder_path):  # Vérifie si c'est un dossier
        if "extraits" in os.listdir(subfolder_path):
            num_folders_with_extracts_subfolder += 1

# print("Nombre de dossiers contenant un sous-dossier 'extraits' :", num_folders_with_extracts_subfolder)

# output_file = "/users/zfne/emanuell/Documents/GitHub/Dolphins/DNN_whistle_detection/fichiers_corrompus.txt"
# find_corrupted_mp4_files(parent_folder, output_file)

# print("Les chemins des fichiers corrompus ont été enregistrés dans", output_file)


# def copy_files_to_folder(source_file, destination_folder):
#     # Vérifier si le fichier source existe
#     if not os.path.exists(source_file):
#         print("Le fichier source n'existe pas.")
#         return
    
#     # Créer le dossier de destination s'il n'existe pas
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
    
#     # Lire la liste des fichiers à partir du fichier source
#     with open(source_file, 'r') as f:
#         file_list = f.readlines()
    
#     # Copier chaque fichier répertorié vers le dossier de destination
#     for file_path in file_list:
#         # Supprimer les espaces blancs autour du chemin du fichier
#         file_path = file_path.strip()
#         # Vérifier si le fichier existe avant de le copier
#         if os.path.exists(file_path):
#             # Extraire le nom du fichier pour la copie
#             file_name = os.path.basename(file_path)
#             # Construire le chemin de destination
#             destination_path = os.path.join(destination_folder, file_name)
#             # Copier le fichier vers le dossier de destination
#             shutil.copy(file_path, destination_path)
#         else:
#             print(f"Le fichier {file_path} n'existe pas.")

# # Chemin du fichier source contenant la liste des fichiers à copier
# source_file_path = "/home/alexis/Documents/GitHub/Dolphins/DNN_whistle_detection/fichiers_corrompus.txt"

# # Chemin du dossier de destination sur le bureau
# destination_folder_path = "/home/alexis/Desktop/corr"

# # Copier les fichiers vers le dossier de destination
# # copy_files_to_folder(source_file_path, destination_folder_path)


def copy_and_delete_files(source_file, destination_folder):
    # Vérifier si le fichier source existe
    if not os.path.exists(source_file):
        print("Le fichier source n'existe pas.")
        return
    
    # Créer le dossier de destination s'il n'existe pas
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Lire la liste des fichiers à partir du fichier source
    with open(source_file, 'r') as f:
        file_list = f.readlines()
    
    # Initialiser le compteur de fichiers supprimés
    files_deleted = 0
    
    # Copier chaque fichier répertorié vers le dossier de destination
    for file_path in file_list:
        # Supprimer les espaces blancs autour du chemin du fichier
        file_path = file_path.strip()
        # Vérifier si le fichier existe avant de le copier
        if os.path.exists(file_path):
            # Extraire le nom du fichier pour la copie
            file_name = os.path.basename(file_path)
            # Construire le chemin de destination
            destination_path = os.path.join(destination_folder, file_name)
            try:
                # Copier le fichier vers le dossier de destination
                shutil.copy(file_path, destination_path)
                # Supprimer le fichier original
                os.remove(file_path)
                files_deleted += 1
            except Exception as e:
                print(f"Erreur lors de la copie/suppression du fichier {file_path}: {e}")
        else:
            print(f"Le fichier {file_path} n'existe pas.")

    return files_deleted

# Chemin du fichier source contenant la liste des fichiers à copier/supprimer
# source_file_path = output_file

# # Chemin du dossier de destination sur le bureau
# destination_folder_path = "/users/zfne/emanuell/Desktop/corr"

# # Copier les fichiers vers le dossier de destination et supprimer les originaux
# deleted_count = copy_and_delete_files(source_file_path, destination_folder_path)

# print(f"Nombre de fichiers supprimés : {deleted_count}")



import os



def get_videos_extracts_list(parent_folder = "/media/DOLPHIN/Analyses_alexis/2023_analysed/"):
    all_video_files = [] 

    for directory in tqdm(os.listdir(parent_folder)):
        subfolder_path = os.path.join(parent_folder, directory)
        if os.path.isdir(subfolder_path):  # Vérifie si c'est un dossier
            if "extraits_avec_audio" in os.listdir(subfolder_path) and "positive" in os.listdir(subfolder_path):
                folder_path = os.path.join(subfolder_path, "extraits_avec_audio")
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        all_video_files.append(file_path)
    return all_video_files 

video_files = get_videos_extracts_list()

for file in video_files:
    print(file)
import os
import shutil

def move_folders_with_2024(src_directory, dst_directory):
    # Vérifie si le répertoire source existe
    if not os.path.exists(src_directory):
        print(f"Le répertoire source {src_directory} n'existe pas.")
        return
    
    # Vérifie si le répertoire destination existe, sinon le crée
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)
        print(f"Le répertoire destination {dst_directory} a été créé.")
    
    # Parcourt tous les éléments du répertoire source
    for folder_name in os.listdir(src_directory):
        # Vérifie si l'élément est un dossier et s'il contient "2024"
        if os.path.isdir(os.path.join(src_directory, folder_name)) and "2024" in folder_name:
            # Chemin complet du dossier source
            src_folder_path = os.path.join(src_directory, folder_name)
            # Chemin complet du dossier de destination
            dst_folder_path = os.path.join(dst_directory, folder_name)
            
            # Déplace le dossier
            shutil.move(src_folder_path, dst_folder_path)
            print(f"Dossier déplacé: {folder_name}")

# Chemin du répertoire source
source_directory = '/media/DOLPHIN/Analyses_alexis/2023_analysed'
# Chemin du répertoire de destination
destination_directory = '/media/DOLPHIN/Analyses_alexis/2024_analysed'

# Appelle la fonction pour déplacer les dossiers
move_folders_with_2024(source_directory, destination_directory)