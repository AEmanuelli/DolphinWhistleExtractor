import os
import csv
from tqdm import tqdm

def extract_info(filename):
    parts = filename.rsplit('-', 1)
    file_name = parts[0]
    timing = parts[1].split('.')[0]  # Pour enlever l'extension de fichier (.jpg)
    return file_name, timing

def main():
    folder_path_1 = "/media/DOLPHIN/Analyses_alexis/dataset/_last/positives"  # Remplacez cela par le chemin de votre dossier
    output_csv_1 = "positives.csv"
    folder_path_0 = "/media/DOLPHIN/Analyses_alexis/dataset/_last/negatives"  # Remplacez cela par le chemin de votre dossier
    output_csv_0 = "negatives.csv"
    
    files_0 = sorted(os.listdir(folder_path_0))
    files_1 = sorted(os.listdir(folder_path_1))
    
    with open(output_csv_0, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File', 'Timing'])
        
        for file in tqdm(files_0, desc="Traitement des fichiers dans le dossier negatives"):
            if file.endswith('.jpg'):
                file_name, timing = extract_info(file)
                writer.writerow([file_name, timing])
                
    with open(output_csv_1, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File', 'Timing'])
        
        for file in tqdm(files_1, desc="Traitement des fichiers dans le dossier positives"):
            if file.endswith('.jpg'):
                file_name, timing = extract_info(file)
                writer.writerow([file_name, timing])


# if __name__ == "__main__":
#     main()







# CREATION DU CSV LABELS



import os
import pandas as pd
import glob

# Chemins des répertoires contenant les images positives et négatives
positive_dir = "/media/DOLPHIN/Analyses_alexis/dataset/_last/positives"
negative_dir = "/media/DOLPHIN/Analyses_alexis/dataset/_last/negatives"
# Listes pour stocker les chemins des fichiers et les étiquettes
file_paths = []
labels = []

# Parcours des fichiers d'images positives
for file_path in glob.glob(os.path.join(positive_dir, '*.jpg')):
    file_paths.append(os.path.basename(file_path))
    labels.append(1)  # Étiquette pour les images positives

# Parcours des fichiers d'images négatives
for file_path in glob.glob(os.path.join(negative_dir, '*.jpg')):
    file_paths.append(os.path.basename(file_path))
    labels.append(0)  # Étiquette pour les images négatives

# Création du DataFrame
data = {'file_names': file_paths, 'labels': labels}
df = pd.DataFrame(data)

# Enregistrement du DataFrame en tant que fichier CSV
# df.to_csv('/home/alexis/Desktop/labels.csv', index=False)

# print("Le fichier CSV a été généré avec succès !")


# import os
# import pandas as pd
# import glob
# from sklearn.model_selection import train_test_split
# import shutil 

# # Chemins des répertoires contenant les images positives et négatives
# positive_dir = "/media/DOLPHIN/Analyses_alexis/dataset/_last/positives"
# negative_dir = "/media/DOLPHIN/Analyses_alexis/dataset/_last/negatives"
# # Listes pour stocker les chemins des fichiers et les étiquettes
# file_paths = []
# labels = []

# # Parcours des fichiers d'images positives
# for file_path in glob.glob(os.path.join(positive_dir, '*.jpg')):
#     file_paths.append(os.path.basename(file_path))
#     labels.append(1)  # Étiquette pour les images positives

# # Parcours des fichiers d'images négatives
# for file_path in glob.glob(os.path.join(negative_dir, '*.jpg')):
#     file_paths.append(os.path.basename(file_path))
#     labels.append(0)  # Étiquette pour les images négatives

# # Création du DataFrame
# data = {'file_names': file_paths, 'labels': labels}
# df = pd.DataFrame(data)

# # Répartition des données en ensembles d'entraînement et de test
# train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'])

# # Création des dossiers train et test s'ils n'existent pas déjà
# train_dir = "/media/DOLPHIN/Analyses_alexis/dataset4kag/train"
# test_dir = "/media/DOLPHIN/Analyses_alexis/dataset4kag/test"
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(test_dir, exist_ok=True)

# # Copie des images dans les dossiers train et test
# for _, row in train_df.iterrows():
#     src = os.path.join(positive_dir if row['labels'] == 1 else negative_dir, row['file_names'])
#     dst = os.path.join(train_dir, row['file_names'])
#     shutil.copy(src, dst)

# for _, row in test_df.iterrows():
#     src = os.path.join(positive_dir if row['labels'] == 1 else negative_dir, row['file_names'])
#     dst = os.path.join(test_dir, row['file_names'])
#     shutil.copy(src, dst)

# print("Les données ont été réparties avec succès entre les dossiers train et test !")


import shutil
import zipfile

# Chemin du dossier à archiver
folder_to_archive = "/media/DOLPHIN/Analyses_alexis/dataset4kag/"

# Nom de l'archive
archive_name = "/media/DOLPHIN/Analyses_alexis/dataset4kag.zip"

# Création de l'archive
with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(folder_to_archive):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.relpath(file_path, folder_to_archive))

print("L'archive a été créée avec succès :", archive_name)

