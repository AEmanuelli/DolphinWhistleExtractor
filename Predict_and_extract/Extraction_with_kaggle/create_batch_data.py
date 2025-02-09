import os
import zipfile

# Fonction pour créer des archives
def create_archives(src_directory, dest_directory, batch_size):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    
    batch_number = 1
    current_size = 0
    current_files = []
    
    for root, dirs, files in os.walk(src_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            
            # Si l'ajout de ce fichier dépasse la taille maximale du lot, créer un nouvel archive
            if current_size + file_size > batch_size:
                archive_name = os.path.join(dest_directory, f'batch_{batch_number}.zip')
                with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as archive:
                    for file_path in current_files:
                        archive.write(file_path, arcname=os.path.relpath(file_path, src_directory))
                batch_number += 1
                current_files = []
                current_size = 0
            
            # Ajouter le fichier au lot actuel
            current_files.append(file_path)
            current_size += file_size
    
    # Créer un archive pour les fichiers restants
    if current_files:
        archive_name = os.path.join(dest_directory, f'batch_{batch_number}.zip')
        with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_STORED) as archive:
            for file_path in current_files:
                archive.write(file_path, arcname=os.path.relpath(file_path, src_directory))

# Définir les répertoires source et destination
source_directory = '/Users/apple/Desktop/2018'
destination_directory = '/Users/apple/Desktop/Aquarium sotto 2017'
batch_size_bytes = 80 * 1024 * 1024 * 1024  # 80 Go en octets

# Créer les archives par batch de 80 Go
create_archives(source_directory, destination_directory, batch_size_bytes)
