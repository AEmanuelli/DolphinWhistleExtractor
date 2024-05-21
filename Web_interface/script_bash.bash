#!/bin/bash

# Chemin vers le dossier source
source_dir="/media/DOLPHIN/Analyses_alexis/2023_analysed"

# Chemin vers le dossier de destination
destination_dir="/media/DOLPHIN/Analyses_alexis/upload_online_gt20"

# Créer le dossier de destination s'il n'existe pas
mkdir -p "$destination_dir"

# Parcourir les sous-dossiers du dossier source
for subdir in "$source_dir"/*; do
    # Vérifier si le chemin est un dossier
    if [ -d "$subdir" ]; then
        # Nom du sous-dossier
        subdir_name=$(basename "$subdir")

        # Créer le même sous-dossier dans le dossier de destination
        mkdir -p "$destination_dir/$subdir_name"

        # Copier les sous-dossiers "positive" et "extraits_avec_audio"
        cp -r "$subdir"/positive "$destination_dir/$subdir_name"
        cp -r "$subdir"/extraits_avec_audio "$destination_dir/$subdir_name"

        # Parcourir les fichiers dans le sous-dossier "extraits_avec_audio"
        for file in "$subdir"/extraits_avec_audio/*; do
            # Vérifier si le chemin est un fichier
            if [ -f "$file" ]; then
                # Nom du fichier
                filename=$(basename "$file")
                
                # Extraire la durée du fichier du nom du fichier
                duration=$(echo "$filename" | cut -d '_' -f 3 | cut -d '.' -f 1)
                
                # Vérifier si la durée dépasse 20 secondes
                if [ "$duration" -gt 20 ]; then
                    # Copier le fichier dans le dossier de destination
                    cp "$file" "$destination_dir/$subdir_name/extraits_avec_audio/$filename"
                fi
            fi
        done
    fi
done



#!/bin/bash

# Chemin vers le dossier des extraits avec audio
source_dir="/media/DOLPHIN/Analyses_alexis/First_launch_website_content"

# Initialiser un tableau JSON
echo "[" > paths.json

# Parcourir les sous-dossiers de niveau 2 nommés "extraits_avec_audio"
find "$source_dir" -mindepth 2 -maxdepth 2 -type d -name "extraits_avec_audio" | while read -r subdir; do
    # Récupérer les chemins complets des fichiers dans le sous-dossier "extraits_avec_audio"
    find "$subdir" -type f | while read -r file; do
        # Remplacer "source_dir" par "zizi" dans le chemin
        modified_path="${file//$source_dir/https://hub.bio.ens.psl.eu/index.php/s/CLGWSJXWESGEKRn/download?path=&files=}"
        # Ajouter le chemin modifié au tableau JSON
        echo "\"$modified_path\"," >> paths.json
    done
done

# Supprimer la virgule finale et fermer le tableau JSON
sed -i '$s/,$//' paths.json
echo "]" >> paths.json
