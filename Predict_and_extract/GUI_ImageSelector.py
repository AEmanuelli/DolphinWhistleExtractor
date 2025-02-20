import os
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Demande des chemins avec valeurs par défaut
file_list_path = input("Entrez le chemin du fichier liste [/home/emanuelli/Bureau/Dolphins_dataset/test_fp]: ") or "/home/emanuelli/Bureau/Dolphins_dataset/test_fp"
base_dataset = input("Entrez le chemin du dataset de base [/home/emanuelli/Bureau/Dolphins_dataset]: ") or "/home/emanuelli/Bureau/Dolphins_dataset"

positive_dir = os.path.join(base_dataset, "positives")
negative_dir = os.path.join(base_dataset, "negatives")

# Valeur de pas attendue pour considérer les images comme immédiatement collées
PAS_ATTENDU = 0.4
TOLERANCE = 1e-6

# --- Fonctions utilitaires ---

def read_file_list(file_path):
    if not os.path.exists(file_path):
        messagebox.showerror("Erreur", f"Le fichier {file_path} n'existe pas.")
        exit(1)
    with open(file_path, "r") as f:
        lines = f.readlines()
    # Enlever espaces et lignes vides
    files = [line.strip() for line in lines if line.strip()]
    return files

def update_file_list():
    """Réécrit le fichier liste avec les images non encore traitées."""
    with open(file_list_path, "w") as f:
        for path, _ in current_images:
            f.write(path + "\n")

def determine_category(filepath):
    """Retourne 'positives' ou 'negatives' si le chemin contient ces mots, sinon 'inconnu'."""
    if "positives" in filepath:
        return "positives"
    elif "negatives" in filepath:
        return "negatives"
    else:
        return "inconnu"

def extract_prefix_and_number(filename):
    """
    Extrait le préfixe et la valeur numérique à partir d'un nom de fichier.
    Exemple : "Exp_02_Feb_2020_1545pm-1942.0.jpg" → ("Exp_02_Feb_2020_1545pm", 1942.0)
    """
    base, ext = os.path.splitext(filename)
    if '-' in base:
        prefix, num_str = base.rsplit('-', 1)
        try:
            num = float(num_str)
        except ValueError:
            num = None
        return prefix, num
    return None, None

# --- Préparation de la liste des images ---
file_paths = read_file_list(file_list_path)
# Chaque entrée est un tuple (chemin, catégorie déduite)
current_images = []
for path in file_paths:
    cat = determine_category(path)
    current_images.append((path, cat))

# --- Interface graphique principale ---
root = tk.Tk()
root.title("Revue des images - Correction de classification")

# Zone d'affichage de l'image courante
image_label = tk.Label(root)
image_label.pack()

# Label d'information (chemin et catégorie)
info_label = tk.Label(root, text="", font=("Helvetica", 10))
info_label.pack(pady=5)

# Variable globale pour l'image affichée
img = None

def show_current_image():
    """Affiche l'image courante (premier élément de current_images)."""
    global img
    if not current_images:
        messagebox.showinfo("Fin", "Aucune image à afficher.")
        root.quit()
        return
    current_filepath, current_category = current_images[0]
    try:
        pil_image = Image.open(current_filepath)
        pil_image.thumbnail((800, 600))
        img = ImageTk.PhotoImage(pil_image)
        image_label.config(image=img)
        info_label.config(text=f"Fichier : {current_filepath}\nCatégorie actuelle : {current_category}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible d'ouvrir l'image : {current_filepath}\n{e}")
        process_current_image('skip')

def process_current_image(action):
    """
    Selon l'action ('move' ou 'skip'), traite l'image courante,
    la retire de la liste et met à jour le fichier liste.
    """
    if not current_images:
        return
    current_filepath, current_category = current_images[0]
    if action == 'move':
        if current_category == "positives":
            target_path = current_filepath.replace("positives", "negatives")
            target_category = "negatives"
        elif current_category == "negatives":
            target_path = current_filepath.replace("negatives", "positives")
            target_category = "positives"
        else:
            messagebox.showerror("Erreur", f"Catégorie inconnue pour le fichier: {current_filepath}")
            return
        try:
            shutil.move(current_filepath, target_path)
            messagebox.showinfo("Déplacement", f"Image déplacée vers {target_category}:\n{target_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de déplacer l'image:\n{e}")
    # Retirer l'image courante de la liste et mettre à jour le fichier
    current_images.pop(0)
    update_file_list()
    show_current_image()

def show_temporal_neighbors():
    """
    Recherche dans le dataset (base_dataset) les images dont le préfixe est identique à
    l'image courante et affiche celles qui sont immédiatement adjacentes (différence numérique égale à PAS_ATTENDU).
    L'image précédente est affichée à gauche et l'image suivante à droite de l'image courante.
    """
    if not current_images:
        messagebox.showerror("Erreur", "Aucune image courante.")
        return
    current_filepath, _ = current_images[0]
    current_basename = os.path.basename(current_filepath)
    prefix, current_num = extract_prefix_and_number(current_basename)
    if prefix is None or current_num is None:
        messagebox.showerror("Erreur", "Impossible d'analyser le nom de l'image actuelle pour déterminer l'ordre temporel.")
        return

    similar_images = []
    valid_ext = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    # Recherche dans l'intégralité du dataset
    for root_dir, dirs, files in os.walk(base_dataset):
        for file in files:
            if file.lower().endswith(valid_ext):
                p, n = extract_prefix_and_number(file)
                if p == prefix and n is not None:
                    full_path = os.path.join(root_dir, file)
                    similar_images.append((full_path, n))
    if not similar_images:
        messagebox.showinfo("Informations", "Aucune image similaire trouvée dans le dataset.")
        return

    similar_images.sort(key=lambda x: x[1])
    index = None
    for i, (path, n) in enumerate(similar_images):
        if os.path.basename(path) == current_basename:
            index = i
            break
    if index is None:
        messagebox.showinfo("Informations", "Image actuelle introuvable parmi les images similaires.")
        return

    # Création d'une fenêtre pour afficher les images côte à côte
    neighbor_win = tk.Toplevel(root)
    neighbor_win.title("Images temporelles voisines")
    frame = tk.Frame(neighbor_win)
    frame.pack()

    # Affichage de l'image précédente à gauche (si immédiatement collée)
    if index > 0:
        prev_path, prev_num = similar_images[index - 1]
        if abs(current_num - prev_num - PAS_ATTENDU) < TOLERANCE:
            prev_category = determine_category(prev_path)
            try:
                pil_prev = Image.open(prev_path)
                pil_prev.thumbnail((300, 300))
                img_prev = ImageTk.PhotoImage(pil_prev)
                label_prev = tk.Label(frame, image=img_prev)
                label_prev.image = img_prev  # garder une référence
                label_prev.grid(row=0, column=0, padx=10, pady=10)
                tk.Label(frame, text=f"Précédente:\n{prev_path}\nCatégorie : {prev_category}", justify="center") \
                  .grid(row=1, column=0, padx=10, pady=10)
            except Exception as e:
                tk.Label(frame, text="Erreur d'affichage de l'image précédente.", justify="center") \
                  .grid(row=0, column=0, padx=10, pady=10)
        else:
            tk.Label(frame, text="Aucune image précédente immédiatement collée.", justify="center") \
              .grid(row=0, column=0, padx=10, pady=10)
    else:
        tk.Label(frame, text="Aucune image précédente trouvée.", justify="center") \
          .grid(row=0, column=0, padx=10, pady=10)

    # Affichage de l'image actuelle au centre
    try:
        pil_current = Image.open(current_filepath)
        pil_current.thumbnail((300, 300))
        img_current = ImageTk.PhotoImage(pil_current)
        label_current = tk.Label(frame, image=img_current)
        label_current.image = img_current
        label_current.grid(row=0, column=1, padx=10, pady=10)
        tk.Label(frame, text=f"Actuelle:\n{current_filepath}", justify="center") \
          .grid(row=1, column=1, padx=10, pady=10)
    except Exception as e:
        tk.Label(frame, text="Erreur d'affichage de l'image actuelle.", justify="center") \
          .grid(row=0, column=1, padx=10, pady=10)

    # Affichage de l'image suivante à droite (si immédiatement collée)
    if index < len(similar_images) - 1:
        next_path, next_num = similar_images[index + 1]
        if abs(next_num - current_num - PAS_ATTENDU) < TOLERANCE:
            next_category = determine_category(next_path)
            try:
                pil_next = Image.open(next_path)
                pil_next.thumbnail((300, 300))
                img_next = ImageTk.PhotoImage(pil_next)
                label_next = tk.Label(frame, image=img_next)
                label_next.image = img_next
                label_next.grid(row=0, column=2, padx=10, pady=10)
                tk.Label(frame, text=f"Suivante:\n{next_path}\nCatégorie : {next_category}", justify="center") \
                  .grid(row=1, column=2, padx=10, pady=10)
            except Exception as e:
                tk.Label(frame, text="Erreur d'affichage de l'image suivante.", justify="center") \
                  .grid(row=0, column=2, padx=10, pady=10)
        else:
            tk.Label(frame, text="Aucune image suivante immédiatement collée.", justify="center") \
              .grid(row=0, column=2, padx=10, pady=10)
    else:
        tk.Label(frame, text="Aucune image suivante trouvée.", justify="center") \
          .grid(row=0, column=2, padx=10, pady=10)

# --- Boutons d'action ---
button_move = tk.Button(root, text="Déplacer vers l'autre dossier", command=lambda: process_current_image('move'), width=30)
button_move.pack(side=tk.LEFT, padx=10, pady=10)

button_next = tk.Button(root, text="Suivant", command=lambda: process_current_image('skip'), width=15)
button_next.pack(side=tk.RIGHT, padx=10, pady=10)

button_neighbors = tk.Button(root, text="Voir images temporelles", command=show_temporal_neighbors, width=30)
button_neighbors.pack(pady=10)

# Affichage de la première image
show_current_image()

root.mainloop()
