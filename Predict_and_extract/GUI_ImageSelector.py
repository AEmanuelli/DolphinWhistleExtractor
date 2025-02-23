import os
import shutil
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# Demande des chemins avec valeurs par défaut
file_list_path = input("Entrez le chemin du fichier liste [/home/emanuelli/Bureau/Dolphins_dataset/test_fp]: ") or "/home/emanuelli/Bureau/Dolphins_dataset/test_fp"
base_dataset = input("Entrez le chemin du dataset de base [/home/emanuelli/Bureau/Dolphins_dataset]: ") or "/home/emanuelli/Bureau/Dolphins_dataset"

positive_dir = os.path.join(base_dataset, "positives")
negative_dir = os.path.join(base_dataset, "negatives")

# Valeur de pas attendue pour considérer les images comme immédiatement collées
PAS_ATTENDU = 0.4
TOLERANCE = 1e-6

# Taille d'affichage pour toutes les images (les trois auront la même taille)
IMAGE_SIZE = (400, 400)

# Variable globale pour l'annulation de la dernière action
last_action = None

# --- Fonctions utilitaires ---

def read_file_list(file_path):
    if not os.path.exists(file_path):
        messagebox.showerror("Erreur", f"Le fichier {file_path} n'existe pas.")
        exit(1)
    with open(file_path, "r") as f:
        lines = f.readlines()
    files = [line.strip() for line in lines if line.strip()]
    return files

def update_file_list():
    """Réécrit le fichier liste avec les images non encore traitées."""
    with open(file_list_path, "w") as f:
        for path, _ in current_images:
            f.write(path + "\n")

def determine_category(filepath):
    if "positives" in filepath:
        return "positives"
    elif "negatives" in filepath:
        return "negatives"
    else:
        return "inconnu"

def extract_prefix_and_number(filename):
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
current_images = []
for path in file_paths:
    cat = determine_category(path)
    current_images.append((path, cat))
total_images_count = len(current_images)

# --- Interface graphique principale ---
root = tk.Tk()
root.title("Revue des images - Correction de classification")
root.geometry("1300x750")

# Cadre pour afficher les images
image_frame = tk.Frame(root)
image_frame.pack(pady=10)

left_image_label = tk.Label(image_frame)
left_image_label.grid(row=0, column=0, padx=5)

center_image_label = tk.Label(image_frame)
center_image_label.grid(row=0, column=1, padx=5)

right_image_label = tk.Label(image_frame)
right_image_label.grid(row=0, column=2, padx=5)

# Cadre pour afficher les infos de chaque image
info_frame = tk.Frame(root)
info_frame.pack(pady=5)

left_info_label = tk.Label(info_frame, text="", font=("Helvetica", 8))
left_info_label.grid(row=0, column=0, padx=5)

center_info_label = tk.Label(info_frame, text="", font=("Helvetica", 8))
center_info_label.grid(row=0, column=1, padx=5)

right_info_label = tk.Label(info_frame, text="", font=("Helvetica", 8))
right_info_label.grid(row=0, column=2, padx=5)

# Barre de progression
progress_frame = tk.Frame(root)
progress_frame.pack(pady=10, fill="x", padx=20)
progress_label = tk.Label(progress_frame, text="Progression:")
progress_label.pack(side="left")
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate", maximum=total_images_count)
progress_bar.pack(side="left", fill="x", expand=True, padx=10)
progress_text = tk.Label(progress_frame, text=f"0 / {total_images_count}")
progress_text.pack(side="right")

# Variable pour le toggle "Voir images temporelles"
show_neighbors_active = tk.BooleanVar(value=False)

# Variables globales pour les images affichées
img_center = None
img_left = None
img_right = None

def update_progress():
    processed = total_images_count - len(current_images)
    progress_bar['value'] = processed
    progress_text.config(text=f"{processed} / {total_images_count}")

def show_current_image():
    global img_center
    left_image_label.config(image="", text="")
    right_image_label.config(image="", text="")
    left_info_label.config(text="")
    right_info_label.config(text="")
    if not current_images:
        messagebox.showinfo("Fin", "Aucune image à afficher.")
        root.quit()
        return
    current_filepath, current_category = current_images[0]
    try:
        pil_image = Image.open(current_filepath)
        pil_image.thumbnail(IMAGE_SIZE)
        img_center = ImageTk.PhotoImage(pil_image)
        center_image_label.config(image=img_center)
        filename = os.path.basename(current_filepath)
        directory = os.path.dirname(current_filepath)
        center_info_label.config(text=f"Nom: {filename}\nDossier: {directory}\nCatégorie: {current_category}")
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible d'ouvrir l'image : {current_filepath}\n{e}")
        process_current_image('skip')
    if show_neighbors_active.get():
        show_temporal_neighbors()
    else:
        # Effacer les images voisines si le toggle est désactivé
        left_image_label.config(image="", text="")
        right_image_label.config(image="", text="")
        left_info_label.config(text="")
        right_info_label.config(text="")
    update_progress()

def process_current_image(action):
    global last_action
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
            last_action = {'action': 'move', 'filepath': current_filepath, 'target_path': target_path}
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de déplacer l'image:\n{e}")
            return
    elif action == 'skip':
        last_action = {'action': 'skip', 'filepath': current_filepath}
    current_images.pop(0)
    update_file_list()
    left_image_label.config(image="", text="")
    right_image_label.config(image="", text="")
    left_info_label.config(text="")
    right_info_label.config(text="")
    show_current_image()

def undo_last_action():
    global last_action
    if last_action is None:
        messagebox.showinfo("Annuler", "Aucune action à annuler.")
        return
    if last_action['action'] == 'move':
        original_path = last_action['filepath']
        target_path = last_action['target_path']
        try:
            shutil.move(target_path, original_path)
            current_images.insert(0, (original_path, determine_category(original_path)))
            update_file_list()
            messagebox.showinfo("Annuler", f"Action annulée. L'image a été replacée dans {determine_category(original_path)}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'annuler l'action de déplacement:\n{e}")
            return
    elif last_action['action'] == 'skip':
        original_path = last_action['filepath']
        current_images.insert(0, (original_path, determine_category(original_path)))
        update_file_list()
        messagebox.showinfo("Annuler", "Action de passage annulée.")
    last_action = None
    show_current_image()

def show_temporal_neighbors():
    global img_left, img_right
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

    if index > 0:
        prev_path, prev_num = similar_images[index - 1]
        if abs(current_num - prev_num - PAS_ATTENDU) < TOLERANCE:
            try:
                pil_prev = Image.open(prev_path)
                pil_prev.thumbnail(IMAGE_SIZE)
                img_left = ImageTk.PhotoImage(pil_prev)
                left_image_label.config(image=img_left)
                left_info_label.config(text=f"Nom: {os.path.basename(prev_path)}\nDossier: {os.path.dirname(prev_path)}\nCatégorie: {determine_category(prev_path)}")
            except Exception as e:
                left_image_label.config(text="Erreur d'affichage\nprécédente.")
                left_info_label.config(text="")
        else:
            left_image_label.config(text="Aucune image précédente\nimmédiate.")
            left_info_label.config(text="")
    else:
        left_image_label.config(text="Aucune image précédente\ntrouvée.")
        left_info_label.config(text="")

    if index < len(similar_images) - 1:
        next_path, next_num = similar_images[index + 1]
        if abs(next_num - current_num - PAS_ATTENDU) < TOLERANCE:
            try:
                pil_next = Image.open(next_path)
                pil_next.thumbnail(IMAGE_SIZE)
                img_right = ImageTk.PhotoImage(pil_next)
                right_image_label.config(image=img_right)
                right_info_label.config(text=f"Nom: {os.path.basename(next_path)}\nDossier: {os.path.dirname(next_path)}\nCatégorie: {determine_category(next_path)}")
            except Exception as e:
                right_image_label.config(text="Erreur d'affichage\nsuivante.")
                right_info_label.config(text="")
        else:
            right_image_label.config(text="Aucune image suivante\nimmédiate.")
            right_info_label.config(text="")
    else:
        right_image_label.config(text="Aucune image suivante\ntrouvée.")
        right_info_label.config(text="")

def toggle_neighbors():
    """Fonction appelée lors du clic sur le bouton-toggle pour les images temporelles.
       Si activé, affiche les voisins ; sinon, les efface."""
    if show_neighbors_active.get():
        show_temporal_neighbors()
    else:
        left_image_label.config(image="", text="")
        right_image_label.config(image="", text="")
        left_info_label.config(text="")
        right_info_label.config(text="")

# --- Boutons d'action ---
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

button_move = tk.Button(button_frame, text="Déplacer vers l'autre dossier", command=lambda: process_current_image('move'), width=30)
button_move.grid(row=0, column=0, padx=10)

button_next = tk.Button(button_frame, text="Suivant", command=lambda: process_current_image('skip'), width=15)
button_next.grid(row=0, column=1, padx=10)

neighbors_toggle = tk.Checkbutton(button_frame, text="Voir images temporelles", variable=show_neighbors_active,
                                  onvalue=True, offvalue=False, indicatoron=0, width=30,
                                  command=toggle_neighbors)
neighbors_toggle.grid(row=0, column=2, padx=10)

button_undo = tk.Button(button_frame, text="Annuler l'action", command=undo_last_action, width=20)
button_undo.grid(row=0, column=3, padx=10)

show_current_image()

root.mainloop()
