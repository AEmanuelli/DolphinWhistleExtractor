import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle
from datetime import datetime
import time
import threading
from queue import Queue
from typing import Tuple, Dict, Any

def predict_in_batches(model, dataset, total_samples: int, batch_size: int,
                      progress_callback=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fait des prédictions batch par batch en s'assurant que les tailles correspondent.

    Args:
        model: Le modèle Keras
        dataset: Le générateur de données
        total_samples: Nombre total d'échantillons
        batch_size: Taille de batch souhaitée
        progress_callback: Fonction callback pour mise à jour de la progression

    Returns:
        Tuple contenant les prédictions brutes et les vraies étiquettes
    """
    predictions = []
    true_labels = []
    steps = int(np.ceil(total_samples / batch_size))

    for i in range(steps):
        if progress_callback:
            progress_callback(i, steps)

        try:
            batch_x, batch_y = next(dataset)
            # Prédiction sur le batch (garder les prédictions brutes)
            pred = model.predict(batch_x, verbose=0)
            predictions.extend(pred)  # pas  Aplatir les prédictions
            true_labels.extend(batch_y)
        except StopIteration:
            break

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # S'assurer que les dimensions correspondent
    min_len = min(len(predictions), len(true_labels))
    predictions = predictions[:min_len]
    true_labels = true_labels[:min_len]

    return predictions, true_labels

def create_data_generator(directory: str, batch_size: int, class_mode: str, preprocessing_function=None, is_default_model=True):
    """
    Crée un générateur de données avec les bons paramètres, incluant class_mode.
    """
    if is_default_model:
        # Pour le modèle par défaut qui utilise categorical_crossentropy
        class_mode = 'categorical'
        datagen = ImageDataGenerator(
        )

    else:
        # Pour les modèles binaires standards
        class_mode = 'binary'
        datagen = ImageDataGenerator(
            rescale=1/255.,
            preprocessing_function=preprocessing_function
        )

    return datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=class_mode, # Utilisation du class_mode passé en argument
        shuffle=False
    )

class ModelAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Prediction Analysis")
        self.root.geometry("800x700")

        # Analysis control
        self.cancel_analysis = False
        self.analysis_thread = None
        self.progress_queue = Queue()

        # Default paths
        self.default_paths = {
            'model': os.path.join(os.getcwd(), 'models', 'model_vgg.h5'), # Chemin par défaut du modèle
            'test_dir': "/home/emanuelli/Bureau/Dolphins_dataset/test",
            'results': os.path.join(os.getcwd(), 'analysis_results')
        }

        # Create directories if they don't exist
        for path in self.default_paths.values():
            if not path.endswith('.h5'): # Ne pas créer le dossier pour le fichier de modèle par défaut
                os.makedirs(path, exist_ok=True)

        # Predictions cache
        self.cache_file = os.path.join(self.default_paths['results'], 'predictions_cache.pkl')
        self.predictions_cache = self.load_cache()

        # Model paths
        self.model_path = self.default_paths['model'] # Initialiser avec le modèle par défaut
        self.test_dir = None

        # Current analysis data
        self.current_predictions = None
        self.current_true_labels = None
        self.current_filenames = None

        # Setup GUI components
        self.setup_gui()

    def setup_gui(self):
        """Configure all GUI elements"""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.create_input_section()
        self.create_results_section()
        self.create_progress_section()
        self.create_image_viewer()
        self.create_threshold_control()
        # Start progress update checker
        self.check_progress_queue()

    def create_input_section(self):
        """Create input controls with default paths"""
        input_frame = ttk.LabelFrame(self.main_frame, text="Input", padding="5")
        input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Model selection
        ttk.Label(input_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_entry = ttk.Entry(input_frame, width=40)
        self.model_entry.insert(0, self.default_paths['model']) # Afficher le chemin par défaut
        self.model_entry.grid(row=0, column=1, padx=5)
        self.model_btn = ttk.Button(input_frame, text="Browse", command=self.load_model_file)
        self.model_btn.grid(row=0, column=2)

        # Test directory selection
        ttk.Label(input_frame, text="Test Directory:").grid(row=1, column=0, sticky=tk.W)
        self.test_dir_entry = ttk.Entry(input_frame, width=40)
        self.test_dir_entry.insert(0, self.default_paths['test_dir'])
        self.test_dir_entry.grid(row=1, column=1, padx=5)
        self.test_dir_btn = ttk.Button(input_frame, text="Browse", command=self.load_test_dir)
        self.test_dir_btn.grid(row=1, column=2)

        # Batch size entry
        ttk.Label(input_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W)
        self.batch_size_var = tk.StringVar(value="32")
        self.batch_size_entry = ttk.Entry(input_frame, textvariable=self.batch_size_var, width=10)
        self.batch_size_entry.grid(row=2, column=1, sticky=tk.W, padx=5)


        # Analysis type selection
        ttk.Label(input_frame, text="Analysis Type:").grid(row=3, column=0, sticky=tk.W)
        self.analysis_type = ttk.Combobox(input_frame,
                                        values=["False Positives", "False Negatives", "All Misclassified"])
        self.analysis_type.grid(row=3, column=1, sticky=tk.W, padx=5)
        self.analysis_type.set("False Positives")
        self.analysis_type.bind('<<ComboboxSelected>>', self.update_results)

        # Run analysis button
        self.run_btn = ttk.Button(input_frame, text="Run Analysis", command=self.run_analysis)
        self.run_btn.grid(row=4, column=0, columnspan=3, pady=10)

        # Save misclassified paths button
        self.save_paths_btn = ttk.Button(input_frame, text="Save Misclassified Paths", command=self.save_misclassified_paths, state=tk.DISABLED) # Initially disabled
        self.save_paths_btn.grid(row=5, column=0, columnspan=3, pady=10)

    def create_progress_section(self):
        """Create progress bar, status label, and cancel button"""
        progress_frame = ttk.LabelFrame(self.main_frame, text="Progress", padding="5")
        progress_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Progress bar and time remaining
        self.progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress_bar.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        # Status and time label
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.grid(row=1, column=0, padx=5, sticky=tk.W)

        self.time_label = ttk.Label(progress_frame, text="")
        self.time_label.grid(row=1, column=1, padx=5, sticky=tk.E)

        # Cancel button
        self.cancel_button = ttk.Button(progress_frame, text="Cancel", command=self.cancel_analysis_task)
        self.cancel_button.grid(row=2, column=0, columnspan=2, pady=5)
        self.cancel_button['state'] = 'disabled'

    def create_results_section(self):
        """Create section to display metrics and misclassified files"""
        results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="5")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(results_frame, text="Metrics:").grid(row=0, column=0, sticky=tk.NW)
        self.metrics_text = tk.Text(results_frame, height=10, width=60)
        self.metrics_text.grid(row=0, column=1, sticky=(tk.W, tk.E))

        files_frame = ttk.LabelFrame(self.main_frame, text="Misclassified Files", padding="5")
        files_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.file_listbox = tk.Listbox(files_frame, height=10, width=40)
        self.file_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.file_listbox.bind("<Double-Button-1>", self.view_selected_image)

        results_frame.columnconfigure(1, weight=1)
        files_frame.columnconfigure(0, weight=1)

    def create_image_viewer(self):
        """Create image viewer section"""
        image_frame = ttk.LabelFrame(self.main_frame, text="Image Viewer", padding="5")
        image_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.image_label = ttk.Label(image_frame)
        self.image_label.grid(row=0, column=0)

    def create_threshold_control(self):
        """Create controls for adjusting prediction threshold"""
        threshold_frame = ttk.LabelFrame(self.main_frame, text="Prediction Threshold", padding="5")
        threshold_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.threshold_var = tk.DoubleVar(value=0.5)
        self.threshold_scale = ttk.Scale(
            threshold_frame,
            from_=0.0,
            to=1.0,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            command=self.update_results
        )
        self.threshold_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)

        self.threshold_label = ttk.Label(threshold_frame, text="0.5")
        self.threshold_label.grid(row=0, column=1, padx=5)

    def load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Charge le cache des prédictions"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                print(f"Cache loaded successfully from {self.cache_file}")
                return cache_data
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
            if os.path.exists(self.cache_file):
                backup_file = f"{self.cache_file}.backup.{int(time.time())}"
                try:
                    os.rename(self.cache_file, backup_file)
                    print(f"Corrupted cache file backed up to {backup_file}")
                except Exception as be:
                    print(f"Error backing up cache file: {str(be)}")
        return {}

    def save_cache(self, key: str, predictions_data: Dict[str, Any]):
        """Sauvegarde les prédictions brutes dans le cache"""
        self.predictions_cache[key] = {
            'timestamp': datetime.now(),
            'predictions_data': predictions_data
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.predictions_cache, f)

    def get_cache_key(self):
        """Generate unique key for current analysis configuration"""
        return f"{self.model_path}_{self.test_dir}"

    def load_model_file(self):
        """Load model file from dialog and update entry"""
        file_path = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.default_paths['model']), # Initial directory is the model folder
            title="Select Model File",
            filetypes=(("Model files", "*.h5 *.keras"), ("All files", "*.*"))
        )
        if file_path:
            self.model_path = file_path
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, self.model_path)

    def load_test_dir(self):
        """Load test directory from dialog and update entry"""
        dir_path = filedialog.askdirectory(
            initialdir=self.default_paths['test_dir'],
            title="Select Test Directory"
        )
        if dir_path:
            self.test_dir = dir_path
            self.test_dir_entry.delete(0, tk.END)
            self.test_dir_entry.insert(0, self.test_dir)

    def cancel_analysis_task(self):
        """Cancel the running analysis"""
        self.cancel_analysis = True
        self.status_label['text'] = "Cancelling..."
        self.cancel_button['state'] = 'disabled'

    def check_progress_queue(self):
        """Check for progress updates from the analysis thread"""
        try:
            while self.progress_queue.qsize():
                progress = self.progress_queue.get_nowait()
                self.progress_bar['value'] = progress['percent']
                self.status_label['text'] = progress['status']
                if 'time_remaining' in progress:
                    self.time_label['text'] = progress['time_remaining']
        except Queue.Empty:
            pass
        finally:
            # Schedule next check
            self.root.after(100, self.check_progress_queue)


    def update_results(self, *args):
        if self.current_predictions is None:
            return

        threshold = self.threshold_var.get()
        self.threshold_label.config(text=f"{threshold:.3f}")

        is_default_model = os.path.abspath(self.model_path) == os.path.abspath(self.default_paths['model'])
        is_categorical = True
        if is_default_model or is_categorical:
            # Convert predictions and true labels to class indices.
            y_pred_classes = np.argmax(self.current_predictions, axis=1)
            # true_labels = np.argmax(self.current_true_labels, axis=1)
        else:
            # Binary case: apply thresholding and flatten arrays.
            y_pred_classes = (self.current_predictions > threshold).astype(int).flatten()
            
        
        # Calculate and display metrics.
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, classification_report(true_labels, y_pred_classes, zero_division=0))

        # Update the list of misclassified files.
        self.file_listbox.delete(0, tk.END)

        if self.analysis_type.get() == "False Positives":
            indices = np.where((true_labels == 0) & (y_pred_classes == 1))[0]
        elif self.analysis_type.get() == "False Negatives":
            indices = np.where((true_labels == 1) & (y_pred_classes == 0))[0]
        else:  # All Misclassified
            indices = np.where(true_labels != y_pred_classes)[0]

        misclassified_filenames = [self.current_filenames[idx] for idx in indices]
        for filename in misclassified_filenames:
            self.file_listbox.insert(tk.END, filename)

        msg = f"Found {len(misclassified_filenames)} {self.analysis_type.get()} at threshold {threshold:.3f}"
        self.status_label['text'] = msg
        if misclassified_filenames: # Enable save button only if there are misclassified files
            self.save_paths_btn['state'] = tk.NORMAL
        else:
            self.save_paths_btn['state'] = tk.DISABLED

    def run_analysis(self):
        """Lance l'analyse avec vérification du cache"""
        if not self.model_path or not self.test_dir:
            messagebox.showerror("Error", "Please select both model and test directory")
            return

        cache_key = self.get_cache_key()
        if cache_key in self.predictions_cache:
            if messagebox.askyesno("Cached Predictions",
                                 "Predictions found in cache. Would you like to use them?"):
                cached_data = self.predictions_cache[cache_key]['predictions_data']
                self.current_predictions = cached_data['raw_predictions']
                self.current_true_labels = cached_data['true_labels']
                self.current_filenames = cached_data['filenames']
                self.update_results()
                return

        # Reset cancel flag and enable cancel button
        self.cancel_analysis = False
        self.cancel_button['state'] = 'normal'
        self.save_paths_btn['state'] = tk.DISABLED # Disable save paths button at the start of a new analysis

        # Start analysis in separate thread
        self.analysis_thread = threading.Thread(target=self.analysis_task)
        self.analysis_thread.start()

    def view_selected_image(self, event):
        """Display selected image from listbox in image viewer"""
        selected_file_index = self.file_listbox.curselection()
        if selected_file_index:
            filename = self.file_listbox.get(selected_file_index[0])
            image_path = os.path.join(self.test_dir, filename)
            self.display_image(image_path)

    def display_image(self, image_path):
        """Display image in the image viewer"""
        try:
            img = tk.PhotoImage(file=image_path)
            self.image_label.config(image=img)
            self.image_label.image = img  # Keep a reference to prevent garbage collection
        except Exception as e:
            self.image_label.config(text=f"Could not display image: {e}")
            self.image_label.image = None
            print(f"Error displaying image: {e}")

    def analysis_task(self):
        """Tâche d'analyse principale exécutée dans un thread séparé"""
        try:
            start_time = time.time()
            self.progress_queue.put({'percent': 0, 'status': "Chargement du modèle..."})

            model = load_model(self.model_path)
            if self.cancel_analysis:
                raise Exception("Analyse annulée par l'utilisateur")

            self.progress_queue.put({'percent': 20, 'status': "Comptage des échantillons..."})
            total_samples = sum([
                len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                for r, d, files in os.walk(os.path.join(self.test_dir, 'positives'))
            ]) + sum([
                len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                for r, d, files in os.walk(os.path.join(self.test_dir, 'negatives'))
            ])

            if self.cancel_analysis:
                raise Exception("Analyse annulée par l'utilisateur")

            batch_size = min(int(self.batch_size_var.get()), total_samples)

            self.progress_queue.put({'percent': 30, 'status': "Préparation des données..."})

            # Déterminer class_mode en fonction du modèle par défaut
            is_default_model = os.path.abspath(self.model_path) == os.path.abspath(self.default_paths['model'])
            class_mode = 'categorical' if is_default_model else 'binary'
            test_ds = create_data_generator(self.test_dir, batch_size, class_mode, is_default_model=is_default_model) # Passer class_mode

            def update_progress(current_step, total_steps):
                if self.cancel_analysis:
                    raise Exception("Analyse annulée par l'utilisateur")

                progress = (current_step + 1) / total_steps
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / progress if progress > 0 else 0
                time_remaining = max(0, estimated_total_time - elapsed_time)

                time_str = (f"Environ {int(time_remaining/60)}m {int(time_remaining%60)}s restant"
                           if time_remaining > 60 else
                           f"Environ {int(time_remaining)}s restant")

                self.progress_queue.put({
                    'percent': 40 + (50 * progress),
                    'status': f"Traitement du batch {current_step+1}/{total_steps}",
                    'time_remaining': time_str
                })

            self.progress_queue.put({'percent': 40, 'status': "Début des prédictions..."})
            raw_predictions, true_labels = predict_in_batches(
                model, test_ds, total_samples, batch_size, update_progress
            )

            if len(raw_predictions) == 0:
                raise Exception("Aucune prédiction n'a pu être faite")

            # Stockage des prédictions brutes
            predictions_data = {
                'raw_predictions': raw_predictions,
                'true_labels': true_labels,
                'filenames': test_ds.filenames[:len(raw_predictions)]
            }

            # Mise à jour des données courantes
            self.current_predictions = raw_predictions
            self.current_true_labels = true_labels
            self.current_filenames = test_ds.filenames[:len(raw_predictions)]

            # Sauvegarde dans le cache
            cache_key = self.get_cache_key()
            self.save_cache(cache_key, predictions_data)

            # Affichage des résultats avec le seuil par défaut
            self.root.after(0, self.update_results)

            self.progress_queue.put({
                'percent': 100,
                'status': "Analyse terminée !",
                'time_remaining': ""
            })

        except Exception as e:
            if str(e) == "Analyse annulée par l'utilisateur":
                self.progress_queue.put({
                    'percent': 0,
                    'status': "Analyse annulée",
                    'time_remaining': ""
                })
            else:
                self.progress_queue.put({
                    'percent': 0,
                    'status': f"Erreur : {str(e)}",
                    'time_remaining': ""
                })
            print(f"Erreur d'analyse : {str(e)}")

        finally:
            self.cancel_button['state'] = 'disabled'
            self.cancel_analysis = False

    def save_misclassified_paths(self):
        """Sauvegarde les chemins des fichiers mal classés dans un fichier texte."""
        misclassified_filenames = self.file_listbox.get(0, tk.END)
        if not misclassified_filenames:
            messagebox.showinfo("Info", "Aucun fichier mal classé à sauvegarder.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Misclassified File Paths"
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    for filename in misclassified_filenames:
                        full_path = os.path.join(self.test_dir, filename)
                        f.write(full_path + '\n')
                messagebox.showinfo("Success", f"Chemins sauvegardés dans:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Erreur lors de la sauvegarde des chemins:\n{e}")

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ModelAnalysisGUI(root)
#     root.mainloop()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import pickle
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score

class ModelComparisonGUI:
    def __init__(self, root, parent_app):
        self.root = root
        self.parent_app = parent_app  # Référence à l'application principale
        self.models_data = []  # Liste pour stocker les données des modèles
        self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Configuration de la fenêtre
        self.root.title("Model Comparison Tool")
        self.root.geometry("1000x800")
        
        # Création de l'interface utilisateur
        self.setup_gui()
        
    def setup_gui(self):
        """Configure tous les éléments de l'interface graphique"""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.create_models_section()
        self.create_comparison_section()
        self.create_visualization_section()
        
        # Configuration du redimensionnement
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        
    def create_models_section(self):
        """Crée la section pour ajouter et gérer les modèles"""
        models_frame = ttk.LabelFrame(self.main_frame, text="Models", padding="5")
        models_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Liste des modèles
        ttk.Label(models_frame, text="Added Models:").grid(row=0, column=0, sticky=tk.W)
        
        # Treeview pour afficher les modèles
        self.models_tree = ttk.Treeview(models_frame, columns=("Name", "Path"), show="headings", height=5)
        self.models_tree.heading("Name", text="Name")
        self.models_tree.heading("Path", text="Path")
        self.models_tree.column("Name", width=150)
        self.models_tree.column("Path", width=400)
        self.models_tree.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Scrollbar pour la liste des modèles
        scrollbar = ttk.Scrollbar(models_frame, orient=tk.VERTICAL, command=self.models_tree.yview)
        scrollbar.grid(row=1, column=3, sticky=(tk.N, tk.S))
        self.models_tree.configure(yscrollcommand=scrollbar.set)
        
        # Boutons pour gérer les modèles
        button_frame = ttk.Frame(models_frame)
        button_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(button_frame, text="Add Current Model", command=self.add_current_model).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Add New Model", command=self.add_new_model).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_model).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Clear All", command=self.clear_models).grid(row=0, column=3, padx=5)
        
        models_frame.columnconfigure(0, weight=1)
        
    def create_comparison_section(self):
        """Crée la section pour configurer et lancer la comparaison"""
        comparison_frame = ttk.LabelFrame(self.main_frame, text="Comparison Settings", padding="5")
        comparison_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Configuration des métriques
        ttk.Label(comparison_frame, text="Metrics:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        metrics_frame = ttk.Frame(comparison_frame)
        metrics_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.metric_vars = {}
        for i, metric in enumerate(self.metrics):
            var = tk.BooleanVar(value=True)
            self.metric_vars[metric] = var
            ttk.Checkbutton(metrics_frame, text=metric.capitalize(), variable=var).grid(row=0, column=i, padx=5)
        
        # Configuration du seuil
        ttk.Label(comparison_frame, text="Threshold:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        threshold_frame = ttk.Frame(comparison_frame)
        threshold_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.threshold_scale = ttk.Scale(
            threshold_frame,
            from_=0.0,
            to=1.0,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        self.threshold_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)
        
        self.threshold_label = ttk.Label(threshold_frame, text="0.5")
        self.threshold_label.grid(row=0, column=1, padx=5)
        
        self.threshold_scale.configure(command=self.update_threshold_label)
        
        # Bouton pour lancer la comparaison
        ttk.Button(comparison_frame, text="Compare Models", command=self.compare_models).grid(row=2, column=0, columnspan=2, pady=10)
        
        comparison_frame.columnconfigure(1, weight=1)
        
    def create_visualization_section(self):
        """Crée la section pour visualiser les résultats de la comparaison"""
        visualization_frame = ttk.LabelFrame(self.main_frame, text="Comparison Results", padding="5")
        visualization_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Interface à onglets pour différents types de visualisations
        self.notebook = ttk.Notebook(visualization_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Onglet pour les graphiques de métriques
        self.metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_tab, text="Metrics Comparison")
        
        # Onglet pour les courbes ROC
        self.roc_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.roc_tab, text="ROC Curves")
        
        # Onglet pour le tableau de résultats détaillés
        self.table_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.table_tab, text="Detailed Results")
        
        # Configuration pour les graphiques
        self.setup_metrics_plot()
        self.setup_roc_plot()
        self.setup_results_table()
        
        visualization_frame.columnconfigure(0, weight=1)
        visualization_frame.rowconfigure(0, weight=1)
        
    def setup_metrics_plot(self):
        """Configure le graphique de comparaison des métriques"""
        self.metrics_fig = Figure(figsize=(8, 5), dpi=100)
        self.metrics_ax = self.metrics_fig.add_subplot(111)
        
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=self.metrics_tab)
        self.metrics_canvas.draw()
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_roc_plot(self):
        """Configure le graphique des courbes ROC"""
        self.roc_fig = Figure(figsize=(8, 5), dpi=100)
        self.roc_ax = self.roc_fig.add_subplot(111)
        
        self.roc_canvas = FigureCanvasTkAgg(self.roc_fig, master=self.roc_tab)
        self.roc_canvas.draw()
        self.roc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_results_table(self):
        """Configure le tableau des résultats détaillés"""
        columns = ("Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC")
        
        self.results_tree = ttk.Treeview(self.table_tab, columns=columns, show="headings", height=10)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100, anchor=tk.CENTER)
        
        self.results_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def update_threshold_label(self, *args):
        """Met à jour l'étiquette du seuil"""
        threshold = self.threshold_var.get()
        self.threshold_label.config(text=f"{threshold:.3f}")
        
    def add_current_model(self):
        """Ajoute le modèle actuellement chargé dans l'application principale"""
        if not hasattr(self.parent_app, 'model_path') or not self.parent_app.model_path:
            messagebox.showerror("Error", "No model currently loaded in the main application.")
            return
            
        # Vérifier si le modèle est déjà dans la liste
        for item_id in self.models_tree.get_children():
            if self.models_tree.item(item_id, 'values')[1] == self.parent_app.model_path:
                messagebox.showinfo("Info", "This model is already in the comparison list.")
                return
                
        # Extraire le nom du modèle à partir du chemin
        model_name = os.path.basename(self.parent_app.model_path)
        
        # Ajouter le modèle à la liste
        self.models_tree.insert('', tk.END, values=(model_name, self.parent_app.model_path))
        
        # Ajouter les données du modèle si des prédictions existent
        if hasattr(self.parent_app, 'current_predictions') and self.parent_app.current_predictions is not None:
            self.models_data.append({
                'name': model_name,
                'path': self.parent_app.model_path,
                'predictions': self.parent_app.current_predictions,
                'true_labels': self.parent_app.current_true_labels,
                'filenames': self.parent_app.current_filenames
            })
            
            messagebox.showinfo("Success", f"Added model '{model_name}' with existing predictions.")
        else:
            self.models_data.append({
                'name': model_name,
                'path': self.parent_app.model_path,
                'predictions': None,
                'true_labels': None,
                'filenames': None
            })
            
            messagebox.showinfo("Info", f"Added model '{model_name}' without predictions data.")
        
    def add_new_model(self):
        """Ajoute un nouveau modèle à partir d'un fichier"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=(("Model files", "*.h5 *.keras"), ("All files", "*.*"))
        )
        
        if not file_path:
            return
            
        # Vérifier si le modèle est déjà dans la liste
        for item_id in self.models_tree.get_children():
            if self.models_tree.item(item_id, 'values')[1] == file_path:
                messagebox.showinfo("Info", "This model is already in the comparison list.")
                return
                
        # Extraire le nom du modèle à partir du chemin
        model_name = os.path.basename(file_path)
        
        # Ajouter le modèle à la liste
        self.models_tree.insert('', tk.END, values=(model_name, file_path))
        
        # Ajouter les données du modèle (sans prédictions pour l'instant)
        self.models_data.append({
            'name': model_name,
            'path': file_path,
            'predictions': None,
            'true_labels': None,
            'filenames': None
        })
        
    def remove_model(self):
        """Supprime le modèle sélectionné de la liste"""
        selected_items = self.models_tree.selection()
        
        if not selected_items:
            messagebox.showinfo("Info", "Please select a model to remove.")
            return
            
        for item_id in selected_items:
            model_path = self.models_tree.item(item_id, 'values')[1]
            
            # Supprimer le modèle de la liste des données
            self.models_data = [data for data in self.models_data if data['path'] != model_path]
            
            # Supprimer le modèle de la liste d'affichage
            self.models_tree.delete(item_id)
        
    def clear_models(self):
        """Supprime tous les modèles de la liste"""
        if not self.models_tree.get_children():
            return
            
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all models?"):
            # Supprimer tous les modèles de la liste d'affichage
            for item_id in self.models_tree.get_children():
                self.models_tree.delete(item_id)
                
            # Vider la liste des données
            self.models_data = []
        
    def compare_models(self):
        """Lance la comparaison des modèles"""
        if not self.models_data:
            messagebox.showinfo("Info", "Please add models to compare.")
            return
            
        # Vérifier que tous les modèles ont des prédictions
        models_without_predictions = [data['name'] for data in self.models_data if data['predictions'] is None]
        
        if models_without_predictions:
            # Proposer de charger les prédictions depuis le cache
            if messagebox.askyesno("Missing Predictions", 
                                  f"The following models don't have predictions: {', '.join(models_without_predictions)}.\n\n"
                                  f"Would you like to try loading predictions from cache?"):
                self.load_predictions_from_cache()
            else:
                return
                
        # Vérifier à nouveau après le chargement du cache
        models_without_predictions = [data['name'] for data in self.models_data if data['predictions'] is None]
        
        if models_without_predictions:
            # Proposer de générer de nouvelles prédictions
            if messagebox.askyesno("Missing Predictions", 
                                  f"Still missing predictions for: {', '.join(models_without_predictions)}.\n\n"
                                  f"Would you like to run predictions for these models?\n"
                                  f"(This might take a while depending on your test dataset)"):
                self.run_predictions_for_models(models_without_predictions)
            else:
                return
                
        # Vérifier une dernière fois
        models_without_predictions = [data['name'] for data in self.models_data if data['predictions'] is None]
        
        if models_without_predictions:
            messagebox.showerror("Error", f"Could not get predictions for: {', '.join(models_without_predictions)}.")
            return
            
        # Calculer les métriques pour tous les modèles
        threshold = self.threshold_var.get()
        results = self.calculate_metrics(threshold)
        
        # Mettre à jour les visualisations
        self.update_metrics_plot(results)
        self.update_roc_plot()
        self.update_results_table(results)
        
    def load_predictions_from_cache(self):
        """Charge les prédictions à partir du cache pour les modèles qui n'en ont pas"""
        if not hasattr(self.parent_app, 'predictions_cache'):
            messagebox.showinfo("Info", "Cache not available in the main application.")
            return
            
        cache = self.parent_app.predictions_cache
        test_dir = self.parent_app.test_dir_entry.get()
        
        for i, model_data in enumerate(self.models_data):
            if model_data['predictions'] is not None:
                continue
                
            # Générer la clé de cache
            cache_key = f"{model_data['path']}_{test_dir}"
            
            if cache_key in cache:
                cached_data = cache[cache_key]['predictions_data']
                self.models_data[i]['predictions'] = cached_data['raw_predictions']
                self.models_data[i]['true_labels'] = cached_data['true_labels']
                self.models_data[i]['filenames'] = cached_data['filenames']
                
                print(f"Loaded predictions from cache for model '{model_data['name']}'")
        
    def run_predictions_for_models(self, model_names):
        """Exécute les prédictions pour les modèles spécifiés"""
        # Cette fonction devrait exécuter les prédictions pour chaque modèle qui en a besoin
        # Elle utiliserait predict_in_batches de l'application principale
        
        messagebox.showinfo("Not Implemented", 
                          "Automatic prediction generation is not yet implemented.\n\n"
                          "Please run predictions for each model in the main application first.")
        
    def calculate_metrics(self, threshold):
        """Calcule les métriques pour tous les modèles"""
        results = []
        
        for model_data in self.models_data:
            if model_data['predictions'] is None or model_data['true_labels'] is None:
                continue
                
            # Déterminer si le modèle utilise une classification catégorielle ou binaire
            is_categorical = len(model_data['predictions'].shape) > 1 and model_data['predictions'].shape[1] > 1
            
            if is_categorical:
                # Classification multinomiale
                y_pred_classes = np.argmax(model_data['predictions'], axis=1)
                y_true = np.argmax(model_data['true_labels'], axis=1) if len(model_data['true_labels'].shape) > 1 else model_data['true_labels']
                
                # Calculer les métriques
                accuracy = accuracy_score(y_true, y_pred_classes)
                precision = precision_score(y_true, y_pred_classes, average='macro', zero_division=0)
                recall = recall_score(y_true, y_pred_classes, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred_classes, average='macro', zero_division=0)
                
                # Pour AUC en multiclasse, utilisez ROC AUC score avec one-vs-rest
                try:
                    # Convertir les prédictions brutes en probabilités pour chaque classe
                    auc_value = roc_auc_score(y_true, model_data['predictions'], multi_class='ovr', average='macro')
                except ValueError:
                    auc_value = np.nan
                
            else:
                # Classification binaire
                predictions_flat = model_data['predictions'].flatten()
                y_pred_classes = (predictions_flat > threshold).astype(int)
                y_true = model_data['true_labels'].flatten()
                
                # Calculer les métriques
                accuracy = accuracy_score(y_true, y_pred_classes)
                precision = precision_score(y_true, y_pred_classes, zero_division=0)
                recall = recall_score(y_true, y_pred_classes, zero_division=0)
                f1 = f1_score(y_true, y_pred_classes, zero_division=0)
                
                # Calculer AUC
                try:
                    fpr, tpr, _ = roc_curve(y_true, predictions_flat)
                    auc_value = auc(fpr, tpr)
                except ValueError:
                    auc_value = np.nan
            
            # Stocker les résultats
            results.append({
                'name': model_data['name'],
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc_value,
                'is_categorical': is_categorical,
                'predictions': model_data['predictions'],
                'true_labels': model_data['true_labels']
            })
            
        return results
        
    def update_metrics_plot(self, results):
        """Met à jour le graphique de comparaison des métriques"""
        # Réinitialiser le graphique
        self.metrics_ax.clear()
        
        # Vérifier quelles métriques sont sélectionnées
        selected_metrics = [metric for metric, var in self.metric_vars.items() if var.get()]
        
        if not selected_metrics:
            messagebox.showinfo("Info", "Please select at least one metric to display.")
            return
            
        if not results:
            return
            
        # Préparer les données pour le graphique
        model_names = [result['name'] for result in results]
        x = np.arange(len(model_names))
        width = 0.8 / len(selected_metrics)
        
        # Couleurs pour les différentes métriques
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Tracer les barres pour chaque métrique
        for i, metric in enumerate(selected_metrics):
            values = [result[metric] for result in results]
            self.metrics_ax.bar(x + i * width - (len(selected_metrics) - 1) * width / 2, 
                              values, 
                              width, 
                              label=metric.capitalize(),
                              color=colors[i % len(colors)])
        
        # Configurer le graphique
        self.metrics_ax.set_xlabel('Models')
        self.metrics_ax.set_ylabel('Score')
        self.metrics_ax.set_title('Model Comparison - Performance Metrics')
        self.metrics_ax.set_xticks(x)
        self.metrics_ax.set_xticklabels(model_names, rotation=45, ha='right')
        self.metrics_ax.legend()
        self.metrics_ax.grid(axis='y', linestyle='--', alpha=0.7)
        self.metrics_ax.set_ylim(0, 1)
        
        # Ajuster la mise en page
        self.metrics_fig.tight_layout()
        
        # Redessiner le canvas
        self.metrics_canvas.draw()
        
    def update_roc_plot(self):
        """Met à jour le graphique des courbes ROC"""
        # Réinitialiser le graphique
        self.roc_ax.clear()
        
        # Tracer les courbes ROC pour chaque modèle binaire
        for model_data in self.models_data:
            if model_data['predictions'] is None or model_data['true_labels'] is None:
                continue
                
            # Vérifier si c'est un modèle binaire
            is_categorical = len(model_data['predictions'].shape) > 1 and model_data['predictions'].shape[1] > 1
            
            if not is_categorical:
                # Classification binaire
                predictions_flat = model_data['predictions'].flatten()
                y_true = model_data['true_labels'].flatten()
                
                try:
                    # Calculer la courbe ROC
                    fpr, tpr, _ = roc_curve(y_true, predictions_flat)
                    roc_auc = auc(fpr, tpr)
                    
                    # Tracer la courbe
                    self.roc_ax.plot(fpr, tpr, lw=2, 
                                   label=f'{model_data["name"]} (AUC = {roc_auc:.3f})')
                except ValueError as e:
                    print(f"Error calculating ROC curve for {model_data['name']}: {e}")
        
        # Ajouter la ligne diagonale (aléatoire)
        self.roc_ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Configurer le graphique
        self.roc_ax.set_xlim([0.0, 1.0])
        self.roc_ax.set_ylim([0.0, 1.05])
        self.roc_ax.set_xlabel('False Positive Rate')
        self.roc_ax.set_ylabel('True Positive Rate')
        self.roc_ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        self.roc_ax.legend(loc="lower right")
        self.roc_ax.grid(linestyle='--', alpha=0.7)
        
        # Ajuster la mise en page
        self.roc_fig.tight_layout()
        
        # Redessiner le canvas
        self.roc_canvas.draw()
        
    def update_results_table(self, results):
        """Met à jour le tableau des résultats détaillés"""
        # Effacer les entrées existantes
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        # Ajouter les nouvelles entrées
        for result in results:
            values = (
                result['name'],
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1']:.4f}",
                f"{result['auc']:.4f}" if not np.isnan(result['auc']) else "N/A"
            )
            self.results_tree.insert('', tk.END, values=values)

def add_model_comparison_to_main_app(app):
    """Ajoute la fonctionnalité de comparaison de modèles à l'application principale"""
    def open_comparison_window():
        comparison_window = tk.Toplevel(app.root)
        ModelComparisonGUI(comparison_window, app)
    
    # Ajouter un bouton dans l'interface principale
    compare_btn = ttk.Button(app.main_frame, text="Compare Models", command=open_comparison_window)
    compare_btn.grid(row=0, column=2, padx=10, pady=5, sticky=tk.N)
    
# Modification de la classe principale pour intégrer la fonctionnalité
def modify_main_class():
    # Sauvegarde de la méthode d'initialisation originale
    original_init = ModelAnalysisGUI.__init__
    
    # Définir la nouvelle méthode d'initialisation
    def new_init(self, root):
        # Appeler l'initialisation originale
        original_init(self, root)
        
        # Ajouter la fonctionnalité de comparaison
        add_model_comparison_to_main_app(self)
    
    # Remplacer la méthode d'initialisation
    ModelAnalysisGUI.__init__ = new_init

if __name__ == "__main__":
    # Modifier la classe principale
    modify_main_class()
    
    # Démarrer l'application normalement
    root = tk.Tk()
    app = ModelAnalysisGUI(root)
    root.mainloop()