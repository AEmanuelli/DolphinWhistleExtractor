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

        if is_default_model:
            # Convert predictions and true labels to class indices.
            y_pred_classes = np.argmax(self.current_predictions, axis=1)
            true_labels = np.argmax(self.current_true_labels, axis=1)
        else:
            # Binary case: apply thresholding and flatten arrays.
            y_pred_classes = (self.current_predictions > threshold).astype(int).flatten()
            true_labels = self.current_true_labels.flatten()

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

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelAnalysisGUI(root)
    root.mainloop()