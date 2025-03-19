# Standard library imports
import os
import pickle
import threading
import time
from datetime import datetime
from glob import glob
from queue import Queue
from typing import Dict, Any, Tuple

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Data analysis and visualization imports
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, auc, roc_auc_score, classification_report
)

# Deep learning imports
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import os

# Import necessary functions from the main app


def predict_in_batches(model, dataset, total_samples: int, batch_size: int, progress_callback=None):
    """Efficiently predicts in batches with TF optimizations."""
    steps = int(np.ceil(total_samples / batch_size))
    predictions = []
    true_labels = []
    
    # Create optimized prediction function with graph execution
    @tf.function(reduce_retracing=True)
    def predict_optimized(x):
        return model(x, training=False)
    
    # Use prefetching to overlap preprocessing and model execution
    if isinstance(dataset, tf.data.Dataset):
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    for i in range(steps):
        if progress_callback:
            progress_callback(i, steps)
        try:
            batch_x, batch_y = next(dataset)
            # Use compiled function for prediction
            batch_pred = predict_optimized(batch_x).numpy()
            predictions.append(batch_pred)
            true_labels.append(batch_y)
            
            # Force release GPU memory if needed
            if hasattr(tf, 'keras'):
                tf.keras.backend.clear_session()
        except StopIteration:
            break
    
    # Efficiently combine results without unnecessary copies
    if predictions:
        predictions = np.vstack(predictions)
        true_labels = np.vstack(true_labels) if isinstance(true_labels[0], np.ndarray) else np.concatenate(true_labels)
    else:
        predictions = np.array([])
        true_labels = np.array([])
    
    # Ensure matching lengths
    min_len = min(len(predictions), len(true_labels)) if predictions.size and true_labels.size else 0
    return predictions[:min_len], true_labels[:min_len]

def create_data_generator(directory: str, batch_size: int, preprocessing_function=preprocess_input, model_categorical=False, model_binary=False, rescale = False) -> ImageDataGenerator:
    """Creates data generator with class_mode based on model type."""
    class_mode = 'categorical' if model_categorical else 'binary' if model_binary else None
    if class_mode is None:
        raise ValueError("Model type (categorical or binary) must be specified.")
    if rescale == False : 
        datagen = ImageDataGenerator()
    else : 
        datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(directory, target_size=(224, 224), batch_size=batch_size, class_mode=class_mode, shuffle=False)

class ModelAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Prediction Analysis")
        self.root.geometry("1000x800")

        self.cancel_analysis = False
        self.analysis_thread = None
        self.progress_queue = Queue()
        self.model_categorical = False
        self.model_binary = False

        self.default_paths = {
            'model': 'models/model_vgg.h5',
            'test_dir': "/home/emanuelli/Bureau/Dolphins_dataset/test",
            'results': 'analysis_results'
        }
        for path in self.default_paths.values():
            if not path.endswith('.h5'):
                os.makedirs(path, exist_ok=True)

        model_name = os.path.splitext(os.path.basename(self.default_paths['model']))[0]
        self.cache_file = os.path.join(self.default_paths['results'], f'{model_name}_predictions_cache.pkl')
        self.predictions_cache = self.load_cache()

        self.model_path = self.default_paths['model']
        self.test_dir = self.default_paths['test_dir']

        self.current_predictions = None
        self.current_true_labels = None
        self.current_filenames = None

        self.setup_gui()

    def setup_gui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        input_frame = ttk.LabelFrame(self.main_frame, text="Input", padding="5")
        input_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        def create_labeled_entry_browse(frame, text, row, default_path, browse_command):
            ttk.Label(frame, text=text).grid(row=row, column=0, sticky="w")
            entry = ttk.Entry(frame, width=40)
            entry.insert(0, default_path)
            entry.grid(row=row, column=1, padx=5, sticky="ew")
            ttk.Button(frame, text="Browse", command=browse_command).grid(row=row, column=2)
            return entry

        self.model_entry = create_labeled_entry_browse(input_frame, "Select Model:", 0, self.default_paths['model'], self.load_model_file)
        self.test_dir_entry = create_labeled_entry_browse(input_frame, "Test Directory:", 1, self.default_paths['test_dir'], self.load_test_dir)

        ttk.Label(input_frame, text="Batch Size:").grid(row=2, column=0, sticky="w")
        self.batch_size_var = tk.StringVar(value="64")
        ttk.Entry(input_frame, textvariable=self.batch_size_var, width=10).grid(row=2, column=1, sticky="w", padx=5)

        ttk.Label(input_frame, text="Analysis Type:").grid(row=3, column=0, sticky="w")
        self.analysis_type = ttk.Combobox(input_frame, values=["False Positives", "False Negatives", "All Misclassified"])
        self.analysis_type.grid(row=3, column=1, sticky="w", padx=5)
        self.analysis_type.set("False Positives")
        self.analysis_type.bind('<<ComboboxSelected>>', self.update_results)

        ttk.Button(input_frame, text="Run Analysis", command=self.run_analysis).grid(row=4, column=0, columnspan=3, pady=10)
        self.save_paths_btn = ttk.Button(input_frame, text="Save Misclassified Paths", command=self.save_misclassified_paths, state=tk.DISABLED)
        self.save_paths_btn.grid(row=5, column=0, columnspan=3, pady=10)
        input_frame.columnconfigure(1, weight=1)

        progress_frame = ttk.LabelFrame(self.main_frame, text="Progress", padding="5")
        progress_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress_bar.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.grid(row=1, column=0, padx=5, sticky="w")
        self.time_label = ttk.Label(progress_frame, text="")
        self.time_label.grid(row=1, column=1, padx=5, sticky="e")
        self.cancel_button = ttk.Button(progress_frame, text="Cancel", command=self.cancel_analysis_task, state='disabled')
        self.cancel_button.grid(row=2, column=0, columnspan=2, pady=5)
        progress_frame.columnconfigure(0, weight=1)

        results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="5")
        results_frame.grid(row=1, column=0, sticky="new", pady=5)
        ttk.Label(results_frame, text="Metrics:").grid(row=0, column=0, sticky="nw")
        self.metrics_text = tk.Text(results_frame, height=10, width=60)
        self.metrics_text.grid(row=0, column=1, sticky="ew")
        results_frame.columnconfigure(1, weight=1)

        files_frame = ttk.LabelFrame(self.main_frame, text="Misclassified Files", padding="5")
        files_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.file_listbox = tk.Listbox(files_frame, height=10, width=40)
        self.file_listbox.grid(row=0, column=0, sticky="nsew")
        self.file_listbox.bind("<Double-Button-1>", self.view_selected_image)
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.columnconfigure(1, weight=1)


        image_frame = ttk.LabelFrame(self.main_frame, text="Image Viewer", padding="5")
        image_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        self.image_label = ttk.Label(image_frame)
        self.image_label.grid(row=0, column=0)
        image_frame.columnconfigure(0, weight=1)

        threshold_frame = ttk.LabelFrame(self.main_frame, text="Prediction Threshold", padding="5")
        threshold_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.threshold_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, variable=self.threshold_var, orient=tk.HORIZONTAL, command=self.update_results)
        self.threshold_scale.grid(row=0, column=0, sticky="ew", padx=5)
        self.threshold_label = ttk.Label(threshold_frame, text="0.5")
        self.threshold_label.grid(row=0, column=1, padx=5)
        threshold_frame.columnconfigure(0, weight=1)

        self.check_progress_queue()

    def load_cache(self) -> Dict[str, Dict[str, Any]]:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Cache load error: {e}")
            if os.path.exists(self.cache_file):
                try:
                    os.rename(self.cache_file, f"{self.cache_file}.backup.{int(time.time())}")
                except Exception as be:
                    print(f"Cache backup error: {be}")
        return {}

    def save_cache(self, key: str, predictions_data: Dict[str, Any]):
        self.predictions_cache[key] = {'timestamp': datetime.now(), 'predictions_data': predictions_data}
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.predictions_cache, f)
        except Exception as e:
            print(f"Cache save error: {e}")

    def get_cache_key(self):
        return f"{self.model_path}_{self.test_dir}"

    def load_model_file(self):
        file_path = filedialog.askopenfilename(initialdir=os.path.dirname(self.default_paths['model']), title="Select Model File", filetypes=(("Model files", "*.h5 *.keras"), ("All files", "*.*")))
        if file_path:
            self.model_path = file_path
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, file_path)

    def load_test_dir(self):
        dir_path = filedialog.askdirectory(initialdir=self.default_paths['test_dir'], title="Select Test Directory")
        if dir_path:
            self.test_dir = dir_path
            self.test_dir_entry.delete(0, tk.END)
            self.test_dir_entry.insert(0, dir_path)

    def cancel_analysis_task(self):
        self.cancel_analysis = True
        self.status_label['text'] = "Cancelling..."
        self.cancel_button['state'] = 'disabled'

    def check_progress_queue(self):
        try:
            while not self.progress_queue.empty():
                progress = self.progress_queue.get_nowait()
                self.progress_bar['value'] = progress['percent']
                self.status_label['text'] = progress['status']
                self.time_label['text'] = progress.get('time_remaining', "")
        except Queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_progress_queue)

    def update_results(self, *args):
        if self.current_predictions is None:
            return

        threshold = self.threshold_var.get()
        self.threshold_label.config(text=f"{threshold:.3f}")

        if self.model_categorical:
            y_pred_classes = np.argmax(self.current_predictions, axis=1)
            true_labels = np.argmax(self.current_true_labels, axis=1)
        elif self.model_binary:
            y_pred_classes = (self.current_predictions > threshold).astype(int).flatten()
            true_labels = self.current_true_labels.flatten()
        else: # Default to categorical if model type not determined
            y_pred_classes = np.argmax(self.current_predictions, axis=1)
            true_labels = np.argmax(self.current_true_labels, axis=1)

        report = classification_report(true_labels, y_pred_classes, zero_division=0, output_dict=False)
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, report)

        self.file_listbox.delete(0, tk.END)
        analysis_type = self.analysis_type.get()
        if analysis_type == "False Positives":
            indices = np.where((true_labels == 0) & (y_pred_classes == 1))[0]
        elif analysis_type == "False Negatives":
            indices = np.where((true_labels == 1) & (y_pred_classes == 0))[0]
        else:  # All Misclassified
            indices = np.where(true_labels != y_pred_classes)[0]

        misclassified_filenames = [self.current_filenames[idx] for idx in indices]
        for filename in misclassified_filenames:
            self.file_listbox.insert(tk.END, filename)

        msg = f"Found {len(misclassified_filenames)} {analysis_type} at threshold {threshold:.3f}"
        self.status_label['text'] = msg
        self.save_paths_btn['state'] = tk.NORMAL if misclassified_filenames else tk.DISABLED

    def run_analysis(self):
        if not self.model_path or not self.test_dir:
            messagebox.showerror("Error", "Select model and test directory")
            return

        cache_key = self.get_cache_key()
        if cache_key in self.predictions_cache and messagebox.askyesno("Cached Predictions", "Use cached predictions?"):
            cached_data = self.predictions_cache[cache_key]['predictions_data']
            self.current_predictions = cached_data['raw_predictions']
            self.current_true_labels = cached_data['true_labels']
            self.current_filenames = cached_data['filenames']
            self.update_results()
            return

        self.cancel_analysis = False
        self.cancel_button['state'] = 'normal'
        self.save_paths_btn['state'] = tk.DISABLED

        self.analysis_thread = threading.Thread(target=self.analysis_task)
        self.analysis_thread.start()

    def view_selected_image(self, event):
        selected_file_index = self.file_listbox.curselection()
        if selected_file_index:
            image_path = os.path.join(self.test_dir, self.file_listbox.get(selected_file_index[0]))
            try:
                img = tk.PhotoImage(file=image_path)
                self.image_label.config(image=img)
                self.image_label.image = img
            except Exception as e:
                self.image_label.config(text=f"Image error: {e}")
                self.image_label.image = None

    def analysis_task(self):
        start_time = time.time()
        try:
            # Performance optimization configurations
            
            # Limit TensorFlow memory growth to prevent using all GPU memory
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU config error: {e}")

            # Set inter/intra parallelism threads for CPU efficiency
            tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
            tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
            
            self.progress_queue.put({'percent': 0, 'status': "Loading model..."})
            model = load_model(self.model_path, compile=False)  # Skip compilation for faster loading
            model.trainable = False
            
            # Create optimized inference function with graph execution
            @tf.function(reduce_retracing=True)
            def predict_optimized(images):
                return model(images, training=False)
            
            # Test model type with small batch
            dummy_input = np.random.rand(1, 224, 224, 3).astype('float32')
            dummy_pred = model.predict(dummy_input, verbose=0)
            self.model_categorical = len(dummy_pred.shape) > 1 and dummy_pred.shape[1] > 1
            self.model_binary = not self.model_categorical

            if self.cancel_analysis: raise Exception("Analysis cancelled")

            self.progress_queue.put({'percent': 20, 'status': "Counting samples..."})
            total_samples = len(glob(os.path.join(self.test_dir, '*/*.*g')))
            if self.cancel_analysis: raise Exception("Analysis cancelled")

            # Determine optimal batch size based on available memory and cores
            batch_size = min(int(self.batch_size_var.get()), total_samples)
            # For most GPUs, powers of 2 tend to be more efficient
            batch_size = max(32, 2**int(np.log2(batch_size)))
            
            self.progress_queue.put({'percent': 30, 'status': "Preparing data..."})
            
            # Create optimized data pipeline with prefetching
            test_ds = create_data_generator(
                self.test_dir, batch_size, 
                model_categorical=self.model_categorical, 
                model_binary=self.model_binary
            )
            
            # Replace standard predict function with faster implementation
            def fast_predict(model, batch_x):
                return predict_optimized(batch_x).numpy()
            
            def update_progress(current_step, total_steps):
                if self.cancel_analysis: raise Exception("Analysis cancelled")
                progress = (current_step + 1) / total_steps
                elapsed_time = time.time() - start_time
                time_remaining = max(0, (elapsed_time / progress) - elapsed_time if progress else 0)
                time_str = f"~{int(time_remaining // 60)}m {int(time_remaining % 60)}s left" if time_remaining > 60 else f"~{int(time_remaining)}s left"
                self.progress_queue.put({'percent': 40 + (50 * progress), 'status': f"Processing batch {current_step+1}/{total_steps}", 'time_remaining': time_str})
            
            self.progress_queue.put({'percent': 40, 'status': "Predicting..."})
            start_time = time.time()
            
            # Modified prediction function to use optimized prediction
            steps = int(np.ceil(total_samples / batch_size))
            predictions = []
            true_labels = []
            
            for i in range(steps):
                if self.cancel_analysis: raise Exception("Analysis cancelled")
                update_progress(i, steps)
                try:
                    batch_x, batch_y = next(test_ds)
                    # Use optimized prediction function 
                    batch_predictions = fast_predict(model, batch_x)
                    predictions.append(batch_predictions)
                    true_labels.append(batch_y)
                except StopIteration:
                    break
            
            raw_predictions = np.concatenate(predictions, axis=0) if predictions else np.array([])
            true_labels = np.concatenate(true_labels, axis=0) if true_labels else np.array([])
            
            # Ensure consistent lengths
            min_len = min(len(raw_predictions), len(true_labels)) if raw_predictions.size and true_labels.size else 0
            raw_predictions = raw_predictions[:min_len]
            true_labels = true_labels[:min_len]
            
            if not raw_predictions.size: raise Exception("No predictions made")
            
            predictions_data = {'raw_predictions': raw_predictions, 'true_labels': true_labels, 'filenames': test_ds.filenames[:min_len]}
            self.current_predictions, self.current_true_labels, self.current_filenames = raw_predictions, true_labels, test_ds.filenames[:min_len]
            
            inference_time = time.time() - start_time
            self.save_cache(self.get_cache_key(), predictions_data)
            self.root.after(0, self.update_results)
            self.progress_queue.put({'percent': 100, 'status': f"Analysis finished! Inference time: {inference_time:.2f}s", 'time_remaining': ""})

        except Exception as e:
            self.progress_queue.put({'percent': 0, 'status': f"Error: {e}", 'time_remaining': ""})
            print(f"Analysis error: {e}")
        finally:
            self.cancel_button['state'] = 'disabled'
            self.cancel_analysis = False

    def save_misclassified_paths(self):
        filenames = self.file_listbox.get(0, tk.END)
        if not filenames:
            messagebox.showinfo("Info", "No misclassified files to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")], title="Save Misclassified File Paths")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    for filename in filenames:
                        f.write(os.path.join(self.test_dir, filename) + '\n')
                messagebox.showinfo("Success", f"Paths saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving paths: {e}")


class ModelComparisonGUI:
    def __init__(self, root, parent_app):
        self.root = root
        self.parent_app = parent_app
        self.models_data = []
        self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        self.test_dir = parent_app.test_dir if hasattr(parent_app, 'test_dir') else ""
        self.cancel_analysis = False
        self.analysis_thread = None
        self.progress_queue = Queue()

        self.root.title("Model Comparison Tool")
        self.root.geometry("1000x800")
        self.setup_gui()
        self.check_progress_queue()

    def setup_gui(self):
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Models section
        models_frame = ttk.LabelFrame(self.main_frame, text="Models", padding="5")
        models_frame.grid(row=0, column=0, sticky="ew", pady=5)
        ttk.Label(models_frame, text="Added Models:").grid(row=0, column=0, sticky="w")

        self.models_tree = ttk.Treeview(models_frame, columns=("Name", "Path", "Status"), show="headings", height=5)
        self.models_tree.heading("Name", text="Name")
        self.models_tree.heading("Path", text="Path")
        self.models_tree.heading("Status", text="Status")
        self.models_tree.column("Name", width=150)
        self.models_tree.column("Path", width=350)
        self.models_tree.column("Status", width=100)
        self.models_tree.grid(row=1, column=0, columnspan=4, sticky="ew", pady=5)
        scrollbar = ttk.Scrollbar(models_frame, orient=tk.VERTICAL, command=self.models_tree.yview)
        scrollbar.grid(row=1, column=4, sticky="ns")
        self.models_tree.configure(yscrollcommand=scrollbar.set)

        button_frame = ttk.Frame(models_frame)
        button_frame.grid(row=2, column=0, columnspan=4, sticky="ew", pady=5)
        ttk.Button(button_frame, text="Add Current Model", command=self.add_current_model).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Add New Model", command=self.add_new_model).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_model).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Clear All", command=self.clear_models).grid(row=0, column=3, padx=5)
        models_frame.columnconfigure(0, weight=1)

        # Test directory section
        test_dir_frame = ttk.LabelFrame(self.main_frame, text="Test Directory", padding="5")
        test_dir_frame.grid(row=1, column=0, sticky="ew", pady=5)
        ttk.Label(test_dir_frame, text="Test Directory:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.test_dir_entry = ttk.Entry(test_dir_frame, width=60)
        self.test_dir_entry.insert(0, self.test_dir)
        self.test_dir_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(test_dir_frame, text="Browse", command=self.browse_test_dir).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(test_dir_frame, text="Batch Size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(test_dir_frame, textvariable=self.batch_size_var, width=10).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        ttk.Button(test_dir_frame, text="Run All Models", command=self.run_all_models).grid(row=2, column=0, columnspan=3, pady=10)
        test_dir_frame.columnconfigure(1, weight=1)

        # Progress bar section
        progress_frame = ttk.LabelFrame(self.main_frame, text="Progress", padding="5")
        progress_frame.grid(row=2, column=0, sticky="ew", pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress_bar.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.grid(row=1, column=0, padx=5, sticky="w")
        self.time_label = ttk.Label(progress_frame, text="")
        self.time_label.grid(row=1, column=1, padx=5, sticky="e")
        self.cancel_button = ttk.Button(progress_frame, text="Cancel", command=self.cancel_analysis_task, state='disabled')
        self.cancel_button.grid(row=2, column=0, columnspan=2, pady=5)
        progress_frame.columnconfigure(0, weight=1)

        # Comparison settings section
        comparison_frame = ttk.LabelFrame(self.main_frame, text="Comparison Settings", padding="5")
        comparison_frame.grid(row=3, column=0, sticky="ew", pady=5)

        ttk.Label(comparison_frame, text="Metrics:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        metrics_frame = ttk.Frame(comparison_frame)
        metrics_frame.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.metric_vars = {metric: tk.BooleanVar(value=True) for metric in self.metrics}
        for i, metric in enumerate(self.metrics):
            ttk.Checkbutton(metrics_frame, text=metric.capitalize(), variable=self.metric_vars[metric]).grid(row=0, column=i, padx=5)

        ttk.Label(comparison_frame, text="Threshold:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        threshold_frame = ttk.Frame(comparison_frame)
        threshold_frame.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.threshold_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, variable=self.threshold_var, orient=tk.HORIZONTAL, length=200, command=self.update_threshold_label)
        self.threshold_scale.grid(row=0, column=0, sticky="ew", padx=5)
        self.threshold_label = ttk.Label(threshold_frame, text="0.5")
        self.threshold_label.grid(row=0, column=1, padx=5)

        ttk.Button(comparison_frame, text="Compare Models", command=self.compare_models).grid(row=2, column=0, columnspan=2, pady=10)
        comparison_frame.columnconfigure(1, weight=1)

        # Results visualization section
        visualization_frame = ttk.LabelFrame(self.main_frame, text="Comparison Results", padding="5")
        visualization_frame.grid(row=4, column=0, sticky="nsew", pady=5)
        self.notebook = ttk.Notebook(visualization_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_tab, text="Metrics Comparison")
        self.roc_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.roc_tab, text="ROC Curves")
        self.table_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.table_tab, text="Detailed Results")
        visualization_frame.columnconfigure(0, weight=1)
        visualization_frame.rowconfigure(0, weight=1)
        self.main_frame.rowconfigure(4, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        self.metrics_plot_canvas = FigureCanvasTkAgg(Figure(figsize=(8, 5), dpi=100), master=self.metrics_tab)
        self.metrics_plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.roc_plot_canvas = FigureCanvasTkAgg(Figure(figsize=(8, 5), dpi=100), master=self.roc_tab)
        self.roc_plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        columns = ("Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC")
        self.results_tree = ttk.Treeview(self.table_tab, columns=columns, show="headings", height=10)
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=100, anchor=tk.CENTER)
        self.results_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def browse_test_dir(self):
        dir_path = filedialog.askdirectory(initialdir=self.test_dir, title="Select Test Directory")
        if dir_path:
            self.test_dir = dir_path
            self.test_dir_entry.delete(0, tk.END)
            self.test_dir_entry.insert(0, dir_path)

    def update_threshold_label(self, *args):
        self.threshold_label.config(text=f"{self.threshold_var.get():.3f}")

    def check_progress_queue(self):
        try:
            while not self.progress_queue.empty():
                progress = self.progress_queue.get_nowait()
                self.progress_bar['value'] = progress['percent']
                self.status_label['text'] = progress['status']
                self.time_label['text'] = progress.get('time_remaining', "")

                # Update model status in tree view if specified
                if 'model_path' in progress and 'new_status' in progress:
                    for item_id in self.models_tree.get_children():
                        if self.models_tree.item(item_id, 'values')[1] == progress['model_path']:
                            values = list(self.models_tree.item(item_id, 'values'))
                            values[2] = progress['new_status']
                            self.models_tree.item(item_id, values=values)
        except Queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_progress_queue)

    def cancel_analysis_task(self):
        self.cancel_analysis = True
        self.status_label['text'] = "Cancelling..."
        self.cancel_button['state'] = 'disabled'

    def add_current_model(self):
        if not hasattr(self.parent_app, 'model_path') or not self.parent_app.model_path:
            messagebox.showerror("Error", "No model loaded in main app.")
            return
        model_path_to_check = self.parent_app.model_path
        for child_id in self.models_tree.get_children():
            if self.models_tree.item(child_id, 'values')[1] == model_path_to_check:
                messagebox.showinfo("Info", "Model already added.")
                return

        model_name = os.path.basename(self.parent_app.model_path)
        status = "With predictions" if self.parent_app.current_predictions is not None else "No predictions"
        self.models_tree.insert('', tk.END, values=(model_name, self.parent_app.model_path, status))
        model_data = {
            'name': model_name,
            'path': self.parent_app.model_path,
            'predictions': self.parent_app.current_predictions,
            'true_labels': self.parent_app.current_true_labels,
            'filenames': self.parent_app.current_filenames
        }
        self.models_data.append(model_data)
        msg = "with predictions" if self.parent_app.current_predictions is not None else "without predictions"
        messagebox.showinfo("Success", f"Added current model '{model_name}' {msg}.")

    def add_new_model(self):
        file_path = filedialog.askopenfilename(title="Select Model File", filetypes=(("Model files", "*.h5 *.keras"), ("All files", "*.*")))
        if not file_path: return
        file_path_to_check = file_path
        for child_id in self.models_tree.get_children():
            if self.models_tree.item(child_id, 'values')[1] == file_path_to_check:
                messagebox.showinfo("Info", "Model already added.")
                return

        model_name = os.path.basename(file_path)
        self.models_tree.insert('', tk.END, values=(model_name, file_path, "No predictions"))
        self.models_data.append({'name': model_name, 'path': file_path, 'predictions': None, 'true_labels': None, 'filenames': None})

    def remove_model(self):
        selected_items = self.models_tree.selection()
        if not selected_items:
            messagebox.showinfo("Info", "Select model to remove.")
            return
        for item_id in selected_items:
            model_path = self.models_tree.item(item_id, 'values')[1]
            self.models_data = [data for data in self.models_data if data['path'] != model_path]
            self.models_tree.delete(item_id)

    def clear_models(self):
        if not self.models_tree.get_children(): return
        if messagebox.askyesno("Confirm", "Clear all models?"):
            self.models_tree.delete(*self.models_tree.get_children())
            self.models_data = []

    def run_all_models(self):
        if not self.models_data:
            messagebox.showinfo("Info", "No models added to run.")
            return
        if not self.test_dir:
            messagebox.showerror("Error", "Please select a test directory.")
            return

        self.cancel_analysis = False
        self.cancel_button['state'] = 'normal'
        self.progress_bar['value'] = 0
        self.status_label['text'] = "Starting analysis..."

        self.analysis_thread = threading.Thread(target=self._run_predictions_for_all_models)
        self.analysis_thread.start()

    def _run_predictions_for_all_models(self):
        batch_size = int(self.batch_size_var.get())
        test_dir = self.test_dir

        for i, model_data in enumerate(self.models_data):
            if self.cancel_analysis:
                self.progress_queue.put({'percent': 0, 'status': "Analysis cancelled", 'time_remaining': "", 'model_path': model_data['path'], 'new_status': 'Cancelled'})
                return

            model_path = model_data['path']
            model_name = model_data['name']

            self.progress_queue.put({'percent': 0, 'status': f"Loading model: {model_name}", 'model_path': model_path, 'new_status': 'Loading Model'})
            try:
                model = load_model(model_path)
                dummy_input = np.random.rand(1, 224, 224, 3)
                dummy_pred = model.predict(dummy_input, verbose=0)
                model_categorical = len(dummy_pred.shape) > 1 and dummy_pred.shape[1] > 1
                model_binary = not model_categorical

                self.progress_queue.put({'percent': 10, 'status': f"Counting samples for {model_name}", 'model_path': model_path, 'new_status': 'Counting Samples'})
                total_samples = len(glob(os.path.join(test_dir, '*/*.*g')))

                self.progress_queue.put({'percent': 20, 'status': f"Preparing data for {model_name}", 'model_path': model_path, 'new_status': 'Preparing Data'})
                test_ds = create_data_generator(test_dir, batch_size, model_categorical=model_categorical, model_binary=model_binary)

                def update_progress(current_step, total_steps):
                    if self.cancel_analysis: raise Exception("Analysis cancelled")
                    progress = (current_step + 1) / total_steps
                    percent = 20 + (70 * progress) # Prediction progress from 20% to 90%
                    self.progress_queue.put({'percent': percent, 'status': f"Predicting for {model_name} ({current_step+1}/{total_steps})", 'model_path': model_path, 'new_status': 'Predicting'})

                self.progress_queue.put({'percent': 30, 'status': f"Predicting for {model_name}...", 'model_path': model_path, 'new_status': 'Predicting'})
                start_time = time.time()
                raw_predictions, true_labels = predict_in_batches(model, test_ds, total_samples, batch_size, update_progress)
                if not raw_predictions.size: raise Exception("No predictions made")
                elapsed_time = time.time() - start_time
                time_str = f"Prediction Time: {elapsed_time:.2f}s"

                predictions_data = {'raw_predictions': raw_predictions, 'true_labels': true_labels, 'filenames': test_ds.filenames[:len(raw_predictions)]}
                model_data['predictions'] = raw_predictions
                model_data['true_labels'] = true_labels
                model_data['filenames'] = test_ds.filenames[:len(raw_predictions)]

                self.progress_queue.put({'percent': 100, 'status': f"Predictions complete for {model_name}. {time_str}", 'model_path': model_path, 'new_status': 'Predictions Done'})

            except Exception as e:
                error_msg = f"Error analyzing {model_name}: {e}"
                self.progress_queue.put({'percent': 0, 'status': error_msg, 'time_remaining': "", 'model_path': model_path, 'new_status': 'Error'})
                print(error_msg)
                continue # Continue to next model even if one fails

        self.progress_queue.put({'percent': 100, 'status': "All models analyzed.", 'time_remaining': ""})
        self.cancel_button['state'] = 'disabled'
        self.cancel_analysis = False


    def compare_models(self):
        if not self.models_data:
            messagebox.showinfo("Info", "No models added to compare.")
            return

        missing_preds_models = [m['name'] for m in self.models_data if m['predictions'] is None]
        if missing_preds_models:
            messagebox.showinfo("Info", f"Predictions are missing for models: {', '.join(missing_preds_models)}.\nPlease run predictions for all models first.")
            return

        threshold = self.threshold_var.get()
        results = self.calculate_metrics(threshold)
        self.update_metrics_plot(results)
        self.update_roc_plot()
        self.update_results_table(results)


    def load_predictions_from_cache(self):
        # In ModelComparisonGUI, loading from cache might not be directly relevant as we rerun predictions.
        # This method can be kept for potential future use or removed if not needed.
        pass

    def calculate_metrics(self, threshold):
        results = []
        for model_data in self.models_data:
            if model_data['predictions'] is None or model_data['true_labels'] is None: continue

            is_categorical = len(model_data['predictions'].shape) > 1 and model_data['predictions'].shape[1] > 1
            preds = model_data['predictions']
            labels = model_data['true_labels']

            if is_categorical:
                y_pred_classes = np.argmax(preds, axis=1)
                y_true = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels
                auc_val = roc_auc_score(y_true, preds, multi_class='ovr', average='macro') if preds.shape[1] > 2 else roc_auc_score(y_true, preds[:, 1]) if preds.shape[1] == 2 else np.nan
            else:
                y_pred_classes = (preds.flatten() > threshold).astype(int)
                y_true = labels.flatten()
                fpr, tpr, _ = roc_curve(y_true, preds.flatten())
                auc_val = auc(fpr, tpr)

            results.append({
                'name': model_data['name'],
                'accuracy': accuracy_score(y_true, y_pred_classes),
                'precision': precision_score(y_true, y_pred_classes, average='macro', zero_division=0),
                'recall': recall_score(y_true, y_pred_classes, average='macro', zero_division=0),
                'f1': f1_score(y_true, y_pred_classes, average='macro', zero_division=0),
                'auc': auc_val if not np.isnan(auc_val) else np.nan,
            })
        return results

    def update_metrics_plot(self, results):
        ax = self.metrics_plot_canvas.figure.axes[0] if self.metrics_plot_canvas.figure.axes else self.metrics_plot_canvas.figure.add_subplot(111)
        ax.clear()
        selected_metrics = [m for m, var in self.metric_vars.items() if var.get()]
        if not selected_metrics or not results: return

        model_names = [res['name'] for res in results]
        x = np.arange(len(model_names))
        width = 0.8 / len(selected_metrics)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, metric in enumerate(selected_metrics):
            values = [res[metric] for res in results]
            ax.bar(x + i * width - (len(selected_metrics) - 1) * width / 2, values, width, label=metric.capitalize(), color=colors[i % len(colors)])

        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison - Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1)
        self.metrics_plot_canvas.figure.tight_layout()
        self.metrics_plot_canvas.draw()

    def update_roc_plot(self):
        ax = self.roc_plot_canvas.figure.axes[0] if self.roc_plot_canvas.figure.axes else self.roc_plot_canvas.figure.add_subplot(111)
        ax.clear()

        for model_data in self.models_data:
            if model_data['predictions'] is None or model_data['true_labels'] is None: continue
            if len(model_data['predictions'].shape) > 1 and model_data['predictions'].shape[1] > 1: continue # Skip categorical models for ROC plot

            preds = model_data['predictions'].flatten()
            y_true = model_data['true_labels'].flatten()
            try:
                fpr, tpr, _ = roc_curve(y_true, preds)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f'{model_data["name"]} (AUC = {roc_auc:.3f})')
            except ValueError as e:
                print(f"ROC curve error for {model_data['name']}: {e}")

        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc="lower right")
        ax.grid(linestyle='--', alpha=0.7)
        self.roc_plot_canvas.figure.tight_layout()
        self.roc_plot_canvas.draw()

    def update_results_table(self, results):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        for result in results:
            self.results_tree.insert('', tk.END, values=[result['name']] + [f"{result[m]:.4f}" for m in self.metrics[0:-1]] + [f"{result['auc']:.4f}" if not np.isnan(result['auc']) else "N/A"])


def add_model_comparison_to_main_app(app):
    def open_comparison_window():
        ModelComparisonGUI(tk.Toplevel(app.root), app)
    ttk.Button(app.main_frame, text="Compare Models", command=open_comparison_window).grid(row=0, column=2, padx=10, pady=5, sticky=tk.N)

def modify_main_class():
    original_init = ModelAnalysisGUI.__init__
    def new_init(self, root):
        original_init(self, root)
        add_model_comparison_to_main_app(self)
    ModelAnalysisGUI.__init__ = new_init

if __name__ == "__main__":
    modify_main_class()
    root = tk.Tk()
    app = ModelAnalysisGUI(root)
    root.mainloop()