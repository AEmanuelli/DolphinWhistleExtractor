import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image, ImageTk

class ModelAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Prediction Analysis")
        self.root.geometry("800x600")
        
        # Model paths
        self.model_path = None
        self.test_dir = None
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create UI elements
        self.create_input_section()
        self.create_results_section()
        self.create_image_viewer()
        
    def create_input_section(self):
        # Model selection
        ttk.Label(self.main_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_btn = ttk.Button(self.main_frame, text="Browse", command=self.load_model_file)
        self.model_btn.grid(row=0, column=1, sticky=tk.W)
        
        # Test directory selection
        ttk.Label(self.main_frame, text="Select Test Directory:").grid(row=1, column=0, sticky=tk.W)
        self.test_dir_btn = ttk.Button(self.main_frame, text="Browse", command=self.load_test_dir)
        self.test_dir_btn.grid(row=1, column=1, sticky=tk.W)
        
        # Analysis type selection
        ttk.Label(self.main_frame, text="Analysis Type:").grid(row=2, column=0, sticky=tk.W)
        self.analysis_type = ttk.Combobox(self.main_frame, 
                                        values=["False Positives", "False Negatives", "All Misclassified"])
        self.analysis_type.grid(row=2, column=1, sticky=tk.W)
        self.analysis_type.set("False Positives")
        
        # Run analysis button
        self.run_btn = ttk.Button(self.main_frame, text="Run Analysis", command=self.run_analysis)
        self.run_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
    def create_results_section(self):
        # Results frame
        results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="5")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Metrics display
        self.metrics_text = tk.Text(results_frame, height=5, width=60)
        self.metrics_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Misclassified files list
        self.file_listbox = tk.Listbox(results_frame, height=10)
        self.file_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.file_listbox.bind('<<ListboxSelect>>', self.on_select_file)
        
    def create_image_viewer(self):
        # Image viewer frame
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Image Preview", padding="5")
        self.image_frame.grid(row=0, column=2, rowspan=5, padx=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image label
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0)
        
    def load_model_file(self):
        self.model_path = filedialog.askopenfilename(
            filetypes=[("H5 files", "*.h5"), ("All files", "*.*")])
        if self.model_path:
            self.model_btn.configure(text="Model Selected")
            
    def load_test_dir(self):
        self.test_dir = filedialog.askdirectory()
        if self.test_dir:
            self.test_dir_btn.configure(text="Directory Selected")
            
    def run_analysis(self):
        if not self.model_path or not self.test_dir:
            messagebox.showerror("Error", "Please select both model and test directory")
            return
            
        try:
            # Load model
            model = load_model(self.model_path)
            
            # Setup data generator
            data_gen = ImageDataGenerator(rescale=1/255.)
            test_ds = data_gen.flow_from_directory(
                self.test_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='binary',
                shuffle=False
            )
            
            # Get predictions
            y_pred = model.predict(test_ds)
            y_true = test_ds.classes
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            
            # Calculate metrics
            report = classification_report(y_true, y_pred_classes)
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, report)
            
            # Get misclassified indices based on selection
            if self.analysis_type.get() == "False Positives":
                indices = np.where((y_true == 0) & (y_pred_classes == 1))[0]
            elif self.analysis_type.get() == "False Negatives":
                indices = np.where((y_true == 1) & (y_pred_classes == 0))[0]
            else:  # All Misclassified
                indices = np.where(y_true != y_pred_classes)[0]
            
            # Update listbox
            self.file_listbox.delete(0, tk.END)
            self.misclassified_files = [test_ds.filenames[i] for i in indices]
            for file in self.misclassified_files:
                self.file_listbox.insert(tk.END, file)
                
            messagebox.showinfo("Success", f"Found {len(indices)} {self.analysis_type.get()}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def on_select_file(self, event):
        if not self.file_listbox.curselection():
            return
            
        # Get selected file
        selection = self.file_listbox.get(self.file_listbox.curselection())
        image_path = os.path.join(self.test_dir, selection)
        
        # Load and display image
        try:
            image = Image.open(image_path)
            image = image.resize((200, 200))  # Resize for display
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelAnalysisGUI(root)
    root.mainloop()