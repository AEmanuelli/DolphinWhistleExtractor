import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import sys
import threading
import queue
from typing import Optional

# -----------------------
# Default Paths based on Current Working Directory
# -----------------------
cwd = os.getcwd()
default_model_path = os.path.join(cwd, "models", "model_vgg.h5")
default_recordings = os.path.join(cwd, "recordings")
default_saving_folder = os.path.join(cwd, "processed")

class ProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whistle Extractor")
        self.root.resizable(False, False)
        
        # Queue for thread-safe communication
        self.output_queue = queue.Queue()
        
        # Advanced settings state
        self.show_advanced = tk.BooleanVar(value=False)
        
        # Initialize UI components
        self.setup_ui()
        
        # Start queue checker
        self.check_output_queue()

    def setup_ui(self):
        # Use ttk style for a nicer look
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        
        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="15 15 15 15")
        self.main_frame.grid(row=0, column=0, sticky="NSEW")
        
        self.create_basic_settings()
        self.create_advanced_settings()
        self.create_buttons()
        self.create_progress_bar()
        self.create_output_area()
        
        # Configure grid
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(self.row_counter, weight=1)

    def create_basic_settings(self):
        self.row_counter = 0
        
        # Add a header for basic settings
        header_label = ttk.Label(self.main_frame, text="Basic Settings", font=('TkDefaultFont', 10, 'bold'))
        header_label.grid(row=self.row_counter, column=0, columnspan=3, sticky="W", pady=(0, 10))
        self.row_counter += 1
        
        # Model Path
        self.model_entry = self.add_file_input("Model Path:", default_model_path, 
                                             lambda: self.browse_file(self.model_entry, "Select Model File (*.h5)"),
                                             file_types=[("H5 files", "*.h5"), ("All files", "*.*")])
        
        # Recordings Folder
        self.recordings_entry = self.add_file_input("Recordings Folder:", default_recordings,
                                                  lambda: self.browse_folder(self.recordings_entry, "Select Recordings Folder"))
        
        # Saving Folder
        self.saving_folder_entry = self.add_file_input("Saving Folder:", default_saving_folder,
                                                     lambda: self.browse_folder(self.saving_folder_entry, "Select Saving Folder"))
        
        # Time settings group
        time_frame = ttk.LabelFrame(self.main_frame, text="Time Settings", padding="5 5 5 5")
        time_frame.grid(row=self.row_counter, column=0, columnspan=3, sticky="EW", pady=5)
        self.row_counter += 1
        
        # Basic parameters in time frame
        self.start_time_entry = self.add_entry_to_frame(time_frame, "Start Time (seconds):", "0", 0)
        self.end_time_entry = self.add_entry_to_frame(time_frame, "End Time (seconds):", "", 1)
        ttk.Label(time_frame, text="(Leave empty to process entire file)", 
                 foreground="gray").grid(row=1, column=2, padx=5)
        
        # Output settings group
        output_frame = ttk.LabelFrame(self.main_frame, text="Output Settings", padding="5 5 5 5")
        output_frame.grid(row=self.row_counter, column=0, columnspan=3, sticky="EW", pady=5)
        self.row_counter += 1
        
        # Checkboxes in output frame
        self.save_var = tk.BooleanVar(value=False)
        self.save_p_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(output_frame, text="Save all spectrograms images", 
                       variable=self.save_var).grid(row=0, column=0, sticky="W", padx=5)
        ttk.Checkbutton(output_frame, text="Save Positives", 
                       variable=self.save_p_var).grid(row=1, column=0, sticky="W", padx=5)

    def create_advanced_settings(self):
        # Advanced settings toggle button with styling
        advanced_btn = ttk.Button(self.main_frame, text="▼ Advanced Settings", 
                                command=self.toggle_advanced_settings,
                                style="Advanced.TButton")
        advanced_btn.grid(row=self.row_counter, column=0, columnspan=3, pady=10, sticky="W")
        self.row_counter += 1
        
        # Create advanced settings frame
        self.advanced_frame = ttk.LabelFrame(self.main_frame, text="Advanced Settings", padding="5 5 5 5")
        self.advanced_frame.grid(row=self.row_counter, column=0, columnspan=3, sticky="NSEW", pady=5)
        self.advanced_frame.grid_remove()  # Initially hidden
        
        # Performance settings group
        perf_frame = ttk.LabelFrame(self.advanced_frame, text="Performance", padding="5 5 5 5")
        perf_frame.grid(row=0, column=0, columnspan=2, sticky="EW", pady=5)
        
        # Batch size and max workers in performance frame
        self.batch_size_entry = self.add_entry_to_frame(perf_frame, "Batch Size:", "64", 0)
        self.max_workers_entry = self.add_entry_to_frame(perf_frame, "Max Workers:", "8", 1)
        
        # Frequency settings group
        freq_frame = ttk.LabelFrame(self.advanced_frame, text="Frequency Settings", padding="5 5 5 5")
        freq_frame.grid(row=1, column=0, columnspan=2, sticky="EW", pady=5)
        
        # Cut frequencies in frequency frame
        self.clf_entry = self.add_entry_to_frame(freq_frame, "Cut Low Frequency (kHz):", "3", 0)
        self.chf_entry = self.add_entry_to_frame(freq_frame, "Cut High Frequency (kHz):", "20", 1)
        
        # Model settings group
        model_frame = ttk.LabelFrame(self.advanced_frame, text="Model Settings", padding="5 5 5 5")
        model_frame.grid(row=2, column=0, columnspan=2, sticky="EW", pady=5)
        
        # Image normalization with warning in model frame
        self.image_norm_var = tk.BooleanVar(value=False)
        norm_frame = ttk.Frame(model_frame)
        norm_frame.grid(row=0, column=0, columnspan=2, sticky="W", pady=5)
        
        ttk.Checkbutton(norm_frame, text="Image Normalization (/255)", 
                       variable=self.image_norm_var).pack(side=tk.LEFT)
        ttk.Label(norm_frame, text="(Leave unchecked unless using a different model)", 
                 foreground="gray").pack(side=tk.LEFT, padx=5)
        
        # Hidden specific files entry
        self.specific_files_entry = ttk.Entry(self.advanced_frame)
        self.specific_files_entry.grid_remove()
        
        self.row_counter += 1

    def add_file_input(self, label: str, default_value: str, browse_command, file_types=None) -> ttk.Entry:
        ttk.Label(self.main_frame, text=label).grid(row=self.row_counter, column=0, sticky="E", pady=5)
        entry = ttk.Entry(self.main_frame, width=50)
        entry.insert(0, default_value)
        entry.grid(row=self.row_counter, column=1, padx=5)
        
        browse_btn = ttk.Button(self.main_frame, text="Browse", command=browse_command)
        browse_btn.grid(row=self.row_counter, column=2, padx=5)
        
        self.row_counter += 1
        return entry

    def add_entry_to_frame(self, frame, label: str, default_value: str, row: int) -> ttk.Entry:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="E", pady=5)
        entry = ttk.Entry(frame, width=50)
        entry.insert(0, default_value)
        entry.grid(row=row, column=1, padx=5)
        return entry

    def toggle_advanced_settings(self):
        if self.advanced_frame.winfo_viewable():
            self.advanced_frame.grid_remove()
        else:
            self.advanced_frame.grid()

    def create_buttons(self):
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=self.row_counter, column=0, columnspan=3, pady=10)
        
        self.install_btn = ttk.Button(button_frame, text="Install Dependencies", 
                                    command=self.install_dependencies)
        self.install_btn.grid(row=0, column=0, padx=5)
        
        self.start_btn = ttk.Button(button_frame, text="Start Processing", 
                                  command=self.run_process)
        self.start_btn.grid(row=0, column=1, padx=5)
        
        self.row_counter += 1

    def create_progress_bar(self):
        self.progress_bar = ttk.Progressbar(self.main_frame, mode="indeterminate", length=400)
        self.progress_bar.grid(row=self.row_counter, column=0, columnspan=3, pady=10, sticky="WE")
        self.row_counter += 1

    def create_output_area(self):
        ttk.Label(self.main_frame, text="Processing Log:").grid(row=self.row_counter, 
                                                              column=0, sticky="W", pady=5)
        self.install_output_text = scrolledtext.ScrolledText(self.main_frame, height=10, 
                                                           width=60, wrap=tk.WORD)
        self.install_output_text.grid(row=self.row_counter + 1, column=0, columnspan=3, 
                                    sticky="NSEW", padx=5, pady=5)
        self.install_output_text.config(state=tk.DISABLED)
        self.row_counter += 2

    def check_output_queue(self):
        """Check for output in the queue and update the UI accordingly"""
        try:
            while True:
                msg = self.output_queue.get_nowait()
                self.update_output_text(msg)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_output_queue)

    def update_output_text(self, text: str):
        """Thread-safe update of the output text widget"""
        self.install_output_text.config(state=tk.NORMAL)
        self.install_output_text.insert(tk.END, text)
        self.install_output_text.see(tk.END)
        self.install_output_text.config(state=tk.DISABLED)

    def browse_file(self, entry: ttk.Entry, file_description: str = "Select File"):
        """Open a file dialog and insert the selected file path into the given entry widget."""
        filename = filedialog.askopenfilename(title=file_description)
        if filename:
            entry.delete(0, tk.END)
            entry.insert(0, filename)

    def browse_folder(self, entry: ttk.Entry, folder_description: str = "Select Folder"):
        """Open a folder dialog and insert the selected folder path into the given entry widget."""
        folder = filedialog.askdirectory(title=folder_description)
        if folder:
            entry.delete(0, tk.END)
            entry.insert(0, folder)

    def toggle_controls(self, enabled: bool):
        """Enable or disable all controls"""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.install_btn.config(state=state)
        self.start_btn.config(state=state)

    def install_dependencies(self):
        """Install dependencies from requirements.txt using pip"""
        requirements_path = "requirements.txt"
        if not os.path.exists(requirements_path):
            messagebox.showerror("Error", "requirements.txt not found!")
            return

        self.install_output_text.config(state=tk.NORMAL)
        self.install_output_text.delete(1.0, tk.END)
        self.install_output_text.insert(tk.END, "Installing dependencies...\n")
        self.install_output_text.config(state=tk.DISABLED)
        
        self.progress_bar.start(10)
        self.toggle_controls(False)

        def install_thread():
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "pip", "install", "-r", requirements_path,
                     "--disable-pip-version-check", "--no-cache-dir"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                for line in process.stdout:
                    self.output_queue.put(line)

                process.wait()
                if process.returncode == 0:
                    self.root.after(0, messagebox.showinfo, "Success", 
                                  "Dependencies installed successfully!")
                else:
                    self.root.after(0, messagebox.showerror, "Installation Error",
                                  "Error installing dependencies. Check output for details.")
            except Exception as e:
                self.root.after(0, messagebox.showerror, "Error", str(e))
            finally:
                self.root.after(0, self.progress_bar.stop)
                self.root.after(0, lambda: self.toggle_controls(True))

        threading.Thread(target=install_thread, daemon=True).start()

    def validate_inputs(self) -> Optional[dict]:
        """Validate all inputs and return them as a dictionary if valid"""
        try:
            return {
                'model_path': self.model_entry.get().strip(),
                'recordings': self.recordings_entry.get().strip(),
                'saving_folder': self.saving_folder_entry.get().strip(),
                'start_time': int(self.start_time_entry.get().strip() or "0"),
                'end_time': int(self.end_time_entry.get().strip()) if self.end_time_entry.get().strip() else None,
                'batch_size': int(self.batch_size_entry.get().strip()),
                'save': self.save_var.get(),
                'save_p': self.save_p_var.get(),
                'max_workers': int(self.max_workers_entry.get().strip()),
                'specific_files_path': self.specific_files_entry.get().strip(),
                'CLF': int(self.clf_entry.get().strip()),
                'CHF': int(self.chf_entry.get().strip()),
                'image_norm': self.image_norm_var.get()
            }
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please check input values:\n{str(e)}")
            return None

    def run_process(self):
        """Run the main processing function"""
        inputs = self.validate_inputs()
        if not inputs:
            return

        # Handle specific files
        if inputs['specific_files_path']:
            try:
                with open(inputs['specific_files_path'], 'r') as f:
                    inputs['specific_files'] = f.read().splitlines()
            except Exception as e:
                messagebox.showerror("Error", f"Error reading specific files list:\n{str(e)}")
                return
        else:
            inputs['specific_files'] = None

        def process_thread():
            self.progress_bar.start(10)
            self.toggle_controls(False)

            # Create a custom writer that queues printed messages
            class QueueWriter:
                def __init__(self, queue_obj):
                    self.queue = queue_obj
                def write(self, msg):
                    if msg.strip():
                        self.queue.put(msg)
                def flush(self):
                    pass

            qw = QueueWriter(self.output_queue)
            # Backup the original stdout and redirect
            original_stdout = sys.stdout
            sys.stdout = qw

            try:
                from predict_and_extract_online import process_predict_extract
                process_predict_extract(
                    recording_folder_path=inputs['recordings'],
                    saving_folder=inputs['saving_folder'],
                    CLF=inputs['CLF'],
                    CHF=inputs['CHF'],
                    image_norm=inputs['image_norm'],
                    start_time=inputs['start_time'],
                    end_time=inputs['end_time'],
                    batch_size=inputs['batch_size'],
                    save=inputs['save'],
                    save_p=inputs['save_p'],
                    model_path=inputs['model_path'],
                    max_workers=inputs['max_workers'],
                    specific_files=inputs['specific_files']
                )
                self.root.after(0, messagebox.showinfo, "Success", 
                                "Processing completed successfully!")
            except Exception as e:
                self.root.after(0, messagebox.showerror, "Processing Error", 
                                f"An error occurred during processing:\n{str(e)}")
            finally:
                # Restore original sys.stdout
                sys.stdout = original_stdout
                self.root.after(0, self.progress_bar.stop)
                self.root.after(0, lambda: self.toggle_controls(True))

        threading.Thread(target=process_thread, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ProcessingGUI(root)
    root.mainloop()