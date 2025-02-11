import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import sys
import threading

# -----------------------
# Default Paths based on Current Working Directory
# -----------------------
cwd = os.getcwd()
default_model_path = os.path.join(cwd, "models", "model_vgg.h5")
default_recordings = os.path.join(cwd, "recordings")
default_saving_folder = os.path.join(cwd, "processed")

# -----------------------
# Helper Functions
# -----------------------

def browse_file(entry, file_description="Select File"):
    """Open a file dialog and insert the selected file path into the given entry widget."""
    filename = filedialog.askopenfilename(title=file_description)
    if filename:
        entry.delete(0, tk.END)
        entry.insert(0, filename)

def browse_folder(entry, folder_description="Select Folder"):
    """Open a folder dialog and insert the selected folder path into the given entry widget."""
    folder = filedialog.askdirectory(title=folder_description)
    if folder:
        entry.delete(0, tk.END)
        entry.insert(0, folder)

def install_dependencies():
    """Install dependencies from requirements.txt using pip."""
    requirements_path = "requirements.txt"
    if not os.path.exists(requirements_path):
        messagebox.showerror("Error", "requirements.txt not found!")
        return

    progress_bar.start(10)
    def install_thread():
        command = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                messagebox.showinfo("Success", "Dependencies installed successfully!")
            else:
                messagebox.showerror("Installation Error", f"Error installing dependencies:\n{stderr}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            progress_bar.stop()

    threading.Thread(target=install_thread, daemon=True).start()

def run_process():
    """Collect parameters from the UI, then run the processing function in a background thread with a progress bar."""
    try:
        model_path = model_entry.get().strip()
        recordings = recordings_entry.get().strip()
        saving_folder = saving_folder_entry.get().strip()
        start_time = int(start_time_entry.get().strip() or "0")
        end_time_val = end_time_entry.get().strip()
        end_time = int(end_time_val) if end_time_val else None
        batch_size = int(batch_size_entry.get().strip())
        save = save_var.get()
        save_p = save_p_var.get()
        max_workers = int(max_workers_entry.get().strip())
        specific_files_path = specific_files_entry.get().strip()
        CLF = int(clf_entry.get().strip())
        CHF = int(chf_entry.get().strip())
        image_norm = image_norm_var.get()

        specific_files = None
        if specific_files_path:
            with open(specific_files_path, 'r') as f:
                specific_files = f.read().splitlines()
    except Exception as e:
        messagebox.showerror("Input Error", f"Please check input values:\n{e}")
        return

    def process_thread():
        progress_bar.start(10)
        try:
            # Import the processing function only when needed
            from predict_and_extract_online import process_predict_extract
            process_predict_extract(
                recording_folder_path=recordings,
                saving_folder=saving_folder,
                CLF=CLF,
                CHF=CHF,
                image_norm=image_norm,
                start_time=start_time,
                end_time=end_time,
                batch_size=batch_size,
                save=save,
                save_p=save_p,
                model_path=model_path,
                max_workers=max_workers,
                specific_files=specific_files
            )
            messagebox.showinfo("Success", "Processing completed successfully!")
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during processing:\n{e}")
        finally:
            progress_bar.stop()

    threading.Thread(target=process_thread, daemon=True).start()

# -----------------------
# Main GUI Setup
# -----------------------

root = tk.Tk()
root.title("Processing GUI")
root.resizable(False, False)

# Use ttk style for a nicer look
style = ttk.Style()
if "clam" in style.theme_names():
    style.theme_use("clam")
else:
    style.theme_use("default")

# Create a main frame with padding
main_frame = ttk.Frame(root, padding="15 15 15 15")
main_frame.grid(row=0, column=0, sticky="NSEW")

# Row counter
r = 0

# Model Path
ttk.Label(main_frame, text="Model Path:").grid(row=r, column=0, sticky="E", pady=5)
model_entry = ttk.Entry(main_frame, width=50)
model_entry.insert(0, default_model_path)
model_entry.grid(row=r, column=1, padx=5)
ttk.Button(main_frame, text="Select Model File", command=lambda: browse_file(model_entry, "Select Model File")).grid(row=r, column=2, padx=5)
r += 1

# Recordings Folder
ttk.Label(main_frame, text="Recordings Folder:").grid(row=r, column=0, sticky="E", pady=5)
recordings_entry = ttk.Entry(main_frame, width=50)
recordings_entry.insert(0, default_recordings)
recordings_entry.grid(row=r, column=1, padx=5)
ttk.Button(main_frame, text="Select Recordings Folder", command=lambda: browse_folder(recordings_entry, "Select Recordings Folder")).grid(row=r, column=2, padx=5)
r += 1

# Saving Folder
ttk.Label(main_frame, text="Saving Folder:").grid(row=r, column=0, sticky="E", pady=5)
saving_folder_entry = ttk.Entry(main_frame, width=50)
saving_folder_entry.insert(0, default_saving_folder)
saving_folder_entry.grid(row=r, column=1, padx=5)
ttk.Button(main_frame, text="Select Saving Folder", command=lambda: browse_folder(saving_folder_entry, "Select Saving Folder")).grid(row=r, column=2, padx=5)
r += 1

# Start Time
ttk.Label(main_frame, text="Start Time:").grid(row=r, column=0, sticky="E", pady=5)
start_time_entry = ttk.Entry(main_frame, width=50)
start_time_entry.insert(0, "0")
start_time_entry.grid(row=r, column=1, padx=5)
r += 1

# End Time
ttk.Label(main_frame, text="End Time:").grid(row=r, column=0, sticky="E", pady=5)
end_time_entry = ttk.Entry(main_frame, width=50)
end_time_entry.grid(row=r, column=1, padx=5)
r += 1

# Batch Size
ttk.Label(main_frame, text="Batch Size:").grid(row=r, column=0, sticky="E", pady=5)
batch_size_entry = ttk.Entry(main_frame, width=50)
batch_size_entry.insert(0, "64")
batch_size_entry.grid(row=r, column=1, padx=5)
r += 1

# Save all the images Option
save_var = tk.BooleanVar(value=False)
ttk.Checkbutton(main_frame, text="Save all the images", variable=save_var).grid(row=r, column=0, columnspan=2, sticky="W", pady=5)
r += 1

# Save Positives Option
save_p_var = tk.BooleanVar(value=True)
ttk.Checkbutton(main_frame, text="Save Positives", variable=save_p_var).grid(row=r, column=0, columnspan=2, sticky="W", pady=5)
r += 1

# Max Workers
ttk.Label(main_frame, text="Max Workers:").grid(row=r, column=0, sticky="E", pady=5)
max_workers_entry = ttk.Entry(main_frame, width=50)
max_workers_entry.insert(0, "8")
max_workers_entry.grid(row=r, column=1, padx=5)
r += 1

# Specific Files List
ttk.Label(main_frame, text="Specific Files (list file):").grid(row=r, column=0, sticky="E", pady=5)
specific_files_entry = ttk.Entry(main_frame, width=50)
specific_files_entry.grid(row=r, column=1, padx=5)
ttk.Button(main_frame, text="Select Specific Files List", command=lambda: browse_file(specific_files_entry, "Select Specific Files List")).grid(row=r, column=2, padx=5)
r += 1

# Cut Low Frequency (CLF)
ttk.Label(main_frame, text="Cut Low Frequency (CLF):").grid(row=r, column=0, sticky="E", pady=5)
clf_entry = ttk.Entry(main_frame, width=50)
clf_entry.insert(0, "3")
clf_entry.grid(row=r, column=1, padx=5)
r += 1

# Cut High Frequency (CHF)
ttk.Label(main_frame, text="Cut High Frequency (CHF):").grid(row=r, column=0, sticky="E", pady=5)
chf_entry = ttk.Entry(main_frame, width=50)
chf_entry.insert(0, "20")
chf_entry.grid(row=r, column=1, padx=5)
r += 1

# Image Normalization Option
image_norm_var = tk.BooleanVar(value=False)
ttk.Checkbutton(main_frame, text="Image Normalization (/255)", variable=image_norm_var).grid(row=r, column=0, columnspan=2, sticky="W", pady=5)
r += 1

# -----------------------
# Buttons & Progress Bar
# -----------------------

# Create a frame for buttons
button_frame = ttk.Frame(main_frame)
button_frame.grid(row=r, column=0, columnspan=3, pady=10)

install_btn = ttk.Button(button_frame, text="Install Dependencies", command=install_dependencies)
install_btn.grid(row=0, column=0, padx=5)

start_btn = ttk.Button(button_frame, text="Start Processing", command=run_process)
start_btn.grid(row=0, column=1, padx=5)

r += 1

# Global Progress Bar
progress_bar = ttk.Progressbar(main_frame, mode="indeterminate", length=400)
progress_bar.grid(row=r, column=0, columnspan=3, pady=10)

# -----------------------
# Run the Application
# -----------------------
root.mainloop()
