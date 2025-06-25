import os
from datetime import datetime

# -----------------------
# Default Paths based on Current Working Directory
# -----------------------
cwd = os.getcwd()
default_model_path = os.path.join(cwd, "models", "base_MobileNetV2.keras")
default_recordings = "/home/emanuelli/Bureau/Watkins/68027" # Or use os.path.join(cwd, "recordings") for relative
default_saving_folder = "/home/emanuelli/Bureau/Watkins/68027/processed" # Or use os.path.join(cwd, "processed") for relative

if not os.path.exists(default_saving_folder):
    os.makedirs(default_saving_folder, exist_ok=True)
default_saving_folder = os.path.join(default_saving_folder, f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")