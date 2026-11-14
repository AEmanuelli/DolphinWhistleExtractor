import os
import sys
from datetime import datetime

# -----------------------
# Default paths (relative to the repository, never machine-specific)
# -----------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
predict_dir = os.path.dirname(os.path.dirname(current_dir))  # .../Predict_and_extract
repo_root = os.path.dirname(predict_dir)

# Make the pipeline modules importable whatever the working directory
if predict_dir not in sys.path:
    sys.path.insert(0, predict_dir)

cwd = os.getcwd()
default_model_path = os.path.join(repo_root, "models", "base_MobileNetV2.keras")
default_recordings = os.path.join(cwd, "recordings")
default_saving_folder = os.path.join(cwd, "processed", f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
