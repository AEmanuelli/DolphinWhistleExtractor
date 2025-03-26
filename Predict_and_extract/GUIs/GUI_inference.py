import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import sys
import threading
import queue
from typing import Optional
from pathlib import Path
from datetime import datetime
# Get the directory containing the current script (GUI directory)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the current directory (Predict directory)
predict_dir = os.path.dirname(current_dir)

# Add the 'Predict' directory to sys.path at the beginning
sys.path.insert(0, predict_dir)

from predict_and_extract_online import process_predict_extract

# -----------------------
# Default Paths based on Current Working Directory
# -----------------------
cwd = os.getcwd()
default_model_path = os.path.join(cwd, "models", "base_MobileNetV2.keras")
default_recordings = "/home/emanuelli/Bureau/Watkins/68027"
# os.path.join(cwd, "recordings") # Relative path
default_saving_folder = "/home/emanuelli/Bureau/Watkins/68027/processed"
# os.path.join(cwd, "processed") # Relative path
if not os.path.exists(default_saving_folder):
    os.makedirs(default_saving_folder, exist_ok=True)
default_saving_folder = os.path.join(default_saving_folder, f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


class ModernUI:
    """Helper class for modern UI components and styling"""

    # Color scheme
    PRIMARY_COLOR = "#1e88e5"
    SECONDARY_COLOR = "#26a69a"
    BG_COLOR = "#f5f5f7"
    DARK_BG = "#2c3e50"
    CARD_BG = "#ffffff"
    ERROR_COLOR = "#e53935"
    SUCCESS_COLOR = "#43a047"
    WARN_COLOR = "#ff9800"

    @staticmethod
    def setup_styles():
        """Setup ttk styles for modern look and feel"""
        style = ttk.Style()

        # Use clam as base theme if available
        if "clam" in style.theme_names():
            style.theme_use("clam")

        # Configure main styling elements
        style.configure("TFrame", background=ModernUI.BG_COLOR)
        style.configure("Card.TFrame", background=ModernUI.CARD_BG, relief="flat")

        style.configure("TLabel", background=ModernUI.BG_COLOR, font=("Segoe UI", 9))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground=ModernUI.PRIMARY_COLOR)
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), foreground=ModernUI.PRIMARY_COLOR)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#757575")

        # Button styles
        style.configure("TButton", font=("Segoe UI", 9), background=ModernUI.PRIMARY_COLOR)
        style.map("TButton",
                  background=[("active", ModernUI.PRIMARY_COLOR), ("pressed", "#1565c0")],
                  foreground=[("active", "white"), ("pressed", "white")])

        style.configure("Primary.TButton", background=ModernUI.PRIMARY_COLOR, foreground="white")
        style.map("Primary.TButton",
                  background=[("active", "#1976d2"), ("pressed", "#1565c0")],
                  foreground=[("active", "white"), ("pressed", "white")])

        style.configure("Secondary.TButton", background=ModernUI.SECONDARY_COLOR, foreground="white")
        style.map("Secondary.TButton",
                  background=[("active", "#00897b"), ("pressed", "#00796b")],
                  foreground=[("active", "white"), ("pressed", "white")])

        style.configure("Success.TButton", background=ModernUI.SUCCESS_COLOR, foreground="white")
        style.map("Success.TButton",
                  background=[("active", "#388e3c"), ("pressed", "#2e7d32")],
                  foreground=[("active", "white"), ("pressed", "white")])

        style.configure("Warning.TButton", background=ModernUI.WARN_COLOR, foreground="white")
        style.map("Warning.TButton",
                  background=[("active", "#f57c00"), ("pressed", "#ef6c00")],
                  foreground=[("active", "white"), ("pressed", "white")])

        style.configure("Error.TButton", background=ModernUI.ERROR_COLOR, foreground="white")
        style.map("Error.TButton",
                  background=[("active", "#d32f2f"), ("pressed", "#c62828")],
                  foreground=[("active", "white"), ("pressed", "white")])

        style.configure("Link.TButton", background=ModernUI.BG_COLOR, foreground=ModernUI.PRIMARY_COLOR, font=("Segoe UI", 9, "underline"))
        style.map("Link.TButton",
                  background=[("active", ModernUI.BG_COLOR), ("pressed", ModernUI.BG_COLOR)],
                  foreground=[("active", "#1976d2"), ("pressed", "#1565c0")])

        # Checkbutton styling
        style.configure("TCheckbutton", background=ModernUI.BG_COLOR)
        style.map("TCheckbutton", background=[("active", ModernUI.BG_COLOR)])

        # Entry styling
        style.configure("TEntry", foreground="black", fieldbackground="white")

        # Labelframe styling
        style.configure("TLabelframe", background=ModernUI.CARD_BG)
        style.configure("TLabelframe.Label", background=ModernUI.BG_COLOR, font=("Segoe UI", 9, "bold"))

        # Notebook styling
        style.configure("TNotebook", background=ModernUI.BG_COLOR, tabmargins=[2, 5, 2, 0])
        style.configure("TNotebook.Tab", background="#e0e0e0", font=("Segoe UI", 9),
                       padding=[10, 4], borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", ModernUI.CARD_BG), ("active", "#f0f0f0")],
                  foreground=[("selected", ModernUI.PRIMARY_COLOR), ("active", "#424242")])

        # Progressbar styling
        style.configure("TProgressbar", thickness=8, background=ModernUI.PRIMARY_COLOR)

        # Scrollbar styling
        style.configure("TScrollbar", gripcount=0, background=ModernUI.BG_COLOR, troughcolor="#f0f0f0",
                       borderwidth=0, arrowsize=14)
        style.map("TScrollbar",
                  background=[("pressed", "#c1c1c1"), ("active", "#d6d6d6")])

class AnimatedProgress(ttk.Frame):
    """Custom progress widget with percentage display and animation"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_label = ttk.Label(self, text="0%", background=ModernUI.CARD_BG,
                                      font=("Segoe UI", 8))
        self.progress_label.grid(row=0, column=0, sticky="e", padx=(0, 5))

        self.progressbar = ttk.Progressbar(self, variable=self.progress_var,
                                         mode="determinate", length=400)
        self.progressbar.grid(row=1, column=0, sticky="ew", pady=(2, 5))

        self.status_label = ttk.Label(self, text="Ready",
                                     font=("Segoe UI", 8), foreground="#757575",
                                     background=ModernUI.CARD_BG)
        self.status_label.grid(row=2, column=0, sticky="w")

    def start_indeterminate(self, status_text="Processing..."):
        """Start indeterminate mode with a custom status text"""
        self.progressbar.config(mode="indeterminate")
        self.progressbar.start(10)
        self.progress_label.config(text="")
        self.status_label.config(text=status_text)

    def set_progress(self, value, status_text=None):
        """Set a specific progress value and optional status text"""
        self.progressbar.config(mode="determinate")
        self.progress_var.set(value)
        self.progress_label.config(text=f"{int(value)}%")

        if status_text:
            self.status_label.config(text=status_text)

    def stop(self, status_text="Complete", reset=True):
        """Stop the progress animation"""
        self.progressbar.stop()
        if reset:
            self.progress_var.set(0)
            self.progress_label.config(text="0%")
        self.status_label.config(text=status_text)

class TooltipManager:
    """Helper class to manage tooltips for widgets"""

    def __init__(self, delay=500, wrap_length=250):
        self.delay = delay
        self.wrap_length = wrap_length
        self.tip_window = None
        self.id = None
        self.widget = None

    def show_tip(self, widget, text):
        """Display the tooltip"""
        if self.tip_window or not text:
            return

        x, y, _, _ = widget.bbox("insert")
        x += widget.winfo_rootx() + 25
        y += widget.winfo_rooty() + 25

        self.tip_window = tw = tk.Toplevel(widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(tw, text=text, background="#ffffdd",
                       foreground="black", relief="solid", borderwidth=1,
                       wraplength=self.wrap_length, font=("Segoe UI", 8),
                       justify="left", padx=5, pady=3)
        label.pack()

    def hide_tip(self):
        """Hide the tooltip"""
        if self.tip_window:
            self.tip_window.destroy()
        self.tip_window = None

    def add_tooltip(self, widget, text):
        """Add a tooltip to a widget"""
        self.widget = widget

        def enter(event):
            self.id = widget.after(self.delay, lambda: self.show_tip(widget, text))

        def leave(event):
            if self.id:
                widget.after_cancel(self.id)
                self.id = None
            self.hide_tip()

        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

class ProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whistle Extractor")
        self.root.geometry("800x700")
        self.root.minsize(800, 900)

        # Apply modern styling
        ModernUI.setup_styles()

        # Set app icon if available
        try:
            icon_path = os.path.join(cwd, "assets", "whistle_icon.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass

        # Configure root appearance
        self.root.configure(bg=ModernUI.BG_COLOR)

        # Queue for thread-safe communication
        self.output_queue = queue.Queue()

        # Tooltip manager
        self.tooltip_manager = TooltipManager()

        # Advanced settings state
        self.show_advanced = tk.BooleanVar(value=False)

        # Initialize UI components
        self.setup_ui()

        # Start queue checker
        self.check_output_queue()

        # Setup keyboard shortcuts
        self.setup_keyboard_shortcuts()

        # Log starting time
        self.log_message(f"Application started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"Working directory: {cwd}")
        self.log_message("Ready to process audio files.")

    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts for common actions"""
        self.root.bind("<F5>", lambda e: self.run_process())
        self.root.bind("<Control-r>", lambda e: self.run_process())
        self.root.bind("<Control-i>", lambda e: self.install_dependencies())

    def setup_ui(self):
        # Create main container with padding
        self.main_container = ttk.Frame(self.root, style="TFrame")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Create header
        self.create_header()

        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create main settings tab
        self.settings_frame = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.settings_frame, text="Settings")

        # Create tabs for other functionality
        self.log_frame = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.log_frame, text="Processing Log")

        self.help_frame = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.help_frame, text="Help")

        # Fill the tabs with content
        self.create_settings_tab()
        self.create_log_tab()
        self.create_help_tab()

        # Create footer with action buttons and progress bar
        self.create_footer()

    def create_header(self):
        """Create the app header with title and description"""
        header_frame = ttk.Frame(self.main_container, style="TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 15))

        # App title
        title_label = ttk.Label(header_frame, text="Whistle Extractor", style="Title.TLabel")
        title_label.pack(anchor="w")

        # App description
        description = "Extract and analyze whistle sounds from audio recordings using machine learning"
        subtitle_label = ttk.Label(header_frame, text=description, style="Subtitle.TLabel")
        subtitle_label.pack(anchor="w", pady=(0, 5))

        # Horizontal separator
        separator = ttk.Separator(header_frame, orient="horizontal")
        separator.pack(fill=tk.X, pady=5)

    def create_settings_tab(self):
        """Create the main settings tab with all configuration options"""
        # Create scrollable canvas for settings
        canvas_frame = ttk.Frame(self.settings_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(canvas_frame, bg=ModernUI.BG_COLOR,
                         highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical",
                                command=canvas.yview)

        self.scrollable_frame = ttk.Frame(canvas, style="TFrame")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Fill the scrollable frame with settings
        self.create_file_paths_section()
        self.create_time_settings_section()
        self.create_output_settings_section()
        self.create_advanced_settings_section()

    def create_file_paths_section(self):
        """Create the file paths section in the settings tab"""
        paths_frame = ttk.LabelFrame(self.scrollable_frame, text="File Paths",
                                   padding=10, style="TLabelframe")
        paths_frame.pack(fill=tk.X, padx=10, pady=10)

        # Model path
        self.model_entry = self.add_file_input(
            paths_frame,
            "Model Path:",
            default_model_path,
            lambda: self.browse_file(self.model_entry, "Select Model File",
                                   [("Keras Model", "*.keras;*.h5"), ("All files", "*.*")]),
            "Path to the trained model file (.keras or .h5)"
        )

        # Recordings folder
        self.recordings_entry = self.add_file_input(
            paths_frame,
            "Recordings Folder:",
            default_recordings,
            lambda: self.browse_folder(self.recordings_entry, "Select Recordings Folder"),
            "Folder containing the audio recordings to process"
        )

        # Output folder
        self.saving_folder_entry = self.add_file_input(
            paths_frame,
            "Output Folder:",
            default_saving_folder,
            lambda: self.browse_folder(self.saving_folder_entry, "Select Output Folder"),
            "Folder where processed files will be saved"
        )

    def create_time_settings_section(self):
        """Create the time settings section"""
        time_frame = ttk.LabelFrame(self.scrollable_frame, text="Time Settings",
                                  padding=10, style="TLabelframe")
        time_frame.pack(fill=tk.X, padx=10, pady=10)

        # Configure grid
        time_frame.columnconfigure(1, weight=1)

        # Start time
        ttk.Label(time_frame, text="Start Time (seconds):").grid(
            row=0, column=0, sticky="w", pady=5, padx=5
        )

        self.start_time_entry = ttk.Entry(time_frame)
        self.start_time_entry.insert(0, "0")
        self.start_time_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        self.tooltip_manager.add_tooltip(
            self.start_time_entry,
            "Starting point in seconds for processing the audio file"
        )

        # End time
        ttk.Label(time_frame, text="End Time (seconds):").grid(
            row=1, column=0, sticky="w", pady=5, padx=5
        )

        self.end_time_entry = ttk.Entry(time_frame)
        self.end_time_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # Help text for end time
        help_label = ttk.Label(time_frame, text="Leave empty to process entire file",
                             foreground="#757575")
        help_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=5)

    def create_output_settings_section(self):
        """Create the output settings section"""
        output_frame = ttk.LabelFrame(self.scrollable_frame, text="Output Settings",
                                    padding=10, style="TLabelframe")
        output_frame.pack(fill=tk.X, padx=10, pady=10)

        # Checkboxes for output options
        self.save_var = tk.BooleanVar(value=False)
        self.save_p_var = tk.BooleanVar(value=True)

        self.save_all_checkbutton = ttk.Checkbutton(  # Store the widget as instance variable
            output_frame,
            text="Save all spectrograms images",
            variable=self.save_var
        )
        self.save_all_checkbutton.pack(anchor="w", pady=5)
        self.tooltip_manager.add_tooltip(
            self.save_all_checkbutton, # Use the widget here for tooltip as well
            "Save spectrograms for all processed segments, including negative results"
        )

        self.save_positives_checkbutton = ttk.Checkbutton( # Store the widget as instance variable
            output_frame,
            text="Save positive detections only",
            variable=self.save_p_var
        )
        self.save_positives_checkbutton.pack(anchor="w", pady=5)
        self.tooltip_manager.add_tooltip(
            self.save_positives_checkbutton, # Use the widget here for tooltip as well
            "Save spectrograms only for segments with positive whistle detections"
        )

    def create_advanced_settings_section(self):
        """Create the advanced settings section with collapsible content"""
        # Advanced settings toggle button
        adv_button_frame = ttk.Frame(self.scrollable_frame, style="TFrame")
        adv_button_frame.pack(fill=tk.X, padx=10, pady=(15, 5))

        self.adv_button = ttk.Button(
            adv_button_frame,
            text="▼ Advanced Settings",
            command=self.toggle_advanced_settings,
            style="Link.TButton"
        )
        self.adv_button.pack(anchor="w")

        # Create advanced settings container
        self.advanced_container = ttk.Frame(self.scrollable_frame, style="TFrame")
        self.advanced_container.pack(fill=tk.X, padx=10, pady=5)

        # Initially hide advanced settings
        self.advanced_container.pack_forget()

        # Performance settings
        perf_frame = ttk.LabelFrame(self.advanced_container, text="Performance Settings",
                                  padding=10, style="TLabelframe")
        perf_frame.pack(fill=tk.X, pady=5)

        perf_frame.columnconfigure(1, weight=1)

        # Batch size
        ttk.Label(perf_frame, text="Batch Size:").grid(
            row=0, column=0, sticky="w", pady=5, padx=5
        )

        self.batch_size_entry = ttk.Entry(perf_frame)
        self.batch_size_entry.insert(0, "64")
        self.batch_size_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.tooltip_manager.add_tooltip(
            self.batch_size_entry,
            "Number of samples processed together. Higher values use more memory but process faster"
        )

        # Max workers
        ttk.Label(perf_frame, text="Max Workers:").grid(
            row=1, column=0, sticky="w", pady=5, padx=5
        )

        self.max_workers_entry = ttk.Entry(perf_frame)
        self.max_workers_entry.insert(0, "8")
        self.max_workers_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.tooltip_manager.add_tooltip(
            self.max_workers_entry,
            "Maximum number of parallel processing threads. Higher values use more CPU"
        )

        # Frequency settings
        freq_frame = ttk.LabelFrame(self.advanced_container, text="Frequency Settings",
                                  padding=10, style="TLabelframe")
        freq_frame.pack(fill=tk.X, pady=5)

        freq_frame.columnconfigure(1, weight=1)

        # Cut low frequency
        ttk.Label(freq_frame, text="Cut Low Frequency (kHz):").grid(
            row=0, column=0, sticky="w", pady=5, padx=5
        )

        self.clf_entry = ttk.Entry(freq_frame)
        self.clf_entry.insert(0, "3")
        self.clf_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.tooltip_manager.add_tooltip(
            self.clf_entry,
            "Lower frequency cutoff in kHz. Frequencies below this will be filtered out"
        )

        # Cut high frequency
        ttk.Label(freq_frame, text="Cut High Frequency (kHz):").grid(
            row=1, column=0, sticky="w", pady=5, padx=5
        )

        self.chf_entry = ttk.Entry(freq_frame)
        self.chf_entry.insert(0, "20")
        self.chf_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.tooltip_manager.add_tooltip(
            self.chf_entry,
            "Upper frequency cutoff in kHz. Frequencies above this will be filtered out"
        )

        # Model settings
        model_frame = ttk.LabelFrame(self.advanced_container, text="Model Settings",
                                   padding=10, style="TLabelframe")
        model_frame.pack(fill=tk.X, pady=5)

        model_frame.columnconfigure(1, weight=1)

        # Detection threshold
        ttk.Label(model_frame, text="Detection Threshold:").grid(
            row=0, column=0, sticky="w", pady=5, padx=5
        )

        self.threshold_entry = ttk.Entry(model_frame)
        self.threshold_entry.insert(0, "0.5")
        self.threshold_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.tooltip_manager.add_tooltip(
            self.threshold_entry,
            "Confidence threshold for whistle detection (0.0 to 1.0). Higher values are more selective"
        )

        # Image normalization
        norm_frame = ttk.Frame(model_frame, style="TFrame")
        norm_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)

        self.image_norm_var = tk.BooleanVar(value=False)
        self.image_norm_checkbutton = ttk.Checkbutton( # Store the widget as instance variable
            norm_frame,
            text="Image Normalization (/255)",
            variable=self.image_norm_var
        )
        self.image_norm_checkbutton.pack(side=tk.LEFT)
        ttk.Label(
            norm_frame,
            text="(Leave unchecked unless using a different model)",
            foreground="#757575"
        ).pack(side=tk.LEFT, padx=5)

        # Specific files section
        files_frame = ttk.LabelFrame(self.advanced_container, text="Specific Files",
                                   padding=10, style="TLabelframe")
        files_frame.pack(fill=tk.X, pady=5)

        self.specific_files_var = tk.BooleanVar(value=False)
        self.specific_files_checkbutton = ttk.Checkbutton( # Store the widget as instance variable
            files_frame,
            text="Process specific files only",
            variable=self.specific_files_var,
            command=self.toggle_specific_files
        )
        self.specific_files_checkbutton.pack(anchor="w", pady=5)
        self.tooltip_manager.add_tooltip(
            self.specific_files_checkbutton,
            "Enable to process only the files listed in the specified file list. If disabled, all audio files in the Recordings Folder will be processed."
        )

        # Container for specific files option
        self.specific_files_container = ttk.Frame(files_frame, style="TFrame")
        self.specific_files_container.pack(fill=tk.X, pady=5)

        # Hide container initially
        self.specific_files_container.pack_forget()

        # Add file list input
        self.specific_files_entry = self.add_file_input(
            self.specific_files_container,
            "File List:",
            "",
            lambda: self.browse_file(self.specific_files_entry, "Select File List",
                                   [("Text files", "*.txt"), ("All files", "*.*")]),
            "Path to a text file containing a list of audio filenames to process, one filename per line. Filenames should be relative to the Recordings Folder or absolute paths."
        )

    def create_log_tab(self):
        """Create the log tab with output console"""
        log_container = ttk.Frame(self.log_frame, style="Card.TFrame", padding=10)
        log_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title and controls
        log_header = ttk.Frame(log_container, style="Card.TFrame")
        log_header.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(log_header, text="Processing Log", style="Header.TLabel",
                background=ModernUI.CARD_BG).pack(side=tk.LEFT)

        # Control buttons
        button_frame = ttk.Frame(log_header, style="Card.TFrame")
        button_frame.pack(side=tk.RIGHT)

        # Clear button
        self.clear_log_btn = ttk.Button(
            button_frame,
            text="Clear Log",
            command=self.clear_log,
            style="Secondary.TButton"
        )
        self.clear_log_btn.pack(side=tk.LEFT, padx=5)

        # Copy button
        self.copy_log_btn = ttk.Button(
            button_frame,
            text="Copy Log",
            command=self.copy_log_to_clipboard,
            style="Primary.TButton"
        )
        self.copy_log_btn.pack(side=tk.LEFT)

        # Log text area with custom styling
        self.log_text = scrolledtext.ScrolledText(
            log_container,
            height=20,
            wrap=tk.WORD,
            font=("Consolas", 9),
            background="#f8f8f8",
            foreground="#212121"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Make text read-only
        self.log_text.config(state=tk.DISABLED)

    def create_help_tab(self):
        """Create the help tab with instructions and about info"""
        # Container for help content
        help_container = ttk.Frame(self.help_frame, style="TFrame")
        help_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create notebook for help subcategories
        help_notebook = ttk.Notebook(help_container)
        help_notebook.pack(fill=tk.BOTH, expand=True)

        # Instructions tab
        instructions_frame = ttk.Frame(help_notebook, style="TFrame", padding=10)
        help_notebook.add(instructions_frame, text="Instructions")

        instructions = """
        # How to Use the Whistle Extractor

        This application allows you to analyze audio recordings to detect and extract whistle sounds using a machine learning model.

        ## Basic Steps:

        1. **Setup**:
           - First, click the "Install Dependencies" button if this is your first time running the app.
           - Select a model file (.keras or .h5 format) or use the default.
           - Choose the folder containing your audio recordings.
           - Select where you want to save the processed results.

        2. **Configure Options**:
           - Set the time range to process (leave End Time empty to process the entire file).
           - Choose whether to save all spectrograms or only positive detections.
           - Adjust advanced settings if needed.

        3. **Run**:
           - Click "Start Processing" to begin the analysis.
           - The progress will be shown in the status bar and log.
           - Results will be saved to the specified output folder.

        ## Keyboard Shortcuts:

        - **F5** or **Ctrl+R**: Start processing
        - **Ctrl+I**: Install dependencies

        ---
        **For detailed information about each setting, hover your mouse over the setting label or input field.**
        """
        instructions_label = ttk.Label(instructions_frame, text=instructions, justify=tk.LEFT)
        instructions_label.pack(fill=tk.BOTH, expand=True)

        # About tab
        about_frame = ttk.Frame(help_notebook, style="TFrame", padding=10)
        help_notebook.add(about_frame, text="About")

        about_text = """
        # About Whistle Extractor

        **Version**: 1.0

        **Description**: This application is designed to automatically detect and extract whistle sounds from audio recordings. It utilizes a machine learning model trained to identify whistle patterns in spectrograms.

        **Developed by**: Alexis Emanuelli

        **License**: It's for you baby

        **Disclaimer**: This software is provided as is, without warranty of any kind. Use it at your own risk.

        ---
        **For support or more information, please visit [Your Website/Repository Link (Optional)]**
        """
        about_label = ttk.Label(about_frame, text=about_text, justify=tk.LEFT)
        about_label.pack(fill=tk.BOTH, expand=True)


    def create_footer(self):
        """Create the footer with action buttons and progress bar"""
        footer_frame = ttk.Frame(self.main_container, style="TFrame")
        footer_frame.pack(fill=tk.X, pady=(10, 0))

        # Horizontal separator above footer
        separator = ttk.Separator(self.main_container, orient="horizontal")
        separator.pack(fill=tk.X, pady=(0, 10))

        # Button frame
        button_action_frame = ttk.Frame(footer_frame, style="TFrame")
        button_action_frame.pack(side=tk.LEFT, padx=10)

        # Install Dependencies Button
        self.install_btn = ttk.Button(
            button_action_frame,
            text="Install Dependencies",
            command=self.install_dependencies,
            style="Warning.TButton"
        )
        self.install_btn.pack(side=tk.LEFT, padx=5)
        self.tooltip_manager.add_tooltip(self.install_btn, "Install required Python libraries (run once)")

        # Start Processing Button
        self.run_btn = ttk.Button(
            button_action_frame,
            text="Start Processing",
            command=self.run_process,
            style="Primary.TButton"
        )
        self.run_btn.pack(side=tk.LEFT)
        self.tooltip_manager.add_tooltip(self.run_btn, "Start the whistle extraction process (F5 or Ctrl+R)")

        # Progress bar and status
        self.progress_frame = ttk.Frame(footer_frame, style="Card.TFrame", padding=5)
        self.progress_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10, pady=5)
        self.progress_bar = AnimatedProgress(self.progress_frame)
        self.progress_bar.pack(fill=tk.X, expand=True)

    def add_file_input(self, parent, label_text, default_path, browse_command, tooltip_text):
        """Reusable method to create a labeled entry with browse button for file/folder paths."""
        input_frame = ttk.Frame(parent, style="TFrame")
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(input_frame, text=label_text).pack(side=tk.LEFT, padx=(0, 5))

        entry = ttk.Entry(input_frame)
        entry.insert(0, default_path)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.tooltip_manager.add_tooltip(entry, tooltip_text)

        browse_btn = ttk.Button(input_frame, text="Browse", command=browse_command)
        browse_btn.pack(side=tk.LEFT)

        return entry

    def browse_file(self, entry_widget, title, filetypes):
        """Open file dialog and set entry with selected file path."""
        filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if filepath:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, filepath)

    def browse_folder(self, entry_widget, title):
        """Open folder dialog and set entry with selected folder path."""
        folder_path = filedialog.askdirectory(title=title)
        if folder_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, folder_path)

    def toggle_advanced_settings(self):
        """Toggle visibility of advanced settings frame."""
        self.show_advanced.set(not self.show_advanced.get())
        if self.show_advanced.get():
            self.advanced_container.pack(fill=tk.X, padx=10, pady=5)
            self.adv_button.config(text="▲ Advanced Settings")
        else:
            self.advanced_container.pack_forget()
            self.adv_button.config(text="▼ Advanced Settings")

    def toggle_specific_files(self):
        """Toggle visibility of specific files container."""
        if self.specific_files_var.get():
            self.specific_files_container.pack(fill=tk.X, pady=5)
        else:
            self.specific_files_container.pack_forget()

    def toggle_controls(self, enabled):
        """Enable or disable interactive controls based on processing state."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.run_btn.config(state=state)
        self.install_btn.config(state=state)
        self.model_entry.config(state=state)
        self.recordings_entry.config(state=state)
        self.saving_folder_entry.config(state=state)
        self.start_time_entry.config(state=state)
        self.end_time_entry.config(state=state)
        self.save_all_checkbutton.config(state=state)
        self.save_positives_checkbutton.config(state=state)
        self.image_norm_checkbutton.config(state=state)
        self.specific_files_checkbutton.config(state=state)
        self.batch_size_entry.config(state=state)
        self.max_workers_entry.config(state=state)
        self.clf_entry.config(state=state)
        self.chf_entry.config(state=state)
        self.threshold_entry.config(state=state)
        self.clear_log_btn.config(state=state)
        self.copy_log_btn.config(state=state)
        self.adv_button.config(state=state)
        # Remove the incorrect line:
        # self.notebook.config(state=state)

        # # Disable/Enable notebook tabs instead of notebook itself
        # notebook_state = "disabled" if not enabled else "normal" # ttk uses "disabled" and "normal" for tab states

        # # Get tab indices (0, 1, 2 for "Settings", "Processing Log", "Help")
        # for tab_index in range(self.notebook.index("end")): # Iterate through all tabs
        #     self.notebook.tab(tab_index, state=notebook_state) # Disable/enable each tab

    def validate_inputs(self) -> Optional[dict]:
        """Validate all inputs and return them as a dictionary if valid. 
        Logs errors to the processing log instead of showing messageboxes.
        """
        try:
            # Path validations
            model_path = self.model_entry.get().strip()
            recordings_path = self.recordings_entry.get().strip()
            saving_folder_path = self.saving_folder_entry.get().strip()
            specific_files_path = self.specific_files_entry.get().strip() if self.specific_files_var.get() else ""

            if not os.path.exists(model_path):
                raise ValueError("Model path does not exist.")
            if not os.path.exists(recordings_path):
                raise ValueError("Recordings folder does not exist.")
            if specific_files_path and not os.path.exists(specific_files_path):
                raise ValueError("Specific files list file does not exist.")

            # Time validations
            start_time = int(self.start_time_entry.get().strip() or "0")
            end_time_str = self.end_time_entry.get().strip()
            end_time = int(end_time_str) if end_time_str else None
            if start_time < 0:
                raise ValueError("Start time must be non-negative.")
            if end_time is not None and end_time <= start_time:
                raise ValueError("End time must be greater than start time.")

            # Numeric validations
            batch_size = int(self.batch_size_entry.get().strip())
            max_workers = int(self.max_workers_entry.get().strip())
            clf = int(self.clf_entry.get().strip())
            chf = int(self.chf_entry.get().strip())
            threshold = float(self.threshold_entry.get().strip())

            if batch_size <= 0:
                raise ValueError("Batch size must be a positive integer.")
            if max_workers <= 0:
                raise ValueError("Max workers must be a positive integer.")
            if not 0 <= threshold <= 1:
                raise ValueError("Detection threshold must be between 0.0 and 1.0.")
            if clf < 0:
                raise ValueError("Cut Low Frequency must be non-negative.")
            if chf <= 0:
                raise ValueError("Cut High Frequency must be positive.")
            if clf >= chf:
                raise ValueError("Cut Low Frequency must be less than Cut High Frequency.")

            return {
                'model_path': model_path,
                'recordings': recordings_path,
                'saving_folder': saving_folder_path,
                'start_time': start_time,
                'end_time': end_time,
                'batch_size': batch_size,
                'save': self.save_var.get(),
                'save_p': self.save_p_var.get(),
                'max_workers': max_workers,
                'specific_files_path': specific_files_path,
                'CLF': clf,
                'CHF': chf,
                'image_norm': self.image_norm_var.get(),
                'threshold': threshold
            }
        except ValueError as e:
            error_message = f"Input Error: Invalid input values - {str(e)}" # Create error message
            self.log_message(f"ERROR: {error_message}") # Log the error
            return None # Return None to indicate validation failure
        except Exception as e: # Catch other potential errors during conversion
            error_message = f"Input Error: Unexpected input error - {str(e)}"
            self.log_message(f"ERROR: {error_message}")
            return None
    def run_process(self):
        """Run the main processing function with enhanced user feedback"""
        # Show that we're validating inputs
        self.progress_bar.start_indeterminate("Validating settings...")

        # Validate all inputs
        inputs = self.validate_inputs()
        if not inputs:
            self.progress_bar.stop("Ready")
            return

        # Create the output directory if it doesn't exist
        try:
            output_dir = Path(inputs['saving_folder'])
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
                self.log_message(f"Created output directory: {output_dir}")
        except Exception as e:
            self.progress_bar.stop("Error")
            messagebox.showerror("Folder Error", f"Could not create output folder:\n{str(e)}")
            return

        # Switch to the Processing Log tab to show progress
        self.notebook.select(self.log_frame)

        # Handle specific files
        if inputs['specific_files_path']:
            try:
                with open(inputs['specific_files_path'], 'r') as f:
                    specific_files = [line.strip() for line in f if line.strip()]

                if not specific_files:
                    self.progress_bar.stop("Error")
                    messagebox.showerror("Empty File List", "The specific files list is empty.")
                    return

                inputs['specific_files'] = specific_files
                self.log_message(f"Processing {len(specific_files)} specific files from list")
            except Exception as e:
                self.progress_bar.stop("Error")
                messagebox.showerror("File List Error", f"Could not read file list:\n{str(e)}")
                return
        else:
            inputs['specific_files'] = None

        # Log the processing settings
        self.log_message("=== Starting Processing ===")
        self.log_message(f"Model: {inputs['model_path']}")
        self.log_message(f"Input folder: {inputs['recordings']}")
        self.log_message(f"Output folder: {inputs['saving_folder']}")
        self.log_message(f"Time range: {inputs['start_time']}s to {inputs['end_time'] or 'end'}")

        # Update progress bar status
        self.progress_bar.start_indeterminate("Initializing processing...")

        def process_thread():
            # Disable controls during processing
            self.toggle_controls(False)

            # Progress updater function that can be called from the processing module
            def update_progress(percent, status_text=None):
                self.output_queue.put(("progress", percent))
                if status_text:
                    self.output_queue.put(("status", status_text))
                    self.output_queue.put(status_text)

            # Create a custom writer that queues printed messages for the log
            class QueueWriter:
                def __init__(self, queue_obj):
                    self.queue = queue_obj
                    self.buffer = ""

                def write(self, msg):
                    # Buffer the message until we get a newline
                    self.buffer += msg
                    if '\n' in self.buffer:
                        lines = self.buffer.split('\n')
                        for line in lines[:-1]:
                            if line.strip():  # Only queue non-empty messages
                                self.queue.put(line.strip())
                        self.buffer = lines[-1]

                def flush(self):
                    if self.buffer.strip():
                        self.queue.put(self.buffer.strip())
                        self.buffer = ""

            # Redirect both stdout and stderr
            qw = QueueWriter(self.output_queue)
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = qw
            sys.stderr = qw

            try:
                # Import the processing module here to avoid any import errors
                # affecting the main GUI
                # try:
                #     from ..predict_and_extract_online import process_predict_extract
                # except ImportError as e:
                #     error_msg = (f"Could not import the processing module.\n"
                #                 f"Error: {str(e)}\n\n"
                #                 f"You may need to install dependencies first. "
                #                 f"Click the 'Install Dependencies' button and try again.")
                #     self.output_queue.put(error_msg)
                #     self.root.after(0, messagebox.showerror, "Import Error", error_msg)
                #     return

                # Set up a basic progress reporter for the processing function
                self.output_queue.put("Starting audio processing pipeline...")

                # Call the actual processing function
                process_predict_extract(
                    recording_folder_path=inputs['recordings'],
                    saving_folder=inputs['saving_folder'],
                    cut_low_freq=inputs['CLF'],
                    cut_high_freq=inputs['CHF'],
                    image_normalize=inputs['image_norm'],
                    start_time=inputs['start_time'],
                    end_time=inputs['end_time'],
                    batch_size=inputs['batch_size'],
                    save=inputs['save'],
                    save_positives=inputs['save_p'],
                    model_path=inputs['model_path'],
                    binary_threshold=inputs['threshold'],
                    max_workers=inputs['max_workers'],
                    specific_files=inputs['specific_files']
                )

                # Add a completion message to the log
                self.output_queue.put("\nProcessing completed successfully!")

                # Create a success message with the output path
                success_msg = (f"Processing completed successfully!\n\n"
                            f"Results are saved in:\n{inputs['saving_folder']}")

                # Show the success message with an option to open the folder
                def show_success_with_open_option():
                    result = messagebox.askquestion("Success",
                                        f"{success_msg}\n\nWould you like to open the output folder?",
                                        icon='info')
                    if result == 'yes':
                        try:
                            if sys.platform == 'win32':
                                os.startfile(inputs['saving_folder'])
                            elif sys.platform == 'darwin':  # macOS
                                subprocess.call(['open', inputs['saving_folder']])
                            else:  # Linux
                                subprocess.call(['xdg-open', inputs['saving_folder']])
                        except Exception as e:
                            messagebox.showerror("Error", f"Could not open folder: {str(e)}")

                # Schedule the success dialog to show from the main thread
                self.root.after(0, show_success_with_open_option)

            except Exception as e:
                # Create detailed error message
                import traceback
                error_details = traceback.format_exc()
                error_output = (f"An error occurred during processing:\n\n"
                            f"{str(e)}\n\n"
                            f"See the Processing Log tab for more details.")

                # Log the full error
                self.output_queue.put(f"ERROR: {str(e)}")
                self.output_queue.put(error_details)

                # Show error dialog with options
                def show_error_dialog():
                    error_dialog = tk.Toplevel(self.root)
                    error_dialog.title("Processing Error")
                    error_dialog.geometry("600x400")
                    error_dialog.minsize(500, 300)

                    # Make dialog modal
                    error_dialog.transient(self.root)
                    error_dialog.grab_set()

                    # Dialog content
                    main_frame = ttk.Frame(error_dialog, padding=10)
                    main_frame.pack(fill=tk.BOTH, expand=True)

                    ttk.Label(main_frame, text="An error occurred during processing:",
                            font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 5))

                    ttk.Label(main_frame, text=str(e), foreground="red").pack(anchor="w", pady=(0, 10))

                    # Scrollable error details
                    ttk.Label(main_frame, text="Error details:").pack(anchor="w")

                    details_frame = ttk.Frame(main_frame)
                    details_frame.pack(fill=tk.BOTH, expand=True, pady=5)

                    details_text = scrolledtext.ScrolledText(details_frame, height=10,
                                                        wrap=tk.WORD, font=("Consolas", 9))
                    details_text.pack(fill=tk.BOTH, expand=True)
                    details_text.insert(tk.END, error_details)
                    details_text.config(state=tk.DISABLED)

                    # Button to copy error details
                    def copy_error():
                        self.root.clipboard_clear()
                        self.root.clipboard_append(error_details)
                        self.root.update()

                    buttons_frame = ttk.Frame(main_frame)
                    buttons_frame.pack(fill=tk.X, pady=10)

                    copy_btn = ttk.Button(buttons_frame, text="Copy Error Details",
                                        command=copy_error)
                    copy_btn.pack(side=tk.LEFT, padx=5)

                    close_btn = ttk.Button(buttons_frame, text="Close",
                                        command=error_dialog.destroy)
                    close_btn.pack(side=tk.RIGHT, padx=5)

                # Show the error dialog from the main thread
                self.root.after(0, show_error_dialog)

            finally:
                # Restore original stdout and stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr

                # Reset UI state
                self.root.after(0, self.progress_bar.stop, "Ready")
                self.root.after(0, lambda: self.toggle_controls(True))
                self.root.after(0, lambda: self.log_message("=== Processing finished ==="))

        # Start the processing in a separate thread
        processing_thread = threading.Thread(target=process_thread, daemon=True)
        processing_thread.start()

    def install_dependencies(self):
        """Install dependencies from requirements.txt using pip"""
        requirements_path = "requirements.txt"
        if not os.path.exists(requirements_path):
            messagebox.showerror("Error", "requirements.txt not found!")
            return

        self.notebook.select(self.log_frame) # Switch to log tab
        self.log_message("=== Starting Dependency Installation ===")
        self.log_message("Installing dependencies from requirements.txt...")

        self.progress_bar.start_indeterminate("Installing...")
        self.toggle_controls(False)

        def install_thread():
            error_output = ""
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
                    self.output_queue.put(line) # Queue output for log

                process.wait()
                if process.returncode == 0:
                    self.output_queue.put("\nDependencies installed successfully!")
                    self.output_queue.put("=== Dependency Installation Complete ===")
                    self.root.after(0, messagebox.showinfo, "Success",
                                "Dependencies installed successfully!")
                else:
                    while not self.output_queue.empty():
                        error_output += self.output_queue.get() + "\n"
                    error_output += f"\nInstallation failed with return code: {process.returncode}"
                    self.output_queue.put(error_output) # Log the full error
                    self.output_queue.put("=== Dependency Installation Failed ===")

                    def show_install_error_dialog(error_details):
                        error_dialog = tk.Toplevel(self.root)
                        error_dialog.title("Installation Error")
                        error_dialog.geometry("600x400")
                        error_dialog.minsize(500, 300)
                        error_dialog.transient(self.root)
                        error_dialog.grab_set()

                        main_frame = ttk.Frame(error_dialog, padding=10)
                        main_frame.pack(fill=tk.BOTH, expand=True)

                        ttk.Label(main_frame, text="Error installing dependencies:",
                                font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 5))
                        ttk.Label(main_frame, text="Check the output below for details.",
                                font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 10))


                        details_frame = ttk.Frame(main_frame)
                        details_frame.pack(fill=tk.BOTH, expand=True, pady=5)

                        details_text = scrolledtext.ScrolledText(details_frame, height=10,
                                                            wrap=tk.WORD, font=("Consolas", 9))
                        details_text.pack(fill=tk.BOTH, expand=True)
                        details_text.insert(tk.END, error_details)
                        details_text.config(state=tk.DISABLED)

                        buttons_frame = ttk.Frame(main_frame)
                        buttons_frame.pack(fill=tk.X, pady=10)
                        close_btn = ttk.Button(buttons_frame, text="Close",
                                            command=error_dialog.destroy)
                        close_btn.pack(side=tk.RIGHT, padx=5)
                    self.root.after(0, show_install_error_dialog, error_output)


            except Exception as e:
                error_output = f"An unexpected error occurred during installation:\n{str(e)}"
                self.output_queue.put(error_output) # Log unexpected error
                self.output_queue.put("=== Dependency Installation Failed ===")
                self.root.after(0, messagebox.showerror, "Error", error_output) # Show basic error dialog

            finally:
                self.root.after(0, self.progress_bar.stop, "Ready")
                self.root.after(0, lambda: self.toggle_controls(True))
                self.root.after(0, lambda: self.log_message("=== Dependency Installation finished ==="))

        threading.Thread(target=install_thread, daemon=True).start()

    def check_output_queue(self):
        """Check for messages in the output queue and process them."""
        try:
            while True:
                message = self.output_queue.get_nowait()
                if isinstance(message, tuple) and message[0] == "progress":
                    self.progress_bar.set_progress(message[1])
                elif isinstance(message, tuple) and message[0] == "status":
                    self.progress_bar.stop(status_text=message[1])
                else:
                    self.log_message(message)
        except queue.Empty:
            pass
        self.root.after(100, self.check_output_queue) # Check queue every 100ms

    def log_message(self, message):
        """Append a message to the log text area."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.log_text.config(state=tk.NORMAL) # Enable editing to append
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END) # Autoscroll to the bottom
        self.log_text.config(state=tk.DISABLED) # Disable editing

    def clear_log(self):
        """Clear the log text area."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    def copy_log_to_clipboard(self):
        """Copy the content of the log text area to the clipboard."""
        log_content = self.log_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(log_content)
        self.root.update()
        self.log_message("Log copied to clipboard.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ProcessingGUI(root)
    root.mainloop()