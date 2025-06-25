import tkinter as tk
from tkinter import ttk

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
        # ... (rest of the setup_styles method from your original file)
        style.configure("TScrollbar", gripcount=0, background=ModernUI.BG_COLOR, troughcolor="#f0f0f0",
                       borderwidth=0, arrowsize=14)
        style.map("TScrollbar",
                  background=[("pressed", "#c1c1c1"), ("active", "#d6d6d6")])