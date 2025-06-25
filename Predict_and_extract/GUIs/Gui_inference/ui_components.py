import tkinter as tk
from tkinter import ttk
from style_ui import ModernUI  # Import ModernUI

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

    # ... (rest of the AnimatedProgress class methods)


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
