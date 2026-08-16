"""
cull_gui.py — GUI entry point for Auto-Culling (CustomTkinter).

Launches the desktop interface (no CLI arguments needed). The GUI mirrors
the cull_photos.py CLI: select a photo directory, tweak the scoring
parameters, watch live progress, and review per-image results.
"""

import sys
import tkinter as tk


def main() -> int:
    import customtkinter as ctk
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    from cull.gui.app import CullApp
    CullApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())