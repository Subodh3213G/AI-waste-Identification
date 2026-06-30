"""CLI launcher for EcoSort AI.

Usage:
    python run.py
"""
import subprocess
import sys
import os

APP_PATH = os.path.join(os.path.dirname(__file__), "app", "main.py")

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", APP_PATH,
         "--server.headless=true",
         "--browser.gatherUsageStats=false"],
    )
