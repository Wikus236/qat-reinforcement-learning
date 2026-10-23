"""Paths anchored to the experiment folder, independent of the working directory."""
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
