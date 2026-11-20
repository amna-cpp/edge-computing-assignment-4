"""Shared utilities for the Edge Computing & Intelligent Systems assignment 4."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
ONNX_DIR = ROOT / "onnx_models"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
