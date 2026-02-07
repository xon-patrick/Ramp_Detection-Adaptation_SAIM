#!/usr/bin/env python3
"""
Requirements installer for YOLOv8 training
Run this once to install necessary dependencies
"""

import subprocess
import sys

def install_requirements():
    """Install required packages for YOLOv8 training."""
    
    requirements = [
        "ultralytics>=8.0.0",  # YOLOv8 main package
        "torch>=2.0.0",         # PyTorch deep learning framework
        "torchvision>=0.15.0",  # Computer vision utilities
        "opencv-python>=4.7.0", # Image processing
        "numpy>=1.23.0",        # Numerical operations
        "matplotlib>=3.7.0",    # Plotting
        "scikit-learn>=1.2.0",  # ML metrics
        "pyyaml>=6.0",          # YAML parsing
        "Pillow>=9.0.0",        # Image handling
    ]
    
    print("="*60)
    print("Installing YOLOv8 Training Dependencies")
    print("="*60)
    print(f"\nPython version: {sys.version}")
    print(f"Python executable: {sys.executable}\n")
    
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\n" + "="*60)
    print("✓ All dependencies installed successfully!")
    print("="*60)
    print("\nYou can now run: python train.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        install_requirements()
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
