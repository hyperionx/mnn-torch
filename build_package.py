#!/usr/bin/env python3
"""
Build script for MNN-Torch PyPI package.

This script helps build and validate the package for PyPI publication.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def clean_build_dirs():
    """Clean build directories."""
    dirs_to_clean = ["build", "dist", "src/mnn_torch.egg-info"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"🧹 Cleaning {dir_name}...")
            shutil.rmtree(dir_name)

def check_package_structure():
    """Check if package structure is correct."""
    print("\n🔍 Checking package structure...")
    
    required_files = [
        "pyproject.toml",
        "README.md", 
        "LICENSE",
        "src/mnn_torch/__init__.py",
        "src/mnn_torch/models.py",
        "src/mnn_torch/devices.py",
        "src/mnn_torch/effects.py",
        "src/mnn_torch/layers.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ Package structure looks good")
    return True

def build_package():
    """Build the package."""
    print("\n📦 Building package...")
    
    # Clean previous builds
    clean_build_dirs()
    
    # Build wheel and source distribution
    if not run_command([sys.executable, "-m", "build"], "Building package"):
        return False
    
    return True

def check_package():
    """Check the built package."""
    print("\n🔍 Checking built package...")
    
    # Check wheel
    if not run_command([sys.executable, "-m", "twine", "check", "dist/*"], "Checking package"):
        return False
    
    return True

def main():
    """Main build process."""
    print("🚀 MNN-Torch Package Builder")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ pyproject.toml not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Check package structure
    if not check_package_structure():
        sys.exit(1)
    
    # Build package
    if not build_package():
        sys.exit(1)
    
    # Check package
    if not check_package():
        sys.exit(1)
    
    print("\n🎉 Package build completed successfully!")
    print("\nNext steps:")
    print("1. Review the built files in the 'dist' directory")
    print("2. Test the package: pip install dist/mnn_torch-*.whl")
    print("3. Upload to PyPI: python -m twine upload dist/*")
    print("\nBuilt files:")
    if os.path.exists("dist"):
        for file in os.listdir("dist"):
            print(f"  - dist/{file}")

if __name__ == "__main__":
    main()
