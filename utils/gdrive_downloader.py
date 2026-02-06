"""
Google Drive Download Utility

Downloads dataset files from Google Drive if they don't exist locally.
Uses gdown library for downloading public Google Drive files.
"""

import os
import sys
from typing import Optional

def download_from_gdrive(
    file_id: str,
    output_path: str,
    quiet: bool = False
) -> bool:
    """
    Download a file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        output_path: Path to save the file
        quiet: Suppress output
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import gdown
    except ImportError:
        print("Installing gdown for Google Drive downloads...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown
    
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        if not quiet:
            print(f"  Downloading from Google Drive: {file_id[:10]}...")
        gdown.download(url, output_path, quiet=quiet)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"  Error downloading file: {e}")
        return False


def ensure_dataset_files(
    data_dir: str,
    gdrive_files: dict,
    force_download: bool = False
) -> bool:
    """
    Ensure dataset files exist, downloading from Google Drive if needed.
    
    Args:
        data_dir: Directory where files should be stored
        gdrive_files: Dict mapping filename to Google Drive file ID
        force_download: Re-download even if files exist
        
    Returns:
        bool: True if all files available, False otherwise
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    all_success = True
    
    for filename, file_id in gdrive_files.items():
        filepath = os.path.join(data_dir, filename)
        
        # Check if file exists
        if os.path.exists(filepath) and not force_download:
            print(f"  [OK] {filename} exists locally")
            continue
        
        # Download from Google Drive
        print(f"  [DOWNLOAD] {filename} not found, downloading from Google Drive...")
        success = download_from_gdrive(file_id, filepath)
        
        if success:
            # Verify file size
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  [OK] Downloaded {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  [FAIL] Failed to download {filename}")
            all_success = False
    
    return all_success


def check_and_download_data(config_class) -> bool:
    """
    Check for dataset files and download if needed.
    
    Args:
        config_class: Config class with DATA_DIR and GDRIVE_FILES attributes
        
    Returns:
        bool: True if all files available
    """
    data_dir = getattr(config_class, 'DATA_DIR', 'dataset/MOOCRadar')
    gdrive_files = getattr(config_class, 'GDRIVE_FILES', {})
    
    if not gdrive_files:
        print("  No Google Drive files configured.")
        return True
    
    print("\n[Checking Dataset Files]")
    return ensure_dataset_files(data_dir, gdrive_files)


if __name__ == "__main__":
    # Test the download functionality
    from config import Config
    check_and_download_data(Config)
