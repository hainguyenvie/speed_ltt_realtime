#!/usr/bin/env python3
"""
Download the Zipformer-30M-RNNT-6000h model from Hugging Face.
"""

from huggingface_hub import hf_hub_download
import os

def download_zipformer_model():
    """Download all required model files from Hugging Face."""
    
    repo_id = "hynt/Zipformer-30M-RNNT-6000h"
    local_dir = "zipformer-30m-rnnt-6000h"
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # List of files to download
    files_to_download = [
        "encoder-epoch-20-avg-10.onnx",
        "decoder-epoch-20-avg-10.onnx", 
        "joiner-epoch-20-avg-10.onnx",
        "bpe.model",
    ]
    
    print(f"Downloading Zipformer model from {repo_id}...")
    print(f"Saving to: {os.path.abspath(local_dir)}\n")
    
    for filename in files_to_download:
        # Check if file already exists
        local_path = os.path.join(local_dir, filename)
        
        if os.path.exists(local_path):
            print(f"✓ {filename} - Already exists, skipping")
            continue
        
        print(f"↓ Downloading {filename}...")
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            print(f"  → Saved to: {downloaded_path}")
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")
            return False
    
    print("\n" + "="*80)
    print("✓ Model download complete!")
    print("="*80)
    print(f"\nModel files saved to: {os.path.abspath(local_dir)}")
    print("\nYou can now run:")
    print("  python test_zipformer.py recording.wav")
    print("="*80)
    
    return True


if __name__ == "__main__":
    try:
        success = download_zipformer_model()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
