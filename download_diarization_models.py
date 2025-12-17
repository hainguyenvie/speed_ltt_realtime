#!/usr/bin/env python3
"""
Download speaker diarization models for sherpa-onnx.
Includes speaker segmentation and speaker embedding models.
"""

import os
import tarfile
import urllib.request


def download_and_extract(url, target_dir):
    """Download and extract tar.bz2 file."""
    filename = url.split('/')[-1]
    filepath = os.path.join(target_dir, filename)
    
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)
    
    print(f"Extracting {filename}...")
    with tarfile.open(filepath, 'r:bz2') as tar:
        tar.extractall(target_dir)
    
    print(f"Removing {filename}...")
    os.remove(filepath)
    
    print(f"✅ {filename} extracted!\n")


def main():
    print("="*80)
    print("Speaker Diarization Models Download")
    print("="*80 + "\n")
    
    # Create directory
    diarization_dir = "speaker-diarization-models"
    os.makedirs(diarization_dir, exist_ok=True)
    
    print("Downloading models for speaker diarization...\n")
    
    # 1. Speaker Segmentation Model (pyannote-based)
    # From: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
    segmentation_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    
    print("1. Speaker Segmentation Model (pyannote-segmentation-3-0)")
    print(f"   URL: {segmentation_url}")
    download_and_extract(segmentation_url, diarization_dir)
    
    # 2. Speaker Embedding Model (3D-Speaker)
    # From: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
    embedding_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    
    print("2. Speaker Embedding Model (3D-Speaker)")
    print(f"   URL: {embedding_url}")
    
    embedding_file = os.path.join(diarization_dir, "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
    urllib.request.urlretrieve(embedding_url, embedding_file)
    print("✅ 3D-Speaker model downloaded!\n")
    
    # Also download INT8 version for faster inference
    embedding_int8_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.int8.onnx"
    
    print("3. Speaker Embedding Model INT8 (3D-Speaker)")
    print(f"   URL: {embedding_int8_url}")
    
    embedding_int8_file = os.path.join(diarization_dir, "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.int8.onnx")
    urllib.request.urlretrieve(embedding_int8_url, embedding_int8_file)
    print("✅ 3D-Speaker INT8 model downloaded!\n")
    
    print("="*80)
    print("Download Complete!")
    print("="*80)
    print(f"\nModels saved to: {os.path.abspath(diarization_dir)}")
    print("\nDirectory structure:")
    
    for root, dirs, files in os.walk(diarization_dir):
        level = root.replace(diarization_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            size_mb = os.path.getsize(os.path.join(root, file)) / (1024*1024)
            print(f'{subindent}{file} ({size_mb:.1f} MB)')
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Run: python transcript_with_speakers.py audio.wav")
    print("  2. See multi-speaker transcript with labels!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
