#!/usr/bin/env python3
"""
Quick demo script for PhoWhisper-base
Sử dụng microphone hoặc tạo file audio mẫu để test
"""

import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")


def quick_test():
    """Quick test with model info"""
    print("="*60)
    print("PhoWhisper-base Quick Test")
    print("="*60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Device available: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Check if transformers is available
    print("\n✓ Transformers library: OK")
    
    print("\nĐể test model, bạn cần:")
    print("1. Chuẩn bị file audio tiếng Việt (wav, mp3, m4a, etc.)")
    print("2. Chạy: python test_phowhisper.py <path-to-audio-file>")
    
    print("\nVí dụ:")
    print("  python test_phowhisper.py recording.wav")
    print("  python test_phowhisper.py audio.mp3 --device cuda")
    
    print("\n" + "="*60)
    print("Nếu chưa có file audio, bạn có thể:")
    print("- Ghi âm bằng phone rồi chuyển vào máy")
    print("- Tải file audio tiếng Việt từ internet")
    print("- Sử dụng script record_audio.py để ghi âm trực tiếp")
    print("="*60)


if __name__ == "__main__":
    try:
        quick_test()
    except ImportError as e:
        print(f"\n❌ Lỗi: Chưa cài đặt đủ dependencies")
        print(f"   {e}")
        print("\nVui lòng chạy: pip install -r requirements.txt")
