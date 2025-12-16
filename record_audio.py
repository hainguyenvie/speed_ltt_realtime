#!/usr/bin/env python3
"""
Record audio from microphone for testing PhoWhisper
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import argparse


def record_audio(filename="recording.wav", duration=5, sample_rate=16000):
    """
    Ghi âm từ microphone
    
    Args:
        filename: Tên file output
        duration: Thời gian ghi âm (giây)
        sample_rate: Tần số mẫu (Hz)
    """
    print(f"Đang ghi âm trong {duration} giây...")
    print("Hãy nói tiếng Việt vào microphone...")
    
    # Record audio
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    sd.wait()  # Wait until recording is finished
    
    # Save to file
    sf.write(filename, audio_data, sample_rate)
    print(f"✓ Đã lưu file: {filename}")
    
    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Ghi âm từ microphone để test PhoWhisper"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="recording.wav",
        help="Tên file output (mặc định: recording.wav)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Thời gian ghi âm (giây, mặc định: 5)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Tần số mẫu (Hz, mặc định: 16000)"
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Tự động transcribe sau khi ghi âm"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vinai/PhoWhisper-base",
        help="Model để transcribe (mặc định: vinai/PhoWhisper-base)"
    )
    
    args = parser.parse_args()
    
    # Record
    filename = record_audio(args.output, args.duration, args.sample_rate)
    
    # Transcribe if requested
    if args.transcribe:
        print("\nĐang transcribe...")
        from test_phowhisper import transcribe_audio
        text = transcribe_audio(filename, model_name=args.model)
        print("\n" + "="*50)
        print("TRANSCRIPTION:")
        print("="*50)
        print(text)
        print("="*50)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\n❌ Lỗi: Chưa cài đặt sounddevice")
        print(f"   {e}")
        print("\nVui lòng chạy: pip install sounddevice soundfile")
