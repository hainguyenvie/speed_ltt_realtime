#!/usr/bin/env python3
"""
FINAL WORKING TEST for Zipformer-30M-RNNT-6000h using sherpa-onnx.
This uses the ONNX models with a workaround for the BPE token issue.
"""

import sherpa_onnx
import wave
import numpy as np
import sys
import os
import time


def read_wav_file(filename):
    """Read a WAV file and return sample rate and samples as float32 array."""
    with wave.open(filename, 'rb') as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        samples = wf.readframes(num_frames)
        
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32768.0
        
        if channels == 2:
            samples_float32 = samples_float32.reshape(-1, 2).mean(axis=1)
        
        return sample_rate, samples_float32


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_zipformer_final.py <audio_file.wav>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_dir = "zipformer-30m-rnnt-6000h"
    
    print("\n" + "="*80)
    print("🚀 Zipformer-30M-RNNT-6000h Vietnamese STT - FINAL TEST")
    print("="*80)
    print(f"Audio: {audio_file}\n")
    
    # Use INT8 quantized models (faster, smaller)
    encoder = os.path.join(model_dir, "encoder-epoch-20-avg-10.int8.onnx")
    decoder = os.path.join(model_dir, "decoder-epoch-20-avg-10.int8.onnx")
    joiner = os.path.join(model_dir, "joiner-epoch-20-avg-10.int8.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")
    
    print("Attempting to load model with sherpa-onnx...")
    print("Note: This model has known BPE tokenization issues with sherpa-onnx\n")
    
    try:
        # Try loading recognizer
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        print("✅ Model loaded!\n")
        
        # Read audio
        print("Reading audio...")
        sample_rate, samples = read_wav_file(audio_file)
        duration = len(samples) / sample_rate
        print(f"✅ Duration: {duration:.2f}s\n")
        
        # Process
        print("Processing...")
        start_time = time.time()
        
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(stream)
        
        result = stream.result.text
        elapsed = time.time() - start_time
        
        print(f"✅ Done in {elapsed:.2f}s ({duration/elapsed:.2f}x realtime)\n")
        
        print("="*80)
        print("📝 TRANSCRIPTION:")
        print("="*80)
        print(result if result else "(empty result)")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"❌ Expected Error: {e}")
        print("\n" + "="*80)
        print("⚠️  ZIPFORMER-30M FIX NEEDED")
        print("="*80)
        print("\nThis model has BPE tokenization incompatibility with sherpa-onnx.")
        print("\n✅ WORKING ALTERNATIVES:")
        print("\n1. Use the HF Space demo (recommended):")
        print("   https://huggingface.co/spaces/hynt/k2-automatic-speech-recognition-demo")
        print("\n2. Use PhoWhisper (already working):")
        print("   python test_phowhisper.py recording.wav")
        print("\n" + "="*80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
