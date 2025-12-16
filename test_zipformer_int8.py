#!/usr/bin/env python3
"""
Test Zipformer model with INT8 quantized ONNX files using sherpa-onnx.
"""

import sherpa_onnx
import wave
import numpy as np
import sys
import os


def read_wav_file(filename):
    """Read a WAV file and return sample rate and samples as float32 array."""
    with wave.open(filename, 'rb') as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        samples = wf.readframes(num_frames)
        
        # Convert to int16 array
        samples_int16 = np.frombuffer(samples, dtype=np.int16)
        
        # Convert to float32 and normalize
        samples_float32 = samples_int16.astype(np.float32) / 32768.0
        
        # Convert stereo to mono if needed
        if channels == 2:
            samples_float32 = samples_float32.reshape(-1, 2).mean(axis=1)
        
        return sample_rate, samples_float32


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_zipformer_int8.py <audio_file.wav>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found!")
        sys.exit(1)
    
    model_dir = "zipformer-30m-rnnt-6000h"
    
    print(f"\n🚀 Testing Zipformer INT8 Quantized Model")
    print(f"Model dir: {model_dir}")
    print(f"Audio file: {audio_file}\n")
    
    # Use INT8 quantized models
    encoder_path = os.path.join(model_dir, "encoder-epoch-20-avg-10.int8.onnx")
    decoder_path = os.path.join(model_dir, "decoder-epoch-20-avg-10.int8.onnx")
    joiner_path = os.path.join(model_dir, "joiner-epoch-20-avg-10.int8.onnx")
    tokens_path = os.path.join(model_dir, "tokens.txt")
    
    print("Loading model...")
    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            tokens=tokens_path,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Read audio
    print("Reading audio...")
    sample_rate, samples = read_wav_file(audio_file)
    duration = len(samples) / sample_rate
    print(f"✅ Audio loaded ({duration:.2f}s, {sample_rate}Hz)\n")
    
    # Create stream and process
    print("Processing audio...")
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    
    recognizer.decode_stream(stream)
    
    # Get result
    result = stream.result.text
    
    print("\n" + "="*80)
    print("📝 TRANSCRIPTION RESULT")
    print("="*80)
    print(result)
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
