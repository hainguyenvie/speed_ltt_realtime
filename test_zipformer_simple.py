#!/usr/bin/env python3
"""
Simple test script for Zipformer-30M-RNNT-6000h model using sherpa-onnx.
Vietnamese speech recognition model optimized for fast CPU inference.
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
    # Check if audio file is provided
    if len(sys.argv) < 2:
        print("Usage: python test_zipformer_simple.py <audio_file.wav>")
        print("\nExample:")
        print("  python test_zipformer_simple.py recording.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found!")
        sys.exit(1)
    
    model_dir = "zipformer-30m-rnnt-6000h"
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found!")
        print("\nPlease run: python download_zipformer.py")
        sys.exit(1)
    
    print(f"\nLoading Zipformer model from '{model_dir}'...")
    print(f"Testing with audio file: {audio_file}\n")
    
    # Model files
    encoder_path = os.path.join(model_dir, "encoder-epoch-20-avg-10.onnx")
    decoder_path = os.path.join(model_dir, "decoder-epoch-20-avg-10.onnx")
    joiner_path = os.path.join(model_dir, "joiner-epoch-20-avg-10.onnx")
    tokens_path = os.path.join(model_dir, "tokens.txt")
    
    # Check if model files exist
    for path in [encoder_path, decoder_path, joiner_path, tokens_path]:
        if not os.path.exists(path):
            print(f"Error: Model file not found: {path}")
            print("\nPlease run:")
            print("  python download_zipformer.py")
            print("  python convert_bpe_to_tokens.py")
            sys.exit(1)
    
    # Create offline recognizer
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
    
    print(f"✓ Model loaded successfully!")
    print(f"  Sample rate: 16000 Hz")
    print(f"  Num threads: 4")
    print(f"  Decoding method: greedy_search\n")
    
    # Read audio file
    print(f"Reading audio file...")
    sample_rate, samples = read_wav_file(audio_file)
    duration = len(samples) / sample_rate
    
    print(f"✓ Audio file loaded")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Samples: {len(samples)}\n")
    
    # Create a stream and feed audio
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    
    print(f"Processing audio...")
    
    # Decode the audio
    recognizer.decode_stream(stream)
    
    # Get the result
    result = stream.result.text
    
    print("\n" + "="*80)
    print("TRANSCRIPTION RESULT")
    print("="*80)
    print(result)
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
