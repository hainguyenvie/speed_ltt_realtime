#!/usr/bin/env python3
"""
Test Zipformer-30M-RNNT-6000h model using sherpa.OfflineRecognizer with JIT Script.
This is the official way to use this model as shown in the HF Space demo.
"""

import sherpa
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
        print("Usage: python test_zipformer_sherpa.py <audio_file.wav>")
        print("\nExample:")
        print("  python test_zipformer_sherpa.py recording.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file '{audio_file}' not found!")
        sys.exit(1)
    
    model_dir = "zipformer-30m-rnnt-6000h"
    
    print("\n" + "="*80)
    print("🚀 Zipformer-30M-RNNT-6000h Vietnamese Speech Recognition")
    print("="*80)
    print(f"Model: {model_dir}")
    print(f"Audio: {audio_file}\n")
    
    # Model files
    jit_script_model = os.path.join(model_dir, "jit_script.pt")
    tokens_path = os.path.join(model_dir, "tokens.txt")
    
    # Check if files exist
    if not os.path.exists(jit_script_model):
        print(f"❌ Error: JIT Script model not found: {jit_script_model}")
        sys.exit(1)
    
    if not os.path.exists(tokens_path):
        print(f"❌ Error: Tokens file not found: {tokens_path}")
        sys.exit(1)
    
    print("Loading model...")
    try:
        # Create recognizer using sherpa (not sherpa-onnx)
        # This is the official way as shown in HF Space demo
        recognizer = sherpa.OfflineRecognizer(
            tokens=tokens_path,
            nn_model=jit_script_model,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            provider="cpu",
            decoding_method="greedy_search",
        )
        print("✅ Model loaded successfully!")
        print(f"  • Sample rate: 16000 Hz")
        print(f"  • Threads: 4")
        print(f"  • Decoding: greedy_search")
        print(f"  • Provider: CPU\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Read audio
    print("Reading audio...")
    try:
        sample_rate, samples = read_wav_file(audio_file)
        duration = len(samples) / sample_rate
        print(f"✅ Audio loaded")
        print(f"  • Duration: {duration:.2f}s")
        print(f"  • Sample rate: {sample_rate} Hz")
        print(f"  • Samples: {len(samples)}\n")
    except Exception as e:
        print(f"❌ Error reading audio: {e}")
        sys.exit(1)
   
    # Create stream and process
    print("Processing audio...")
    import time
    start_time = time.time()
    
    try:
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(stream)
        
        # Get result
        result = stream.result.text
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Processing complete in {elapsed_time:.2f}s")
        print(f"  • Speed: {duration/elapsed_time:.2f}x real-time\n")
        
        print("="*80)
        print("📝 TRANSCRIPTION RESULT")
        print("="*80)
        print(result)
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during decoding: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
