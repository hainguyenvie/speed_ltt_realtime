#!/usr/bin/env python3
"""
WORKING TEST for Zipformer-30M-RNNT-6000h using sherpa_onnx with config.json.
Based on HF Space implementation.
"""

import sherpa_onnx
import time
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_zipformer_working.py <audio_file.wav>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_dir = "zipformer-30m-rnnt-6000h"
    
    print("\n" + "="*80)
    print("🚀 Zipformer-30M-RNNT-6000h Vietnamese STT - WORKING VERSION")
    print("="*80)
    print(f"Audio: {audio_file}\n")
    
    # Use INT8 models (faster) with config.json for tokens
    encoder = os.path.join(model_dir, "encoder-epoch-20-avg-10.int8.onnx")
    decoder = os.path.join(model_dir, "decoder-epoch-20-avg-10.int8.onnx")
    joiner = os.path.join(model_dir, "joiner-epoch-20-avg-10.int8.onnx")
    tokens = os.path.join(model_dir, "config.json")  # Use config.json, not tokens.txt!
    
    print("Loading model...")
    print(f"  Encoder: {encoder}")
    print(f"  Decoder: {decoder}")
    print(f"  Joiner: {joiner}")
    print(f"  Tokens: {tokens}\n")
    
    try:
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
        print("✅ Model loaded successfully!\n")
        
        # Create stream and decode
        print("Processing audio...")
        start_time = time.time()
        
        stream = recognizer.create_stream()
        
        # Read audio file
        import wave
        import numpy as np
        
        with wave.open(audio_file, 'rb') as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            samples = wf.readframes(num_frames)
            
            samples_int16 = np.frombuffer(samples, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32) / 32768.0
            
            if channels == 2:
                samples_float32 = samples_float32.reshape(-1, 2).mean(axis=1)
            
            duration = len(samples_float32) / sample_rate
        
        stream.accept_waveform(sample_rate, samples_float32)
        recognizer.decode_stream(stream)
        
        result = stream.result.text
        elapsed = time.time() - start_time
        
        print(f"✅ Done in {elapsed:.2f}s ({duration/elapsed:.2f}x realtime)\n")
        
        print("="*80)
        print("📝 TRANSCRIPTION RESULT:")
        print("="*80)
        print(result if result else "(empty)")
        print("="*80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
