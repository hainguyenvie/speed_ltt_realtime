#!/usr/bin/env python3
"""
Quick test script - record 5 seconds and transcribe with new Vietnamese model.
Usage: python3 quick_test_vietnamese.py
"""

import sherpa_onnx
import sounddevice as sd
import numpy as np
import os
import time


def quick_test():
    """Quick 5-second recording test."""
    model_dir = "sherpa-onnx-zipformer-vi-2025-04-20"
    sample_rate = 16000
    duration = 5
    
    print("🇻🇳 Quick Vietnamese Speech Test")
    print("="*60)
    
    # Load model
    print("🔄 Loading model...")
    encoder = os.path.join(model_dir, "encoder-epoch-12-avg-8.onnx")
    decoder = os.path.join(model_dir, "decoder-epoch-12-avg-8.onnx")
    joiner = os.path.join(model_dir, "joiner-epoch-12-avg-8.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")
    
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        num_threads=4,
        sample_rate=sample_rate,
        feature_dim=80,
        decoding_method="modified_beam_search",
        max_active_paths=4,
    )
    print("✅ Model loaded!\n")
    
    # Record
    print(f"🎤 Recording {duration} seconds... Speak now!")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    
    for i in range(duration, 0, -1):
        print(f"   ⏱️  {i}...", end='\r')
        sd.wait(1000)
    
    sd.wait()
    print("\n✅ Recording complete!")
    
    # Transcribe
    print("🔄 Transcribing...")
    start_time = time.time()
    
    samples = recording.flatten()
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    
    result = stream.result.text.strip()
    elapsed = time.time() - start_time
    
    print(f"⚡ Processed in {elapsed:.2f}s")
    print("\n" + "="*60)
    print("📝 RESULT:")
    print("="*60)
    if result:
        print(f"   {result}")
    else:
        print("   (No speech detected)")
    print("="*60)


if __name__ == "__main__":
    try:
        quick_test()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
