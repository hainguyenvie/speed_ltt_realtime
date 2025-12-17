#!/usr/bin/env python3
"""Quick test to verify the new Vietnamese model loads correctly."""

import sherpa_onnx
import os
import sys

def test_model_loading():
    model_dir = "sherpa-onnx-zipformer-vi-2025-04-20"
    
    print("🧪 Testing Sherpa-ONNX Vietnamese Model...")
    print(f"Model directory: {model_dir}\n")
    
    # Check if model files exist
    required_files = [
        "encoder-epoch-12-avg-8.onnx",
        "decoder-epoch-12-avg-8.onnx",
        "joiner-epoch-12-avg-8.onnx",
        "tokens.txt"
    ]
    
    print("📁 Checking model files:")
    for file in required_files:
        filepath = os.path.join(model_dir, file)
        exists = os.path.exists(filepath)
        status = "✅" if exists else "❌"
        size = f"({os.path.getsize(filepath) / 1024 / 1024:.1f} MB)" if exists else ""
        print(f"  {status} {file} {size}")
        if not exists:
            print(f"\n❌ Error: Missing file {file}")
            return False
    
    print("\n🔄 Loading model...")
    try:
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
            sample_rate=16000,
            feature_dim=80,
            decoding_method="modified_beam_search",
            max_active_paths=4,
        )
        
        print("✅ Model loaded successfully!\n")
        
        # Test with sample audio if available
        test_wav = os.path.join(model_dir, "test_wavs", "0.wav")
        if os.path.exists(test_wav):
            print(f"🎵 Testing with sample audio: {test_wav}")
            
            stream = recognizer.create_stream()
            
            # Simple test - just verify it can create a stream
            print("✅ Stream created successfully!")
            print("\n✅ All tests passed! Model is ready to use.")
        else:
            print("⚠️  No test audio found, but model loaded successfully.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
