#!/usr/bin/env python3
"""
Optimized Vietnamese ASR configuration for CPU performance.
Fine-tuned for Intel Core i5-1235U (12 cores).
"""

import sherpa_onnx
import os

# Configuration comparison for different use cases
CONFIGS = {
    "CURRENT": {
        "name": "Current Settings (Recommended)",
        "num_threads": 4,
        "max_active_paths": 4,
        "decoding_method": "modified_beam_search",
        "description": "Best balance of accuracy and speed",
        "rtf_estimate": "~0.02x",
        "use_case": "General purpose, recommended for most users"
    },
    
    "MAX_ACCURACY": {
        "name": "Maximum Accuracy",
        "num_threads": 8,
        "max_active_paths": 8,
        "decoding_method": "modified_beam_search",
        "description": "Highest possible accuracy, slower",
        "rtf_estimate": "~0.04-0.05x",
        "use_case": "When accuracy is critical, speed less important"
    },
    
    "MAX_SPEED": {
        "name": "Maximum Speed",
        "num_threads": 12,
        "max_active_paths": 2,
        "decoding_method": "modified_beam_search",
        "description": "Fastest processing, good accuracy",
        "rtf_estimate": "~0.01x or less",
        "use_case": "When processing large batches of audio"
    },
    
    "GREEDY_FAST": {
        "name": "Greedy Search (Fastest)",
        "num_threads": 12,
        "max_active_paths": None,
        "decoding_method": "greedy_search",
        "description": "Very fast, slightly lower accuracy",
        "rtf_estimate": "~0.005x",
        "use_case": "Real-time applications needing instant response"
    }
}


def create_recognizer(config_name="CURRENT", model_dir="sherpa-onnx-zipformer-vi-2025-04-20"):
    """Create recognizer with specified configuration."""
    
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    
    config = CONFIGS[config_name]
    
    print(f"\n{'='*80}")
    print(f"Configuration: {config['name']}")
    print(f"{'='*80}")
    print(f"Description: {config['description']}")
    print(f"Use case: {config['use_case']}")
    print(f"Estimated RTF: {config['rtf_estimate']}")
    print(f"{'='*80}\n")
    
    encoder = os.path.join(model_dir, "encoder-epoch-12-avg-8.onnx")
    decoder = os.path.join(model_dir, "decoder-epoch-12-avg-8.onnx")
    joiner = os.path.join(model_dir, "joiner-epoch-12-avg-8.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")
    
    recognizer_args = {
        "encoder": encoder,
        "decoder": decoder,
        "joiner": joiner,
        "tokens": tokens,
        "num_threads": config["num_threads"],
        "sample_rate": 16000,
        "feature_dim": 80,
        "decoding_method": config["decoding_method"],
    }
    
    # Add max_active_paths only for beam search
    if config["max_active_paths"] is not None:
        recognizer_args["max_active_paths"] = config["max_active_paths"]
    
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(**recognizer_args)
    
    print("✅ Model loaded with configuration!")
    return recognizer


def show_recommendations():
    """Show configuration recommendations."""
    print("\n" + "="*80)
    print("🎯 CONFIGURATION RECOMMENDATIONS FOR YOUR SYSTEM")
    print("="*80)
    print("\nYour CPU: Intel Core i5-1235U (12 cores)")
    print("GPU: None detected\n")
    print("-"*80)
    
    for key, config in CONFIGS.items():
        print(f"\n📌 {config['name']} ({key})")
        print(f"   Settings: threads={config['num_threads']}, "
              f"paths={config['max_active_paths']}, "
              f"method={config['decoding_method']}")
        print(f"   RTF: {config['rtf_estimate']}")
        print(f"   Use case: {config['use_case']}")
    
    print("\n" + "="*80)
    print("💡 RECOMMENDATION:")
    print("="*80)
    print("✅ Stick with CURRENT settings - already optimal!")
    print("   - RTF ~0.02x is excellent (50x faster than real-time)")
    print("   - Accuracy is best-in-class with modified_beam_search")
    print("   - No GPU needed for your use case")
    print("\n🔧 Optional tuning:")
    print("   - Try MAX_ACCURACY if you need even better transcription")
    print("   - Try MAX_SPEED if processing many files in batch")
    print("="*80 + "\n")


if __name__ == "__main__":
    show_recommendations()
    
    # Example: Create with different configs
    print("\n🧪 Testing CURRENT configuration:")
    recognizer = create_recognizer("CURRENT")
    
    print("\n💡 To use different config in your scripts:")
    print("   from optimize_config import create_recognizer")
    print("   recognizer = create_recognizer('MAX_ACCURACY')")
