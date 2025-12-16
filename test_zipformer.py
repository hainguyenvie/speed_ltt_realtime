#!/usr/bin/env python3
"""
Test script for Zipformer-30M-RNNT-6000h model using sherpa-onnx.
This model is optimized for Vietnamese speech recognition with fast CPU inference.
"""

import sherpa_onnx
import wave
import numpy as np
import sys
import os

def read_wav_file(filename):
    """Read a WAV file and return the sample rate and samples."""
    with wave.open(filename, 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        
        # Read all frames
        samples = wf.readframes(num_frames)
        
        # Convert to numpy array
        if sample_width == 2:
            samples = np.frombuffer(samples, dtype=np.int16)
        elif sample_width == 4:
            samples = np.frombuffer(samples, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Convert to float32 and normalize to [-1, 1]
        samples = samples.astype(np.float32) / 32768.0
        
        # If stereo, convert to mono by averaging channels
        if channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        
        return sample_rate, samples


def download_model():
    """Download the Zipformer model from Hugging Face if not already present."""
    import urllib.request
    import zipfile
    
    model_dir = "zipformer-30m-rnnt-6000h"
    
    if os.path.exists(model_dir):
        print(f"Model directory '{model_dir}' already exists.")
        return model_dir
    
    print("Downloading Zipformer model from Hugging Face...")
    
    # Note: You'll need to get the actual download URL from the Hugging Face page
    # For now, we'll print instructions
    print("\n" + "="*80)
    print("MODEL DOWNLOAD REQUIRED")
    print("="*80)
    print("\nPlease download the model files manually:")
    print("1. Visit: https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h")
    print("2. Click on 'Files and versions' tab")
    print("3. Download the following files to './zipformer-30m-rnnt-6000h/' directory:")
    print("   - encoder-epoch-99-avg-1.onnx")
    print("   - decoder-epoch-99-avg-1.onnx")
    print("   - joiner-epoch-99-avg-1.onnx")
    print("   - tokens.txt")
    print("\nAlternatively, use git-lfs to clone the repository:")
    print("   git lfs install")
    print("   git clone https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h zipformer-30m-rnnt-6000h")
    print("="*80 + "\n")
    
    return None


def main():
    # Check if audio file is provided
    if len(sys.argv) < 2:
        print("Usage: python test_zipformer.py <audio_file.wav>")
        print("\nExample:")
        print("  python test_zipformer.py recording.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found!")
        sys.exit(1)
    
    # Download or check for model
    model_dir = download_model()
    
    if model_dir is None:
        print("\nPlease download the model first and run the script again.")
        sys.exit(1)
    
    # Check if model files exist
    encoder_path = os.path.join(model_dir, "encoder-epoch-20-avg-10.onnx")
    decoder_path = os.path.join(model_dir, "decoder-epoch-20-avg-10.onnx")
    joiner_path = os.path.join(model_dir, "joiner-epoch-20-avg-10.onnx")
    tokens_path = os.path.join(model_dir, "bpe.model")
    
    for path in [encoder_path, decoder_path, joiner_path, tokens_path]:
        if not os.path.exists(path):
            print(f"Error: Model file not found: {path}")
            print("\nPlease download all required model files.")
            sys.exit(1)
    
    print(f"\nLoading Zipformer model from '{model_dir}'...")
    
    # Configure the recognizer
    recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
        model_config=sherpa_onnx.OnlineModelConfig(
            transducer=sherpa_onnx.OnlineTransducerModelConfig(
                encoder=encoder_path,
                decoder=decoder_path,
                joiner=joiner_path,
            ),
            tokens=tokens_path,
            num_threads=4,
            provider="cpu",
            debug=False,
        ),
        decoding_config=sherpa_onnx.OnlineRecognizerDecodingConfig(
            decoding_method="greedy_search",
        ),
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20.0,
    )
    
    # Create recognizer
    recognizer = sherpa_onnx.OnlineRecognizer(recognizer_config)
    
    print(f"Loading audio file: {audio_file}")
    sample_rate, samples = read_wav_file(audio_file)
    
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(samples) / sample_rate:.2f} seconds")
    print(f"  Samples: {len(samples)}")
    
    # Create a stream
    stream = recognizer.create_stream()
    
    # Feed audio samples to the recognizer
    # Process in chunks for more realistic streaming behavior
    chunk_size = int(0.1 * sample_rate)  # 100ms chunks
    
    print("\nProcessing audio...")
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i+chunk_size]
        stream.accept_waveform(sample_rate, chunk)
        
        # Decode periodically
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
    
    # Signal end of input
    stream.input_finished()
    
    # Final decoding
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    
    # Get the result
    result = recognizer.get_result(stream)
    
    print("\n" + "="*80)
    print("TRANSCRIPTION RESULT")
    print("="*80)
    print(result)
    print("="*80)
    
    # Also print token-level info if available
    if hasattr(result, 'tokens') and result.tokens:
        print("\nTokens:", result.tokens)


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
