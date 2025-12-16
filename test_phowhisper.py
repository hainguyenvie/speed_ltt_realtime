#!/usr/bin/env python3
"""
Test script for PhoWhisper-base Vietnamese Speech-to-Text model
"""

import argparse
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")


def transcribe_audio(audio_path, model_name="vinai/PhoWhisper-base", device=None):
    """
    Transcribe audio file using PhoWhisper model
    
    Args:
        audio_path: Path to audio file (wav, mp3, m4a, etc.)
        model_name: Model name from Hugging Face
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        Transcription text
    """
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")
    
    # Create ASR pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device
    )
    
    print(f"Processing audio file: {audio_path}")
    
    # Transcribe
    result = pipe(audio_path)
    
    return result["text"]


def main():
    parser = argparse.ArgumentParser(
        description="Test PhoWhisper Vietnamese Speech-to-Text model"
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file (wav, mp3, m4a, etc.)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vinai/PhoWhisper-base",
        help="Model name (default: vinai/PhoWhisper-base)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = None if args.device == "auto" else args.device
    
    # Transcribe
    transcription = transcribe_audio(args.audio_file, args.model, device)
    
    # Print result
    print("\n" + "="*50)
    print("TRANSCRIPTION:")
    print("="*50)
    print(transcription)
    print("="*50)


if __name__ == "__main__":
    main()
