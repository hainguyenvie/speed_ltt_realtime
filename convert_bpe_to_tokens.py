#!/usr/bin/env python3
"""
Convert bpe.model (SentencePiece) to tokens.txt format for sherpa-onnx.
"""

import sentencepiece as spm
import os

def convert_bpe_to_tokens(bpe_model_path, output_path):
    """Convert a SentencePiece model to tokens.txt format."""
    
    print(f"Loading BPE model from: {bpe_model_path}")
    
    # Load the SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_path)
    
    vocab_size = sp.vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Write all tokens to tokens.txt
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(vocab_size):
            token = sp.id_to_piece(i)
            f.write(f"{token}\n")
    
    print(f"✓ Tokens saved to: {output_path}")
    print(f"  Total tokens: {vocab_size}")


if __name__ == "__main__":
    bpe_model = "zipformer-30m-rnnt-6000h/bpe.model"
    tokens_output = "zipformer-30m-rnnt-6000h/tokens.txt"
    
    if not os.path.exists(bpe_model):
        print(f"Error: BPE model not found: {bpe_model}")
        exit(1)
    
    convert_bpe_to_tokens(bpe_model, tokens_output)
    print("\n✓ Conversion complete!")
    print(f"\nYou can now run:")
    print(f"  python test_zipformer_simple.py recording.wav")
