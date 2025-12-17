#!/usr/bin/env python3
"""
Multi-speaker transcription using pyannote.audio + Zipformer.
Automatically detects speakers and transcribes each segment.
"""

from pyannote.audio import Pipeline
import sherpa_onnx
import wave
import numpy as np
import sys
import os


def format_time(seconds):
    """Format seconds to MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcript_with_speakers_pyannote.py <audio_file.wav> [hf_token]")
        print("\nExample:")
        print("  python transcript_with_speakers_pyannote.py meeting.wav")
        print("  python transcript_with_speakers_pyannote.py meeting.wav hf_xxxxx")
        print("\nIf token not provided, will use logged-in HuggingFace token.")
        print("Login with: huggingface-cli login")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    hf_token = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file '{audio_file}' not found!")
        sys.exit(1)
    
    print("="*80)
    print("🎙️  MULTI-SPEAKER TRANSCRIPTION (Pyannote + Zipformer)")
    print("="*80 + "\n")
    
    # 1. Load Speaker Diarization
    print("Loading speaker diarization model...")
    try:
        if hf_token:
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
        else:
            # Use logged-in HF token
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
        print("✅ Diarization model loaded!\n")
    except Exception as e:
        print(f"❌ Error loading diarization model: {e}")
        print("\nMake sure:")
        print("  1. You have a valid HuggingFace token")
        print("  2. You accepted the model license at:")
        print("     https://huggingface.co/pyannote/speaker-diarization-3.1")
        sys.exit(1)
    
    # 2. Load Speech-to-Text
    print("Loading Zipformer STT model...")
    model_dir = "zipformer-30m-rnnt-6000h"
    
    encoder = os.path.join(model_dir, "encoder-epoch-20-avg-10.int8.onnx")
    decoder = os.path.join(model_dir, "decoder-epoch-20-avg-10.int8.onnx")
    joiner = os.path.join(model_dir, "joiner-epoch-20-avg-10.int8.onnx")
    tokens = os.path.join(model_dir, "config.json")
    
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
    print("✅ STT model loaded!\n")
    
    # 3. Load audio file first (avoid torchcodec issue)
    import soundfile as sf
    import torch
    
    print(f"Loading audio: {audio_file}")
    audio_data, sample_rate_file = sf.read(audio_file)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Convert to torch tensor
    waveform = torch.from_numpy(audio_data[np.newaxis, :]).float()
    
    # Create audio dict for pyannote
    audio_dict = {
        "waveform": waveform,
        "sample_rate": sample_rate_file
    }
    
    print(f"Audio loaded: {len(audio_data)/sample_rate_file:.1f}s\n")
    
    # 4. Run Speaker Diarization
    print("Running speaker diarization...")
    
    diarization_result = diarization_pipeline(audio_dict)
    
    # Get the annotation object (has itertracks method)
    diarization = diarization_result.speaker_diarization
    
    num_speakers = len(set(speaker for _, _, speaker in diarization.itertracks(yield_label=True)))
    num_segments = len(list(diarization.itertracks(yield_label=True)))
    
    print(f"✅ Found {num_speakers} speaker(s), {num_segments} segment(s)\n")
    
    # 5. Load Audio for Transcription (reuse loaded audio)
    # Already loaded as audio_data and sample_rate_file
    full_audio = audio_data.astype(np.float32)
    sample_rate = sample_rate_file
    
    # 6. Transcribe Each Segment
    print("Transcribing segments...")
    results = []
    
    MIN_SEGMENT_DURATION = 0.5  # Minimum 0.5s for Zipformer
    
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        # Skip segments that are too short
        segment_duration = segment.end - segment.start
        if segment_duration < MIN_SEGMENT_DURATION:
            print(f"  ⏭️  Skipping short segment ({segment_duration:.2f}s)")
            continue
        
        # Extract audio segment
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        segment_audio = full_audio[start_sample:end_sample]
        
        # Transcribe
        try:
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, segment_audio)
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            
            if text:
                results.append({
                    'start': segment.start,
                    'end': segment.end,
                    'speaker': speaker,
                    'text': text
                })
                print(f"  [{format_time(segment.start)}-{format_time(segment.end)}] {speaker}: {text}")
        except Exception as e:
            print(f"  ⚠️  Error transcribing segment {segment.start:.1f}s-{segment.end:.1f}s: {e}")
    
    # 7. Display Results
    print("\n" + "="*80)
    print("📝 TRANSCRIPT WITH SPEAKERS")
    print("="*80 + "\n")
    
    for r in results:
        print(f"[{format_time(r['start'])}] {r['speaker']}: {r['text']}\n")
    
    print("="*80 + "\n")
    
    # Summary
    speakers = set(r['speaker'] for r in results)
    print(f"Summary:")
    print(f"  • Total segments transcribed: {len(results)}")
    print(f"  • Speakers detected: {len(speakers)}")
    for speaker in sorted(speakers):
        count = sum(1 for r in results if r['speaker'] == speaker)
        print(f"    - {speaker}: {count} segments")
    print()


if __name__ == "__main__":
    main()
