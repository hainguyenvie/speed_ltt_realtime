#!/usr/bin/env python3
"""
Advanced pyannote diarization tuning script.
Allows customization of segmentation and clustering parameters.
"""

from pyannote.audio import Pipeline
import soundfile as sf
import torch
import numpy as np
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python tune_diarization.py <audio_file.wav> [min_speakers] [max_speakers]")
        print("\nExample:")
        print("  python tune_diarization.py recording.wav")
        print("  python tune_diarization.py recording.wav 1 2  # Force 1-2 speakers")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    min_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    max_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    print("="*70)
    print("🔧 Pyannote Diarization Tuning")
    print("="*70)
    print(f"\nAudio: {audio_file}")
    
    if min_speakers or max_speakers:
        print(f"Speaker range: {min_speakers or '?'} - {max_speakers or '?'}")
    
    # Load pipeline
    print("\nLoading pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    
    # Load audio
    audio_data, sr = sf.read(audio_file)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    waveform = torch.from_numpy(audio_data[np.newaxis, :]).float()
    audio_dict = {"waveform": waveform, "sample_rate": sr}
    
    # Test 1: Default settings
    print("\n" + "-"*70)
    print("Test 1: Default settings")
    print("-"*70)
    
    result1 = pipeline(audio_dict)
    diar1 = result1.speaker_diarization
    
    speakers1 = set(s for _, _, s in diar1.itertracks(yield_label=True))
    segments1 = len(list(diar1.itertracks(yield_label=True)))
    
    print(f"Speakers: {len(speakers1)} ({', '.join(sorted(speakers1))})")
    print(f"Segments: {segments1}")
    
    # Test 2: With speaker constraints
    if min_speakers or max_speakers:
        print("\n" + "-"*70)
        print("Test 2: With speaker constraints")
        print("-"*70)
        
        result2 = pipeline(
            audio_dict,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        diar2 = result2.speaker_diarization
        
        speakers2 = set(s for _, _, s in diar2.itertracks(yield_label=True))
        segments2 = len(list(diar2.itertracks(yield_label=True)))
        
        print(f"Speakers: {len(speakers2)} ({', '.join(sorted(speakers2))})")
        print(f"Segments: {segments2}")
        
        # Compare
        print("\n" + "-"*70)
        print("Comparison:")
        print(f"  Speakers: {len(speakers1)} → {len(speakers2)}")
        print(f"  Segments: {segments1} → {segments2}")
    
    # Show segments
    print("\n" + "="*70)
    
    constraint_text = "applied" if (min_speakers or max_speakers) else "default"
    print(f"SEGMENTS (with constraints {constraint_text}):")
    print("="*70)
    
    diar = diar2 if (min_speakers or max_speakers) else diar1
    
    for segment, _, speaker in diar.itertracks(yield_label=True):
        duration = segment.end - segment.start
        print(f"  [{segment.start:5.1f}s - {segment.end:5.1f}s] ({duration:4.1f}s) {speaker}")
    
    print("\n" + "="*70)
    print("TUNING TIPS:")
    print("="*70)
    print("""
1. Over-segmentation (too many segments):
   - Set realistic min_speakers/max_speakers
   - Increase segmentation.min_duration_on (filter short speech)
   - Increase clustering.threshold (merge more)

2. Wrong speaker count:
   - Provide min_speakers/max_speakers hints
   - Increase clustering.threshold (fewer speakers)
   - Check audio quality (noise can create false speakers)

3. One person detected as multiple:
   - Increase clustering.threshold to merge
   - Provide max_speakers=1 to force single speaker

For advanced tuning, modify pipeline parameters:
  pipeline.instantiate({"clustering": {"threshold": 0.8}})
    """)


if __name__ == "__main__":
    main()
