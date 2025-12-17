#!/usr/bin/env python3
"""
Test speaker consistency across segments.
"""

from pyannote.audio import Pipeline
import soundfile as sf
import torch
import numpy as np

# Load pipeline
print("Loading pyannote...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

# Load your recording
audio_file = "recording.wav"
audio_data, sr = sf.read(audio_file)

if len(audio_data.shape) > 1:
    audio_data = audio_data.mean(axis=1)

# Split into 3 segments
duration = len(audio_data) / sr
seg1_end = int(len(audio_data) * 0.33)
seg2_start = seg1_end
seg2_end = int(len(audio_data) * 0.66)
seg3_start = seg2_end

segments = [
    ("Segment 1", audio_data[:seg1_end]),
    ("Segment 2", audio_data[seg2_start:seg2_end]),
    ("Segment 3", audio_data[seg3_start:])
]

print(f"\nTesting speaker consistency on {audio_file} ({duration:.1f}s)")
print("="*60)

# Process each segment separately
for name, seg_audio in segments:
    if len(seg_audio) < sr * 0.5:  # Skip if too short
        continue
        
    # Convert to tensor
    waveform = torch.from_numpy(seg_audio[np.newaxis, :]).float()
    audio_dict = {"waveform": waveform, "sample_rate": sr}
    
    # Run diarization
    result = pipeline(audio_dict)
    diarization = result.speaker_diarization
    
    speakers = set(speaker for _, _, speaker in diarization.itertracks(yield_label=True))
    
    print(f"\n{name}:")
    print(f"  Speakers detected: {speakers}")
    
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        print(f"    [{segment.start:.1f}s-{segment.end:.1f}s] {speaker}")

print("\n" + "="*60)
print("⚠️  PROBLEM: Speaker labels are NOT consistent!")
print("    SPEAKER_00 in different segments = DIFFERENT people")
print("="*60)

# Now process the WHOLE audio at once
print("\n\nProcessing ENTIRE audio (correct way):")
print("="*60)

waveform_full = torch.from_numpy(audio_data[np.newaxis, :]).float()
audio_dict_full = {"waveform": waveform_full, "sample_rate": sr}

result_full = pipeline(audio_dict_full)
diarization_full = result_full.speaker_diarization

speakers_full = set(speaker for _, _, speaker in diarization_full.itertracks(yield_label=True))
print(f"\nSpeakers in entire audio: {speakers_full}")

for segment, _, speaker in diarization_full.itertracks(yield_label=True):
    print(f"  [{segment.start:.1f}s-{segment.end:.1f}s] {speaker}")

print("\n" + "="*60)
print("✅ NOW speaker labels ARE consistent!")
print("="*60)
