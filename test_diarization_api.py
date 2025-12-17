#!/usr/bin/env python3
"""
Test using callback with speaker diarization.
"""

import sherpa_onnx
import wave
import numpy as np

# Load audio
with wave.open("recording.wav", 'rb') as wf:
    sample_rate = wf.getframerate()
    num_frames = wf.getnframes()
    samples = wf.readframes(num_frames)
    
    samples_int16 = np.frombuffer(samples, dtype=np.int16)
    samples_float32 = samples_int16.astype(np.float32) / 32768.0
    
    if wf.getnchannels() == 2:
        samples_float32 = samples_float32.reshape(-1, 2).mean(axis=1)

# Create diarization config
segmentation_model = "speaker-diarization-models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
embedding_model = "speaker-diarization-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=segmentation_model
        )
    ),
    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=embedding_model,
        num_threads=4,
        debug=False,
        provider="cpu",
    ),
    clustering=sherpa_onnx.FastClusteringConfig(
        num_clusters=-1,
        threshold=0.5,
    ),
    min_duration_on=0.3,
    min_duration_off=0.5,
)

diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)

# Store segments using callback
segments_list = []

def progress_callback(num_processed, num_total):
    """Progress callback - returns 1 to continue."""
    print(f"Progress: {num_processed}/{num_total}")
    return 1

print("Processing audio with callback...")
result = diarizer.process(samples_float32.tolist(), progress_callback)

print(f"\nNum segments: {result.num_segments}")
print(f"Num speakers: {result.num_speakers}")

# The result itself might need to be converted to string or has a different API
print(f"\nResult as string:\n{str(result)}")

# Try calling sort methods
try:
    result.sort_by_start_time()
    print("\nSorted by start time successfully!")
except Exception as e:
    print(f"\nSort error: {e}")

print("\nDone!")
