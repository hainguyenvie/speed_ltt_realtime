#!/usr/bin/env python3
"""
Speaker Diarization + Speech-to-Text using sherpa-onnx.
Identifies different speakers and transcribes what each person said.
"""

import sherpa_onnx
import wave
import numpy as np
import sys
import os
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict


class SpeakerDiarizer:
    """Speaker diarization using sherpa-onnx models."""
    
    def __init__(self, models_dir="speaker-diarization-models"):
        """Initialize speaker diarization models."""
        print("Loading speaker diarization models...")
        
        # Segmentation model (VAD + speaker change detection)
        segmentation_model = os.path.join(
            models_dir,
            "sherpa-onnx-pyannote-segmentation-3-0",
            "model.onnx"
        )
        
        # Speaker embedding model
        embedding_model = os.path.join(
            models_dir,
            "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
        )
        
        # Check models exist
        if not os.path.exists(segmentation_model):
            raise FileNotFoundError(
                f"Segmentation model not found: {segmentation_model}\n"
                "Run: python download_diarization_models.py"
            )
        
        if not os.path.exists(embedding_model):
            raise FileNotFoundError(
                f"Embedding model not found: {embedding_model}\n"
                "Run: python download_diarization_models.py"
            )
        
        # Create speaker diarization config
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
                num_clusters=-1,  # Auto-detect number of speakers
                threshold=0.5,    # Similarity threshold
            ),
            min_duration_on=0.3,      # Min speech duration (seconds)
            min_duration_off=0.5,     # Min silence duration (seconds)
        )
        
        self.diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
        print("✅ Speaker diarization models loaded!\n")
    
    def process(self, audio_file):
        """
        Process audio file and return speaker segments.
        
        Returns:
            List of (start_time, end_time, speaker_id) tuples
        """
        print(f"Processing: {audio_file}")
        
        # Read audio
        with wave.open(audio_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            samples = wf.readframes(num_frames)
            
            # Convert to float32
            samples_int16 = np.frombuffer(samples, dtype=np.int16)
            samples_float32 = samples_int16.astype(np.float32) / 32768.0
            
            # Handle stereo
            if wf.getnchannels() == 2:
                samples_float32 = samples_float32.reshape(-1, 2).mean(axis=1)
            
            duration = len(samples_float32) / sample_rate
            print(f"Duration: {duration:.2f}s, Sample rate: {sample_rate} Hz\n")
        
        # Run diarization
        print("Running speaker diarization...")
        segments = self.diarizer.process(samples_float32, sample_rate)
        
        # Convert to list of tuples
        speaker_segments = []
        for segment in segments:
            speaker_segments.append((
                segment.start,
                segment.end,
                f"Speaker {segment.speaker + 1}"  # 1-indexed for display
            ))
        
        print(f"✅ Found {len(set(s[2] for s in speaker_segments))} speakers\n")
        
        return speaker_segments


class ZipformerSTT:
    """Speech-to-text using Zipformer model."""
    
    def __init__(self, model_dir="zipformer-30m-rnnt-6000h"):
        """Initialize Z ipformer STT model."""
        print("Loading Zipformer STT model...")
        
        encoder = os.path.join(model_dir, "encoder-epoch-20-avg-10.int8.onnx")
        decoder = os.path.join(model_dir, "decoder-epoch-20-avg-10.int8.onnx")
        joiner = os.path.join(model_dir, "joiner-epoch-20-avg-10.int8.onnx")
        tokens = os.path.join(model_dir, "config.json")
        
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
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
    
    def transcribe_segment(self, audio_samples, sample_rate):
        """Transcribe an audio segment."""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio_samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()


def format_time(seconds):
    """Format seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcript_with_speakers.py <audio_file.wav>")
        print("\nExample:")
        print("  python transcript_with_speakers.py meeting.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file '{audio_file}' not found!")
        sys.exit(1)
    
    print("="*80)
    print("🎙️  MULTI-SPEAKER TRANSCRIPTION")
    print("="*80 + "\n")
    
    try:
        # Initialize models
        diarizer = SpeakerDiarizer()
        stt = ZipformerSTT()
        
        # Get speaker segments
        speaker_segments = diarizer.process(audio_file)
        
        # Read full audio for transcription
        with wave.open(audio_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            samples = wf.readframes(num_frames)
            
            samples_int16 = np.frombuffer(samples, dtype=np.int16)
            full_audio = samples_int16.astype(np.float32) / 32768.0
            
            if wf.getnchannels() == 2:
                full_audio = full_audio.reshape(-1, 2).mean(axis=1)
        
        # Transcribe each segment
        print("Transcribing segments...")
        results = []
        
        for start, end, speaker in speaker_segments:
            # Extract segment
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_audio = full_audio[start_sample:end_sample]
            
            # Transcribe
            text = stt.transcribe_segment(segment_audio, sample_rate)
            
            if text:  # Only add if there's actual text
                results.append({
                    'start': start,
                    'end': end,
                    'speaker': speaker,
                    'text': text
                })
                print(f"  [{format_time(start)}-{format_time(end)}] {speaker}: {text}")
        
        # Display final result
        print("\n" + "="*80)
        print("📝 TRANSCRIPT WITH SPEAKERS")
        print("="*80 + "\n")
        
        for result in results:
            print(f"[{format_time(result['start'])}] {result['speaker']}: {result['text']}\n")
        
        print("="*80 + "\n")
        
        # Summary
        speakers = set(r['speaker'] for r in results)
        print(f"Summary:")
        print(f"  • Total segments: {len(results)}")
        print(f"  • Speakers detected: {len(speakers)}")
        for speaker in sorted(speakers):
            count = sum(1 for r in results if r['speaker'] == speaker)
            print(f"    - {speaker}: {count} segments")
        print()
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
