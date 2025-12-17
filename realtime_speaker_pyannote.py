#!/usr/bin/env python3
"""
Real-time multi-speaker transcription using pyannote + Zipformer.
Record audio, detect speakers, transcribe each person.
"""

from pyannote.audio import Pipeline
import sherpa_onnx
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
from pynput import keyboard
import time


class RealtimeSpeakerTranscription:
    """Real-time recording with speaker diarization + STT."""
    
    def __init__(self, hf_token, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = False
        self.frames = []
        self.stream = None
        
        print("Loading models...")
        
        # Speaker Diarization
        print("  Loading pyannote speaker diarization...")
        if hf_token:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
        else:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
        
        # Speech-to-Text
        print("  Loading Zipformer STT model...")
        model_dir = "zipformer-30m-rnnt-6000h"
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
            sample_rate=sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        
        print("✅ All models loaded!\n")
    
    def start_recording(self):
        """Start recording audio."""
        self.recording = True
        self.frames = []
        print("\n🔴 RECORDING... (Press SPACE to stop)")
        
        def callback(indata, frames, time_info, status):
            if status:
                print(f"Status: {status}")
            if self.recording:
                self.frames.append(indata.copy())
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=callback
        )
        self.stream.start()
    
    def stop_recording(self):
        """Stop recording and return audio data."""
        if not self.recording:
            return None
            
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        if not self.frames:
            return None
        
        audio_data = np.concatenate(self.frames, axis=0).flatten()
        duration = len(audio_data) / self.sample_rate
        
        print(f"⏹️  STOPPED (Duration: {duration:.1f}s)\n")
        return audio_data
    
    def process_audio(self, audio_data):
        """Process audio with speaker diarization + STT."""
        import torch
        
        # Convert to torch tensor
        waveform = torch.from_numpy(audio_data[np.newaxis, :]).float()
        
        # Create audio dict for pyannote
        audio_dict = {
            "waveform": waveform,
            "sample_rate": self.sample_rate
        }
        
        try:
            # Run speaker diarization
            print("🔍 Detecting speakers...")
            diarization_result = self.diarization_pipeline(audio_dict)
            
            # Get the annotation object
            diarization = diarization_result.speaker_diarization
            
            num_speakers = len(set(speaker for _, _, speaker in diarization.itertracks(yield_label=True)))
            num_segments = len(list(diarization.itertracks(yield_label=True)))
            
            print(f"✅ Found {num_speakers} speaker(s), {num_segments} segment(s)")
            
            # Transcribe each segment
            print("🔄 Transcribing...\n")
            results = []
            
            MIN_SEGMENT_DURATION = 0.5  # Minimum 0.5s for Zipformer
            
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                # Skip segments that are too short
                segment_duration = segment.end - segment.start
                if segment_duration < MIN_SEGMENT_DURATION:
                    print(f"  ⏭️  Skipping short segment ({segment_duration:.2f}s)")
                    continue
                
                # Extract segment audio
                start_sample = int(segment.start * self.sample_rate)
                end_sample = int(segment.end * self.sample_rate)
                segment_audio = audio_data[start_sample:end_sample]
                
                # Transcribe
                try:
                    stream = self.recognizer.create_stream()
                    stream.accept_waveform(self.sample_rate, segment_audio)
                    self.recognizer.decode_stream(stream)
                    text = stream.result.text.strip()
                    
                    if text:
                        results.append({
                            'start': segment.start,
                            'end': segment.end,
                            'speaker': speaker,
                            'text': text
                        })
                except Exception as e:
                    print(f"  ⚠️  Error transcribing segment {segment.start:.1f}s-{segment.end:.1f}s: {e}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def format_time(self, seconds):
        """Format seconds to MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def display_results(self, results):
        """Display transcription with speakers."""
        print("\n" + "="*80)
        print("📝 TRANSCRIPTION WITH SPEAKERS")
        print("="*80 + "\n")
        
        for r in results:
            print(f"[{self.format_time(r['start'])}] {r['speaker']}: {r['text']}\n")
        
        print("="*80)
        
        # Summary
        speakers = set(r['speaker'] for r in results)
        print(f"\nSummary:")
        print(f"  • Total segments: {len(results)}")
        print(f"  • Speakers: {len(speakers)}")
        for speaker in sorted(speakers):
            count = sum(1 for r in results if r['speaker'] == speaker)
            print(f"    - {speaker}: {count} segments")
        print()


def main():
    import sys
    
    hf_token = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("="*80)
    print("🎙️  REAL-TIME MULTI-SPEAKER TRANSCRIPTION")
    print("="*80)
    print("\nControls:")
    print("  SPACE - Start/Stop recording")
    print("  ESC   - Quit")
    print("\n" + "="*80 + "\n")
    
    try:
        transcriber = RealtimeSpeakerTranscription(hf_token)
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nMake sure:")
        print("  1. Valid HuggingFace token")
        print("  2. Accepted model license: https://huggingface.co/pyannote/speaker-diarization-3.1")
        return
    
    is_recording = False
    running = True
    
    print("Ready! Press SPACE to start recording...")
    
    def on_press(key):
        nonlocal is_recording, running
        
        try:
            # ESC to quit
            if key == keyboard.Key.esc:
                print("\n👋 Exiting...")
                running = False
                return False
            
            # SPACE to toggle recording
            if key == keyboard.Key.space:
                if not is_recording:
                    # Start recording
                    is_recording = True
                    transcriber.start_recording()
                else:
                    # Stop and process
                    is_recording = False
                    audio_data = transcriber.stop_recording()
                    
                    if audio_data is not None and len(audio_data) > 0:
                        try:
                            results = transcriber.process_audio(audio_data)
                            
                            if results:
                                transcriber.display_results(results)
                            else:
                                print("⚠️  No speech detected or transcription failed.\n")
                                
                        except Exception as e:
                            print(f"❌ Error: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    print("\nReady! Press SPACE to record again...")
                    
        except AttributeError:
            pass
    
    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    try:
        while running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if is_recording:
            transcriber.stop_recording()
        listener.stop()


if __name__ == "__main__":
    main()
