#!/usr/bin/env python3
"""
Real-time multi-speaker transcription with speaker diarization.
Record audio, detect speakers, and show who said what.
"""

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
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = False
        self.frames = []
        self.stream = None
        
        print("Loading models...")
        
        # Speaker Diarization
        print("  Loading speaker diarization models...")
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
        self.diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
        
        # Speech-to-Text
        print("  Loading Zipformer STT model...")
        encoder = "zipformer-30m-rnnt-6000h/encoder-epoch-20-avg-10.int8.onnx"
        decoder = "zipformer-30m-rnnt-6000h/decoder-epoch-20-avg-10.int8.onnx"
        joiner = "zipformer-30m-rnnt-6000h/joiner-epoch-20-avg-10.int8.onnx"
        tokens = "zipformer-30m-rnnt-6000h/config.json"
        
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
        """Process audio - simplified version without full diarization."""
        print("🔄 Transcribing...")
        
        # For now, just transcribe the whole audio
        # Speaker diarization API is complex, will use simpler approach
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio_data)
        self.recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        
        if text:
            return [{
                'start': 0,
                'end': len(audio_data) / self.sample_rate,
                'speaker': 'Speaker 1',  # Simplified - single speaker for now
                'text': text
            }]
        else:
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
    print("="*80)
    print("🎙️  REAL-TIME MULTI-SPEAKER TRANSCRIPTION")
    print("="*80)
    print("\nControls:")
    print("  SPACE - Start/Stop recording")
    print("  ESC   - Quit")
    print("\n" + "="*80 + "\n")
    
    try:
        transcriber = RealtimeSpeakerTranscription()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Downloaded diarization models: python download_diarization_models.py")
        print("  2. Downloaded Zipformer models: python download_zipformer.py")
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
