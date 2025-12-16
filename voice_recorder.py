#!/usr/bin/env python3
"""
Record audio from microphone and transcribe using Zipformer.
Press SPACE to start/stop recording, ESC to quit.
"""

import sounddevice as sd
import numpy as np
import wave
import os
import tempfile
from pynput import keyboard
import sherpa_onnx
import time
import threading


class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = False
        self.frames = []
        self.stream = None
        
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
        
        audio_data = np.concatenate(self.frames, axis=0)
        duration = len(audio_data) / self.sample_rate
        
        print(f"⏹️  STOPPED (Duration: {duration:.1f}s)")
        return audio_data.flatten()


class ZipformerTranscriber:
    def __init__(self, model_dir="zipformer-30m-rnnt-6000h"):
        """Initialize Zipformer model."""
        print("Loading Zipformer model...")
        
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
        print("✅ Model loaded!\n")
    
    def transcribe(self, audio_data, sample_rate=16000):
        """Transcribe audio data."""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio_data)
        
        start_time = time.time()
        self.recognizer.decode_stream(stream)
        elapsed = time.time() - start_time
        
        duration = len(audio_data) / sample_rate
        speed = duration / elapsed if elapsed > 0 else 0
        
        result = stream.result.text
        return result, elapsed, speed


def main():
    # Initialize
    recorder = AudioRecorder()
    transcriber = ZipformerTranscriber()
    
    is_recording = False
    running = True
    
    print("="*80)
    print("🎙️  VOICE RECORDER + SPEECH TO TEXT")
    print("="*80)
    print("\nControls:")
    print("  SPACE - Start/Stop recording")
    print("  ESC   - Quit")
    print("\n" + "="*80)
    print("Ready! Press SPACE to start recording...")
    print("="*80 + "\n")
    
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
                    recorder.start_recording()
                else:
                    # Stop recording and transcribe
                    is_recording = False
                    audio_data = recorder.stop_recording()
                    
                    if audio_data is not None and len(audio_data) > 0:
                        print("🔄 Transcribing...")
                        
                        try:
                            text, elapsed, speed = transcriber.transcribe(audio_data)
                            
                            print("\n" + "="*80)
                            print("📝 TRANSCRIPTION:")
                            print("="*80)
                            print(text if text else "(empty)")
                            print("="*80)
                            print(f"⏱️  Processing: {elapsed:.2f}s ({speed:.1f}x realtime)")
                            print("="*80 + "\n")
                            
                        except Exception as e:
                            print(f"❌ Error during transcription: {e}")
                    
                    print("Ready! Press SPACE to record again...")
                    
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
            recorder.stop_recording()
        listener.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
