#!/usr/bin/env python3
"""
Voice Activity Detection (VAD) based real-time transcription.
Automatically transcribes when you stop speaking (silence detected).
"""

import sherpa_onnx
import sounddevice as sd
import numpy as np
import sys
import os
import threading
import time


class VADRealtimeTranscriber:
    def __init__(self, model_dir="zipformer-30m-rnnt-6000h", sample_rate=16000):
        """Initialize transcriber with VAD."""
        self.sample_rate = sample_rate
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.running = False
        
        # VAD parameters
        self.silence_threshold = 0.01  # Energy threshold for silence
        self.silence_duration = 1.5    # Seconds of silence to trigger transcription
        self.min_speech_duration = 0.5 # Minimum speech duration to process
        
        self.silent_chunks = 0
        self.chunks_per_second = 10  # 100ms chunks
        self.is_speaking = False
        
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
            sample_rate=sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
        )
        
        print("✅ Model loaded!\n")
    
    def get_audio_energy(self, audio_chunk):
        """Calculate RMS energy of audio chunk."""
        return np.sqrt(np.mean(audio_chunk**2))
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream."""
        if status:
            print(f"Status: {status}", file=sys.stderr)
        
        if not self.running:
            return
        
        # Calculate energy
        energy = self.get_audio_energy(indata)
        is_silent = energy < self.silence_threshold
        
        with self.buffer_lock:
            if not is_silent:
                # Speech detected
                if not self.is_speaking:
                    print("🎤 Listening...", end='\r', flush=True)
                    self.is_speaking = True
                
                self.audio_buffer.append(indata.copy())
                self.silent_chunks = 0
                
            else:
                # Silence detected
                if self.is_speaking:
                    self.silent_chunks += 1
                    self.audio_buffer.append(indata.copy())
                    
                    # Check if silence duration reached
                    silence_duration = self.silent_chunks / self.chunks_per_second
                    if silence_duration >= self.silence_duration:
                        # Trigger transcription
                        self.transcribe_buffer()
                        self.is_speaking = False
                        self.silent_chunks = 0
    
    def transcribe_buffer(self):
        """Transcribe accumulated audio buffer."""
        if not self.audio_buffer:
            return
        
        # Get audio data
        audio_data = np.concatenate(self.audio_buffer, axis=0)
        self.audio_buffer = []  # Clear buffer
        
        duration = len(audio_data) / self.sample_rate
        
        # Skip if too short
        if duration < self.min_speech_duration:
            return
        
        try:
            # Transcribe
            print("\r\033[K🔄 Processing...", end='', flush=True)
            
            samples = audio_data.flatten().astype(np.float32)
            
            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, samples)
            self.recognizer.decode_stream(stream)
            
            result = stream.result.text.strip()
            
            if result:
                print(f"\r\033[K📝 {result}")
            else:
                print("\r\033[K", end='')
                
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
    
    def start(self):
        """Start VAD-based transcription."""
        self.running = True
        
        print("="*80)
        print("🎙️  VOICE ACTIVITY DETECTION - SPEECH-TO-TEXT")
        print("="*80)
        print("\nHow it works:")
        print("  • Start speaking - recording starts automatically")
        print("  • Stop speaking - transcript appears after 1.5s silence")
        print("  • Keep talking - it continues until you pause")
        print("\nPress Ctrl+C to stop\n")
        print("="*80)
        print("\nReady! Start speaking...\n")
        
        # Start audio input
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.1),  # 100ms chunks
            ):
                while self.running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n\n👋 Stopping...")
            self.running = False
            
            # Process any remaining audio
            with self.buffer_lock:
                if self.audio_buffer:
                    self.transcribe_buffer()


def main():
    try:
        transcriber = VADRealtimeTranscriber()
        transcriber.start()
    except FileNotFoundError:
        print("\n❌ Error: Model files not found!")
        print("Run: python download_zipformer.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
