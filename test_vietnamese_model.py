#!/usr/bin/env python3
"""
Test script for the new Vietnamese Sherpa-ONNX model (70k hours training).
Supports testing with audio files or real-time microphone input.
"""

import sherpa_onnx
import sounddevice as sd
import numpy as np
import sys
import os
import wave
import time
from pathlib import Path


class VietnameseASRTester:
    def __init__(self, model_dir="sherpa-onnx-zipformer-vi-2025-04-20"):
        """Initialize the Vietnamese ASR model."""
        self.model_dir = model_dir
        self.sample_rate = 16000
        
        print("="*80)
        print("🇻🇳 VIETNAMESE SPEECH RECOGNITION TESTER")
        print("="*80)
        print(f"\n📦 Model: {model_dir}")
        print("📊 Training: 70,000 hours Vietnamese speech")
        print("🎯 Precision: Float32 (high accuracy)")
        print("🔍 Decoding: Modified Beam Search\n")
        print("="*80)
        
        print("\n🔄 Loading model...")
        
        encoder = os.path.join(model_dir, "encoder-epoch-12-avg-8.onnx")
        decoder = os.path.join(model_dir, "decoder-epoch-12-avg-8.onnx")
        joiner = os.path.join(model_dir, "joiner-epoch-12-avg-8.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")
        
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=4,
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method="modified_beam_search",
            max_active_paths=4,
        )
        
        print("✅ Model loaded successfully!\n")
    
    def transcribe_file(self, audio_file):
        """Transcribe an audio file."""
        print(f"\n📁 Processing file: {audio_file}")
        
        if not os.path.exists(audio_file):
            print(f"❌ Error: File not found: {audio_file}")
            return None
        
        try:
            # Read WAV file
            with wave.open(audio_file, 'rb') as wf:
                sample_rate = wf.getframerate()
                num_channels = wf.getnchannels()
                num_frames = wf.getnframes()
                audio_data = wf.readframes(num_frames)
                
                # Convert to numpy array
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Handle stereo -> mono
                if num_channels == 2:
                    samples = samples.reshape(-1, 2).mean(axis=1)
                
                # Resample if needed
                if sample_rate != self.sample_rate:
                    print(f"⚠️  Resampling from {sample_rate}Hz to {self.sample_rate}Hz...")
                    # Simple resampling (for production, use a proper resampling library)
                    duration = len(samples) / sample_rate
                    target_length = int(duration * self.sample_rate)
                    samples = np.interp(
                        np.linspace(0, len(samples) - 1, target_length),
                        np.arange(len(samples)),
                        samples
                    )
            
            duration = len(samples) / self.sample_rate
            print(f"⏱️  Audio duration: {duration:.2f} seconds")
            
            # Transcribe
            print("🔄 Transcribing...")
            start_time = time.time()
            
            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, samples)
            self.recognizer.decode_stream(stream)
            
            result = stream.result.text.strip()
            
            elapsed = time.time() - start_time
            rtf = elapsed / duration  # Real-time factor
            
            print(f"⚡ Processing time: {elapsed:.2f}s (RTF: {rtf:.2f}x)")
            print(f"\n📝 Transcription:\n{'-'*80}")
            if result:
                print(f"   {result}")
            else:
                print("   (No speech detected)")
            print('-'*80)
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def record_and_transcribe(self, duration=5):
        """Record from microphone and transcribe."""
        print(f"\n🎤 Recording {duration} seconds from microphone...")
        print("   Speak now!\n")
        
        try:
            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            
            # Show countdown
            for i in range(duration, 0, -1):
                print(f"   ⏱️  {i}...", end='\r')
                sd.wait(1000)
            
            sd.wait()  # Wait for recording to finish
            print("\n✅ Recording complete!")
            
            # Transcribe
            print("🔄 Transcribing...")
            start_time = time.time()
            
            samples = recording.flatten()
            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, samples)
            self.recognizer.decode_stream(stream)
            
            result = stream.result.text.strip()
            
            elapsed = time.time() - start_time
            
            print(f"⚡ Processing time: {elapsed:.2f}s")
            print(f"\n📝 Transcription:\n{'-'*80}")
            if result:
                print(f"   {result}")
            else:
                print("   (No speech detected)")
            print('-'*80)
            
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_sample_audios(self):
        """Test with sample audio files from the model."""
        test_dir = os.path.join(self.model_dir, "test_wavs")
        
        if not os.path.exists(test_dir):
            print(f"⚠️  No sample audio files found in {test_dir}")
            return
        
        wav_files = sorted(Path(test_dir).glob("*.wav"))
        
        if not wav_files:
            print(f"⚠️  No .wav files found in {test_dir}")
            return
        
        print(f"\n📂 Found {len(wav_files)} sample audio file(s)")
        print("="*80)
        
        for wav_file in wav_files:
            self.transcribe_file(str(wav_file))
            print()


def main():
    """Main interactive test menu."""
    try:
        tester = VietnameseASRTester()
        
        while True:
            print("\n" + "="*80)
            print("🎯 TEST OPTIONS")
            print("="*80)
            print("1. Test with sample audio files (from model)")
            print("2. Test with custom audio file")
            print("3. Record from microphone (5 seconds)")
            print("4. Record from microphone (10 seconds)")
            print("5. Test with recording.wav (if exists)")
            print("0. Exit")
            print("="*80)
            
            choice = input("\n👉 Select option (0-5): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye!")
                break
            
            elif choice == "1":
                tester.test_sample_audios()
            
            elif choice == "2":
                file_path = input("📁 Enter audio file path: ").strip()
                tester.transcribe_file(file_path)
            
            elif choice == "3":
                tester.record_and_transcribe(duration=5)
            
            elif choice == "4":
                tester.record_and_transcribe(duration=10)
            
            elif choice == "5":
                if os.path.exists("recording.wav"):
                    tester.transcribe_file("recording.wav")
                else:
                    print("❌ recording.wav not found")
            
            else:
                print("❌ Invalid option. Please try again.")
            
            input("\n⏸️  Press Enter to continue...")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
