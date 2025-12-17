#!/usr/bin/env python3
"""
Hybrid Multi-Speaker Transcription System

Mode 1 (Live): Real-time streaming transcript WITHOUT speaker labels
                - Zero delay
                - Saves audio + transcript to file
                
Mode 2 (Review): Post-meeting accurate speaker diarization
                 - Process entire recording
                 - Add accurate speaker labels
                 - Generate final transcript
"""

from pyannote.audio import Pipeline
import sherpa_onnx
import sounddevice as sd
import numpy as np
import torch
import os
import threading
import time
from queue import Queue
from datetime import datetime
import wave


class HybridSpeakerTranscription:
    """Hybrid system: Live streaming + Post-meeting accurate labels."""
    
    def __init__(self, hf_token=None, sample_rate=16000, output_dir="transcripts"):
        self.sample_rate = sample_rate
        self.is_running = False
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.audio_file = os.path.join(output_dir, f"meeting_{self.session_id}.wav")
        self.live_transcript_file = os.path.join(output_dir, f"transcript_live_{self.session_id}.txt")
        self.final_transcript_file = os.path.join(output_dir, f"transcript_final_{self.session_id}.txt")
        
        # Audio recording
        self.audio_buffer = []
        self.wav_file = None
        
        # Queue for transcription
        self.processing_queue = Queue()
        
        # VAD settings
        self.vad_window_size = int(0.5 * sample_rate)
        self.silence_threshold = 0.01
        self.min_silence_duration = 1.5
        self.min_speech_duration = 1.0
        
        # Store HF token for later
        self.hf_token = hf_token
        
        print("Loading Zipformer STT model...")
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
        
        print("✅ STT model loaded!\n")
        
        # State tracking
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_counter = 0
        self.speech_start_time = 0
        self.total_time = 0
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()
    
    def calculate_rms(self, audio_chunk):
        """Calculate RMS."""
        return np.sqrt(np.mean(audio_chunk**2))
    
    def processing_worker(self):
        """Background worker for live transcription."""
        with open(self.live_transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"# Live Transcript - {self.session_id}\n")
            f.write(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        while True:
            item = self.processing_queue.get()
            
            if item is None:
                break
            
            audio_data, start_time = item
            duration = len(audio_data) / self.sample_rate
            
            if duration < self.min_speech_duration:
                self.processing_queue.task_done()
                continue
            
            # Transcribe WITHOUT diarization (fast!)
            try:
                stream = self.recognizer.create_stream()
                stream.accept_waveform(self.sample_rate, audio_data)
                self.recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                
                if text:
                    timestamp = self.format_time(start_time)
                    output = f"[{timestamp}] {text}"
                    
                    print(f"\n{output}")
                    
                    # Save to file
                    with open(self.live_transcript_file, 'a', encoding='utf-8') as f:
                        f.write(output + '\n')
                        
            except Exception as e:
                print(f"   ⚠️  Transcription error: {e}")
            
            self.processing_queue.task_done()
    
    def format_time(self, seconds):
        """Format seconds to MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback - records and transcribes."""
        if status:
            print(f"Status: {status}")
        
        audio_chunk = indata.flatten().copy()
        
        # Save to WAV file
        self.audio_buffer.append(audio_chunk)
        
        # VAD for transcription
        rms = self.calculate_rms(audio_chunk)
        
        if rms > self.silence_threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = self.total_time
                print("🎤 Speaking...", end='\r')
            
            self.speech_buffer.append(audio_chunk)
            self.silence_counter = 0
        else:
            if self.is_speaking:
                self.silence_counter += len(audio_chunk) / self.sample_rate
                self.speech_buffer.append(audio_chunk)
                
                if self.silence_counter >= self.min_silence_duration:
                    if self.speech_buffer:
                        audio_data = np.concatenate(self.speech_buffer)
                        self.processing_queue.put((audio_data.copy(), self.speech_start_time))
                    
                    self.speech_buffer = []
                    self.is_speaking = False
                    self.silence_counter = 0
        
        self.total_time += len(audio_chunk) / self.sample_rate
    
    def start_live_mode(self, auto_review=False, min_speakers=1, max_speakers=5):
        """Mode 1: Live transcription.
        
        Args:
            auto_review: If True, automatically run Review Mode after stopping
            min_speakers: Min speakers for auto review (default: 1)
            max_speakers: Max speakers for auto review (default: 5)
        """
        self.is_running = True
        
        print("="*70)
        print("🎙️  LIVE MODE - Real-time Transcript")
        print("="*70)
        print(f"\n📝 Live transcript: {self.live_transcript_file}")
        print(f"🔊 Recording audio: {self.audio_file}")
        print("\n✨ Speak naturally - transcript appears in real-time")
        print("   (Speaker labels will be added in Review Mode)")
        print("   Press Ctrl+C to stop.\n")
        
        # Open WAV file for writing
        self.wav_file = wave.open(self.audio_file, 'wb')
        self.wav_file.setnchannels(1)
        self.wav_file.setsampwidth(2)  # 16-bit
        self.wav_file.setframerate(self.sample_rate)
        
        # Start audio stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=int(self.vad_window_size),
            callback=self.audio_callback
        ):
            try:
                while self.is_running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\n👋 Stopping live mode...")
                
                # Save remaining audio
                if self.audio_buffer:
                    all_audio = np.concatenate(self.audio_buffer)
                    audio_int16 = (all_audio * 32767).astype(np.int16)
                    self.wav_file.writeframes(audio_int16.tobytes())
                
                self.wav_file.close()
                
                # Wait for queue
                self.processing_queue.join()
                self.processing_queue.put(None)
                self.processing_thread.join(timeout=5)
                
                print(f"✅ Audio saved: {self.audio_file}")
                print(f"✅ Live transcript saved: {self.live_transcript_file}")
                
                # Auto-review if enabled
                if auto_review:
                    print("\n" + "="*70)
                    print("🔄 Auto-starting Review Mode...")
                    print("="*70)
                    time.sleep(1)  # Brief pause
                    
                    self.review_mode(self.audio_file, min_speakers, max_speakers)
                else:
                    print(f"\n💡 Run Review Mode to add speaker labels:")
                    print(f"   python meeting_transcript.py review {self.audio_file}\n")
                
                self.is_running = False
    
    def review_mode(self, audio_file, min_speakers=1, max_speakers=5):
        """Mode 2: Add accurate speaker labels.
        
        Args:
            audio_file: Path to audio file
            min_speakers: Minimum expected speakers (default: 1)
            max_speakers: Maximum expected speakers (default: 5)
                          Set to lower value if you know speaker count
                          (e.g., max_speakers=1 for single speaker recording)
        """
        print("="*70)
        print("🔍 REVIEW MODE - Adding Speaker Labels")
        print("="*70)
        print(f"\nProcessing: {audio_file}")
        print(f"Expected speakers: {min_speakers}-{max_speakers}\n")
        
        # Load diarization pipeline
        print("Loading speaker diarization model...")
        if self.hf_token:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token
            )
        else:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
        print("✅ Loaded!\n")
        
        # Load audio
        import soundfile as sf
        audio_data, sr = sf.read(audio_file)
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        full_audio = audio_data.astype(np.float32)
        
        # Convert to tensor
        waveform = torch.from_numpy(audio_data[np.newaxis, :]).float()
        audio_dict = {"waveform": waveform, "sample_rate": sr}
        
        print("Running speaker diarization on entire recording...")
        
        # Apply speaker constraints for better accuracy
        diarization_result = pipeline(
            audio_dict,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        diarization = diarization_result.speaker_diarization
        
        num_speakers = len(set(speaker for _, _, speaker in diarization.itertracks(yield_label=True)))
        num_segments = len(list(diarization.itertracks(yield_label=True)))
        
        print(f"✅ Found {num_speakers} speaker(s), {num_segments} segment(s)\n")
        
        # Transcribe each segment with speaker labels
        print("Transcribing with speaker labels...")
        results = []
        
        MIN_SEGMENT_DURATION = 0.5
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segment_duration = segment.end - segment.start
            if segment_duration < MIN_SEGMENT_DURATION:
                continue
            
            start_sample = int(segment.start * sr)
            end_sample = int(segment.end * sr)
            segment_audio = full_audio[start_sample:end_sample]
            
            try:
                stream = self.recognizer.create_stream()
                stream.accept_waveform(sr, segment_audio)
                self.recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                
                if text:
                    results.append({
                        'start': segment.start,
                        'end': segment.end,
                        'speaker': speaker,
                        'text': text
                    })
                    print(f"  [{self.format_time(segment.start)}] {speaker}: {text}")
            except Exception as e:
                print(f"  ⚠️  Error at {segment.start:.1f}s: {e}")
        
        # Save final transcript
        print(f"\n📝 Saving final transcript...")
        
        with open(self.final_transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"# Final Transcript with Speaker Labels\n")
            f.write(f"# Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Audio: {audio_file}\n")
            f.write(f"# Speakers: {num_speakers}\n\n")
            
            for r in results:
                timestamp = self.format_time(r['start'])
                f.write(f"[{timestamp}] {r['speaker']}: {r['text']}\n")
        
        print(f"✅ Final transcript saved: {self.final_transcript_file}")
        
        # Summary
        print(f"\n" + "="*70)
        print(f"📊 Summary:")
        print(f"   Total speakers: {num_speakers}")
        
        speakers = set(r['speaker'] for r in results)
        for speaker in sorted(speakers):
            count = sum(1 for r in results if r['speaker'] == speaker)
            print(f"   - {speaker}: {count} segments")
        
        print("="*70 + "\n")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Live mode:   python meeting_transcript.py live")
        print("  Auto mode:   python meeting_transcript.py auto [min_speakers] [max_speakers]")
        print("  Review mode: python meeting_transcript.py review <audio_file.wav> [min_speakers] [max_speakers]")
        print("\nExamples:")
        print("  python meeting_transcript.py live")
        print("  python meeting_transcript.py auto 1 1  # Live + Auto review with 1 speaker")
        print("  python meeting_transcript.py review transcripts/meeting_20231217_083000.wav")
        print("  python meeting_transcript.py review audio.wav 1 1  # Force single speaker")
        print("  python meeting_transcript.py review audio.wav 2 3  # 2-3 speakers expected")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    hf_token = None  # Will use logged-in token
    
    system = HybridSpeakerTranscription(hf_token)
    
    if mode == "live":
        system.start_live_mode(auto_review=False)
    elif mode == "auto":
        # Auto mode: Live + automatic Review
        min_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        max_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        
        print(f"🤖 AUTO MODE: Live transcript → Auto review ({min_speakers}-{max_speakers} speakers)\n")
        system.start_live_mode(auto_review=True, min_speakers=min_speakers, max_speakers=max_speakers)
    elif mode == "review":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide audio file for review mode")
            print("   python meeting_transcript.py review <audio_file.wav> [min_speakers] [max_speakers]")
            sys.exit(1)
        
        audio_file = sys.argv[2]
        if not os.path.exists(audio_file):
            print(f"❌ Error: File not found: {audio_file}")
            sys.exit(1)
        
        # Optional speaker count parameters
        min_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        max_speakers = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        
        system.review_mode(audio_file, min_speakers, max_speakers)
    else:
        print(f"❌ Error: Unknown mode '{mode}'")
        print("   Use 'live', 'auto', or 'review'")
        sys.exit(1)


if __name__ == "__main__":
    main()
