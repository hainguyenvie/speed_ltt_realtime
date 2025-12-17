#!/usr/bin/env python3
"""
Continuous multi-speaker transcription with automatic speech detection.
Non-blocking: Recording continues while processing happens in background.
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


class ContinuousSpeakerTranscription:
    """Continuous recording with automatic speaker-labeled transcription."""
    
    def __init__(self, hf_token=None, sample_rate=16000):
        self.sample_rate = sample_rate
        self.is_running = False
        
        # Queue for segments to process
        self.processing_queue = Queue()
        
        # VAD settings
        self.vad_window_size = int(0.5 * sample_rate)  # 0.5s chunks
        self.silence_threshold = 0.01  # RMS threshold
        self.min_silence_duration = 1.5  # 1.5s silence to trigger
        self.min_speech_duration = 1.0  # Minimum 1s speech
        
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
        
        # State tracking for VAD
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_counter = 0
        self.speech_start_time = 0
        self.total_time = 0
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()
    
    def calculate_rms(self, audio_chunk):
        """Calculate RMS (Root Mean Square) of audio chunk."""
        return np.sqrt(np.mean(audio_chunk**2))
    
    def processing_worker(self):
        """Background worker that processes queued segments."""
        while True:
            # Get segment from queue (blocks until available)
            item = self.processing_queue.get()
            
            if item is None:  # Poison pill to stop thread
                break
            
            audio_data, start_time = item
            duration = len(audio_data) / self.sample_rate
            
            # Skip if too short
            if duration < self.min_speech_duration:
                self.processing_queue.task_done()
                continue
            
            print(f"\n🔍 Processing {duration:.1f}s segment (queue size: {self.processing_queue.qsize()})...")
            
            # Convert to torch tensor
            waveform = torch.from_numpy(audio_data[np.newaxis, :]).float()
            audio_dict = {
                "waveform": waveform,
                "sample_rate": self.sample_rate
            }
            
            try:
                # Run speaker diarization
                diarization_result = self.diarization_pipeline(audio_dict)
                diarization = diarization_result.speaker_diarization
                
                num_speakers = len(set(speaker for _, _, speaker in diarization.itertracks(yield_label=True)))
                num_segments = len(list(diarization.itertracks(yield_label=True)))
                
                print(f"   Found {num_speakers} speaker(s), {num_segments} segment(s)")
                
                # Transcribe each segment
                MIN_SEGMENT_DURATION = 0.5
                results = []
                
                for segment, _, speaker in diarization.itertracks(yield_label=True):
                    segment_duration = segment.end - segment.start
                    if segment_duration < MIN_SEGMENT_DURATION:
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
                            # Calculate absolute timestamp
                            abs_start = start_time + segment.start
                            abs_end = start_time + segment.end
                            
                            results.append({
                                'start': abs_start,
                                'end': abs_end,
                                'speaker': speaker,
                                'text': text
                            })
                    except Exception:
                        pass
                
                # Display results
                if results:
                    print("\n" + "─"*60)
                    for r in results:
                        timestamp = self.format_time(r['start'])
                        print(f"[{timestamp}] {r['speaker']}: {r['text']}")
                    print("─"*60)
                else:
                    print("   ⚠️  No speech detected")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Mark task as done
            self.processing_queue.task_done()
    
    def format_time(self, seconds):
        """Format seconds to MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream - NON-BLOCKING."""
        if status:
            print(f"Status: {status}")
        
        audio_chunk = indata.flatten().copy()
        
        # Calculate RMS
        rms = self.calculate_rms(audio_chunk)
        
        # Detect speech vs silence
        if rms > self.silence_threshold:
            # Speech detected
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = self.total_time
                print("🎤 Speaking...", end='\r')
            
            self.speech_buffer.append(audio_chunk)
            self.silence_counter = 0
        else:
            # Silence detected
            if self.is_speaking:
                self.silence_counter += len(audio_chunk) / self.sample_rate
                self.speech_buffer.append(audio_chunk)
                
                # Check if silence duration exceeded
                if self.silence_counter >= self.min_silence_duration:
                    # Queue the buffered speech for processing
                    if self.speech_buffer:
                        audio_data = np.concatenate(self.speech_buffer)
                        
                        # Add to queue (non-blocking)
                        self.processing_queue.put((audio_data.copy(), self.speech_start_time))
                        print("✓ Queued for processing             ")
                    
                    # Reset buffer - KEEP RECORDING
                    self.speech_buffer = []
                    self.is_speaking = False
                    self.silence_counter = 0
        
        self.total_time += len(audio_chunk) / self.sample_rate
    
    def start(self):
        """Start continuous transcription."""
        self.is_running = True
        
        print("="*60)
        print("🎙️  CONTINUOUS MULTI-SPEAKER TRANSCRIPTION")
        print("="*60)
        print("\n✨ Speak naturally - recording never stops!")
        print("   Processing happens in background.")
        print("   Press Ctrl+C to stop.\n")
        
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
                print("\n\n👋 Stopping... waiting for queue to finish...")
                
                # Wait for queue to be processed
                self.processing_queue.join()
                
                # Stop processing thread
                self.processing_queue.put(None)
                self.processing_thread.join(timeout=5)
                
                print("✅ Done!\n")
                self.is_running = False


def main():
    import sys
    
    hf_token = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        transcriber = ContinuousSpeakerTranscription(hf_token)
        transcriber.start()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

