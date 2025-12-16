# 🎙️ Real-time Speech-to-Text

Continuous streaming transcription with live partial results as you speak!

## Quick Start

```bash
python realtime_transcript.py
```

Then just start speaking - transcription appears in real-time!

## Features

- ✅ **Continuous streaming** - No need to press buttons
- ✅ **Partial results** - See text appear as you speak
- ✅ **Endpoint detection** - Automatically detects when you finish speaking
- ✅ **Super fast** - 46-95x realtime speed
- ✅ **CPU only** - No GPU required

## How It Works

1. Script starts listening to microphone
2. Audio is processed in **100ms chunks**
3. Partial transcription updates in real-time
4. When you pause, model detects endpoint and finalizes text
5. Ready for next utterance immediately

## Output Example

```
================================================================================
🎙️  REAL-TIME SPEECH-TO-TEXT
================================================================================

Listening... speak into your microphone!
Press Ctrl+C to stop

================================================================================

📝 HÔM NAY TÔI
📝 HÔM NAY TÔI MUA MỘT QUYỂN
📝 HÔM NAY TÔI MUA MỘT QUYỂN SÁCH

📝 VÀ TÔI  
📝 VÀ TÔI ĐANG ĐỌC
📝 VÀ TÔI ĐANG ĐỌC NÓ

^C
👋 Stopping...
```

## Technical Details

- **Model**: Zipformer-30M INT8 (streaming mode)
- **API**: `sherpa_onnx.OnlineRecognizer`
- **Latency**: ~100ms per chunk
- **Endpoint detection**: 1.2-2.4s silence
- **Min utterance**: 20 frames

## Comparison

| Feature | realtime_transcript.py | voice_recorder.py |
|---------|----------------------|-------------------|
| Mode | Continuous streaming | Press to record |
| Display | Live partial results | Final result only |
| Use case | Dictation, live captions | Voice notes |
| Latency | ~100ms | After recording ends |

## Troubleshooting

### No audio input
Check microphone permissions and device:
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### Model not found
Download model files first:
```bash
python download_zipformer.py
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('hynt/Zipformer-30M-RNNT-6000h', 'config.json', local_dir='zipformer-30m-rnnt-6000h')"
```

## Tips

- Speak clearly with moderate pace
- Brief pauses (1-2s) will trigger endpoint detection
- For best results, use in quiet environment
- Lower `blocksize` in code for lower latency (at cost of CPU)
