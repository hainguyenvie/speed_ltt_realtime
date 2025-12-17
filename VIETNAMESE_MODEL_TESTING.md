# 🇻🇳 Testing Vietnamese Speech Recognition Model

Model: **sherpa-onnx-zipformer-vi-2025-04-20** (70,000 hours training)

## Quick Start

### 🚀 Option 1: Quick 5-Second Test

Fastest way to test - just run and speak:

```bash
python3 quick_test_vietnamese.py
```

This will:
- Load the model
- Record 5 seconds from your microphone
- Show the transcription result

### 📋 Option 2: Interactive Test Menu

Full-featured testing with multiple options:

```bash
python3 test_vietnamese_model.py
```

**Features:**
1. Test with sample audio files (included in model)
2. Test with custom audio file
3. Record from microphone (5 seconds)
4. Record from microphone (10 seconds)
5. Test with recording.wav (if exists)

### 🎙️ Option 3: Real-time VAD Transcription

Continuous recording with automatic transcription:

```bash
python3 realtime_transcript.py
```

**How it works:**
- Start speaking → automatically starts recording
- Stop speaking → transcribes after 1.5s of silence
- Keep talking → continues until you pause

Press `Ctrl+C` to stop.

## Test Scripts Overview

| Script | Purpose | Best For |
|--------|---------|----------|
| `quick_test_vietnamese.py` | Simple 5-second test | Quick verification |
| `test_vietnamese_model.py` | Interactive menu | Comprehensive testing |
| `realtime_transcript.py` | Continuous VAD mode | Real-world usage |

## Model Specifications

- **Model**: sherpa-onnx-zipformer-vi-2025-04-20
- **Training**: 70,000 hours Vietnamese speech
- **Precision**: Float32 (high accuracy)
- **Decoding**: Modified Beam Search (max_active_paths=4)
- **Sample Rate**: 16kHz
- **Architecture**: Zipformer-RNN-Transducer

## Expected Performance

- **WER (Word Error Rate)**: 7-14% depending on dataset
- **Processing Speed**: ~0.5-2x real-time (slower but more accurate)
- **Accuracy**: Best-in-class for Vietnamese

## Tips for Best Results

1. **Speak clearly** in Vietnamese
2. **Minimize background noise**
3. **Use a good microphone** if available
4. **Speak at normal pace** - not too fast or slow
5. **Wait for silence** in VAD mode to trigger transcription

## Audio File Format

When testing with audio files:
- **Format**: WAV (recommended)
- **Sample Rate**: 16kHz (auto-resampling supported)
- **Channels**: Mono (stereo auto-converted)
- **Bit Depth**: 16-bit PCM

## Example Usage

### Quick Test
```bash
# Test immediately with 5-second recording
python3 quick_test_vietnamese.py
```

### Interactive Testing
```bash
# Run interactive menu
python3 test_vietnamese_model.py

# Select option:
# 1 - Test sample audios
# 2 - Test your audio file
# 3 - Record 5 seconds
# 4 - Record 10 seconds
```

### Real-time Continuous
```bash
# For natural conversation transcription
python3 realtime_transcript.py
```

## Troubleshooting

**Model not found?**
```bash
# Check if model directory exists
ls -lh sherpa-onnx-zipformer-vi-2025-04-20/
```

**Audio device issues?**
```bash
# List available audio devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

**ImportError?**
```bash
# Install required packages
pip install sherpa-onnx sounddevice numpy
```

## Comparison with Previous Model

| Aspect | Old (zipformer-30m) | New (sherpa-onnx-vi) |
|--------|---------------------|----------------------|
| Training Data | 6,000 hours | 70,000 hours |
| Precision | INT8 quantized | Float32 |
| Decoding | Greedy search | Modified beam search |
| Speed | Faster | Slower |
| Accuracy | Good | **Better** |

## Sample Test Phrases (Vietnamese)

Try testing with these phrases:

- "Xin chào, hôm nay thời tiết thế nào?"
- "Tôi muốn đặt một chiếc bánh pizza lớn"
- "Vui lòng cho tôi biết giá của sản phẩm này"
- "Hẹn gặp lại bạn vào tuần sau"

Enjoy testing! 🎉
