# Zipformer-30M Vietnamese Speech-to-Text

Complete setup guide for **Zipformer-30M-RNNT-6000h** - Ultra-fast Vietnamese ASR model for CPU inference.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install sherpa-onnx sounddevice pynput numpy

# 2. Install special wheels (required!)
pip install "https://huggingface.co/csukuangfj/sherpa/resolve/main/cpu/1.4.0.dev20250307/linux-x64/k2_sherpa-1.4.0.dev20250307+cpu.torch1.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

pip install "https://huggingface.co/csukuangfj/kaldifeat/resolve/main/cpu/1.25.5.dev20250307/linux-x64/kaldifeat-1.25.5.dev20250307+cpu.torch1.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

# 3. Download model
python download_zipformer.py

# 4. Download config.json (critical!)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('hynt/Zipformer-30M-RNNT-6000h', 'config.json', local_dir='zipformer-30m-rnnt-6000h')"

# 5. Test it!
python test_zipformer_working.py recording.wav
```

## 📋 Table of Contents

- [Model Information](#-model-information)
- [Installation](#-installation)
- [Model Download](#-model-download)
- [Usage](#-usage)
- [Features](#-features)
- [Troubleshooting](#-troubleshooting)

## ℹ️ Model Information

**Model**: [hynt/Zipformer-30M-RNNT-6000h](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h)

- **Architecture**: Zipformer with RNN-Transducer
- **Parameters**: 30M (INT8 quantized)
- **Training Data**: 6000 hours Vietnamese speech
- **Speed**: 46-95x realtime on CPU
- **Size**: ~30MB (INT8), ~100MB (Float32)
- **Language**: Vietnamese
- **Use Case**: Fast CPU-based ASR

**Performance:**
- 10s audio → 0.22s processing time
- No GPU required
- Optimized for real-time applications

## 🔧 Installation

### Step 1: Basic Dependencies

```bash
pip install sherpa-onnx>=1.12.6
pip install sounddevice pynput numpy huggingface_hub
```

### Step 2: Special Wheels (REQUIRED!)

The model requires `k2_sherpa` and `kaldifeat` packages:

```bash
# k2_sherpa (NOT standard sherpa!)
pip install "https://huggingface.co/csukuangfj/sherpa/resolve/main/cpu/1.4.0.dev20250307/linux-x64/k2_sherpa-1.4.0.dev20250307+cpu.torch1.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

# kaldifeat
pip install "https://huggingface.co/csukuangfj/kaldifeat/resolve/main/cpu/1.25.5.dev20250307/linux-x64/kaldifeat-1.25.5.dev20250307+cpu.torch1.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
```

**Note**: This will downgrade PyTorch to 1.13.1 (required for k2 compatibility).

### Step 3: Verify Installation

```bash
python -c "import sherpa_onnx; print('✅ sherpa-onnx:', sherpa_onnx.__version__)"
python -c "import sherpa; print('✅ k2_sherpa installed')"
```

## 📥 Model Download

### Option 1: Using download script (Recommended)

```bash
python download_zipformer.py
```

This downloads:
- `encoder-epoch-20-avg-10.int8.onnx` (27.7 MB)
- `decoder-epoch-20-avg-10.int8.onnx` (1.31 MB)
- `joiner-epoch-20-avg-10.int8.onnx` (1.03 MB)
- `bpe.model` (268 KB)

### Option 2: Manual download

```python
from huggingface_hub import hf_hub_download

repo_id = "hynt/Zipformer-30M-RNNT-6000h"
local_dir = "zipformer-30m-rnnt-6000h"

files = [
    "encoder-epoch-20-avg-10.int8.onnx",
    "decoder-epoch-20-avg-10.int8.onnx",
    "joiner-epoch-20-avg-10.int8.onnx",
    "bpe.model",
    "config.json",  # CRITICAL!
]

for filename in files:
    hf_hub_download(repo_id, filename, local_dir=local_dir)
```

### ⚠️ IMPORTANT: Download config.json

**The model REQUIRES config.json (not tokens.txt):**

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('hynt/Zipformer-30M-RNNT-6000h', 'config.json', local_dir='zipformer-30m-rnnt-6000h')"
```

**Without config.json, you'll get `_Map_base::at` error!**

## 🎯 Usage

### 1. Basic Transcription

```bash
python test_zipformer_working.py recording.wav
```

**Code example:**

```python
import sherpa_onnx

# Load model
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder="zipformer-30m-rnnt-6000h/encoder-epoch-20-avg-10.int8.onnx",
    decoder="zipformer-30m-rnnt-6000h/decoder-epoch-20-avg-10.int8.onnx",
    joiner="zipformer-30m-rnnt-6000h/joiner-epoch-20-avg-10.int8.onnx",
    tokens="zipformer-30m-rnnt-6000h/config.json",  # Use config.json!
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
)

# Transcribe
stream = recognizer.create_stream()
stream.accept_waveform(sample_rate, audio_samples)
recognizer.decode_stream(stream)

result = stream.result.text
print(result)
```

### 2. Voice Recorder (Press to Talk)

```bash
python voice_recorder.py
```

- **SPACE**: Start/Stop recording
- **ESC**: Quit

Records audio when you press SPACE, transcribes when you press SPACE again.

### 3. Real-time Transcript (VAD)

```bash
python realtime_transcript.py
```

Automatically detects speech and transcribes when you stop talking:
- Start speaking → auto records
- Stop speaking (1.5s silence) → auto transcribes
- Continue speaking → continues recording

## ✨ Features

| Feature | Script | Description |
|---------|--------|-------------|
| Basic transcription | `test_zipformer_working.py` | Transcribe audio files |
| Voice recorder | `voice_recorder.py` | Press SPACE to record/transcribe |
| Real-time VAD | `realtime_transcript.py` | Auto-detect speech, transcribe on silence |
| Download tool | `download_zipformer.py` | Download model files from HF |

## 📁 Project Structure

```
.
├── zipformer-30m-rnnt-6000h/          # Model directory
│   ├── encoder-epoch-20-avg-10.int8.onnx
│   ├── decoder-epoch-20-avg-10.int8.onnx
│   ├── joiner-epoch-20-avg-10.int8.onnx
│   ├── config.json                     # REQUIRED!
│   └── bpe.model
│
├── test_zipformer_working.py           # Basic transcription
├── voice_recorder.py                   # Press-to-talk recorder
├── realtime_transcript.py              # VAD-based real-time
├── download_zipformer.py               # Model downloader
└── README.md                           # This file
```

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'sherpa'`

Wrong sherpa package installed! Uninstall and install correct one:

```bash
pip uninstall -y sherpa
pip install "https://huggingface.co/csukuangfj/sherpa/resolve/main/cpu/1.4.0.dev20250307/linux-x64/k2_sherpa-1.4.0.dev20250307+cpu.torch1.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
```

### Error: `_Map_base::at`

Missing `config.json` file! Download it:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('hynt/Zipformer-30M-RNNT-6000h', 'config.json', local_dir='zipformer-30m-rnnt-6000h')"
```

### Error: `ModuleNotFoundError: No module named 'kaldifeat'`

```bash
pip install "https://huggingface.co/csukuangfj/kaldifeat/resolve/main/cpu/1.25.5.dev20250307/linux-x64/kaldifeat-1.25.5.dev20250307+cpu.torch1.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
```

### Microphone not working

Check available devices:

```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### PyTorch version conflicts

The model requires PyTorch 1.13.1. If you have conflicts:

```bash
pip install torch==1.13.1
```

## 📊 Performance Comparison

| Model | Speed | Size | Accuracy | Best For |
|-------|-------|------|----------|----------|
| **Zipformer-30M INT8** | 46-95x | 30 MB | Good | Speed, real-time |
| Zipformer-30M Float32 | 20-30x | 100 MB | Better | Balanced |
| PhoWhisper-base | 0.67x | 500 MB | Excellent | Accuracy |

## 🔗 Links

- **Model**: https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h
- **Demo**: https://huggingface.co/spaces/hynt/k2-automatic-speech-recognition-demo
- **sherpa-onnx**: https://k2-fsa.github.io/sherpa/onnx/

## 📝 Requirements Summary

```txt
sherpa-onnx>=1.12.6
sounddevice>=0.5.0
pynput>=1.8.0
numpy<2.0
huggingface_hub
k2_sherpa==1.4.0.dev20250307+cpu.torch1.13.1 (custom wheel)
kaldifeat==1.25.5.dev20250307+cpu.torch1.13.1 (custom wheel)
```

## 🎓 Tips

1. **For best accuracy**: Use Float32 models instead of INT8
2. **For minimum latency**: Adjust `blocksize` in realtime scripts
3. **For better VAD**: Adjust `silence_threshold` and `silence_duration`
4. **Save transcripts**: Redirect output to file: `python voice_recorder.py > output.txt`

## 📄 License

Model license: See [Hugging Face model page](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h)

---

**Made with ❤️ for Vietnamese Speech Recognition**
