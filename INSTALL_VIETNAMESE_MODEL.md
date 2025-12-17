# 🇻🇳 Hướng Dẫn Cài Đặt Model Vietnamese Sherpa-ONNX

Hướng dẫn chi tiết cách tải và cài đặt model **sherpa-onnx-zipformer-vi-2025-04-20** - model nhận dạng tiếng Việt tốt nhất hiện nay với 70,000 giờ training data.

## 📋 Mục Lục

1. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
2. [Cài Đặt Dependencies](#cài-đặt-dependencies)
3. [Tải Model](#tải-model)
4. [Cấu Hình Model](#cấu-hình-model)
5. [Kiểm Tra Model](#kiểm-tra-model)
6. [Troubleshooting](#troubleshooting)

---

## Yêu Cầu Hệ Thống

### Phần Cứng
- **CPU**: Intel/AMD đời mới (khuyến nghị 4+ cores)
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB+)
- **Disk**: ~300MB cho model
- **GPU**: Không bắt buộc (CPU đủ nhanh)

### Phần Mềm
- **OS**: Linux, macOS, hoặc Windows
- **Python**: 3.7 trở lên (khuyến nghị 3.8+)
- **pip**: Phiên bản mới nhất

---

## Cài Đặt Dependencies

### Bước 1: Cài đặt Python packages

```bash
pip install sherpa-onnx sounddevice numpy
```

**Chi tiết các packages:**
- `sherpa-onnx`: Framework chạy model ONNX
- `sounddevice`: Thu âm từ microphone
- `numpy`: Xử lý audio array

### Bước 2: Kiểm tra cài đặt

```bash
python3 -c "import sherpa_onnx; print('✅ sherpa-onnx version:', sherpa_onnx.__version__)"
python3 -c "import sounddevice; print('✅ sounddevice installed')"
```

**Expected output:**
```
✅ sherpa-onnx version: [version number]
✅ sounddevice installed
```

---

## Tải Model

### Phương Pháp 1: Download từ GitHub Releases (Khuyến nghị)

**Bước 1:** Download model archive

```bash
cd /path/to/your/project
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-2025-04-20.tar.bz2
```

**Bước 2:** Extract model files

```bash
tar xvf sherpa-onnx-zipformer-vi-2025-04-20.tar.bz2
```

**Bước 3:** Xóa file archive (tùy chọn)

```bash
rm sherpa-onnx-zipformer-vi-2025-04-20.tar.bz2
```

### Phương Pháp 2: Download từ Hugging Face

```bash
# Cài git-lfs nếu chưa có
git lfs install

# Clone repository
git clone https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20
```

### Kiểm tra files đã tải

```bash
ls -lh sherpa-onnx-zipformer-vi-2025-04-20/
```

**Expected output:**
```
encoder-epoch-12-avg-8.onnx  (~249 MB)
decoder-epoch-12-avg-8.onnx  (~4.9 MB)
joiner-epoch-12-avg-8.onnx   (~3.9 MB)
tokens.txt                   (~26 KB)
bpe.model                    (~270 KB)
test_wavs/                   (sample audio files)
README.md
```

---

## Cấu Hình Model

### Code Template Cơ Bản

Tạo file `test_model.py`:

```python
#!/usr/bin/env python3
import sherpa_onnx
import os

def create_recognizer(model_dir="sherpa-onnx-zipformer-vi-2025-04-20"):
    """Initialize Vietnamese ASR model."""
    
    print("🔄 Loading model...")
    
    # Model file paths
    encoder = os.path.join(model_dir, "encoder-epoch-12-avg-8.onnx")
    decoder = os.path.join(model_dir, "decoder-epoch-12-avg-8.onnx")
    joiner = os.path.join(model_dir, "joiner-epoch-12-avg-8.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")
    
    # Create recognizer
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        num_threads=4,              # Số CPU threads
        sample_rate=16000,          # Sample rate (16kHz)
        feature_dim=80,             # Feature dimension
        decoding_method="modified_beam_search",  # Decoding method
        max_active_paths=4,         # Beam search paths
    )
    
    print("✅ Model loaded successfully!")
    return recognizer

if __name__ == "__main__":
    recognizer = create_recognizer()
```

### Các Tham Số Quan Trọng

| Tham số | Giá trị mặc định | Mô tả |
|---------|------------------|-------|
| `num_threads` | 4 | Số CPU threads (1-12) |
| `sample_rate` | 16000 | Sample rate (Hz) |
| `decoding_method` | `modified_beam_search` | Phương pháp decode |
| `max_active_paths` | 4 | Số paths trong beam search (1-16) |

**Tối ưu theo use case:**

```python
# ⚡ Maximum Speed (nhanh nhất)
num_threads=12, max_active_paths=2, decoding_method="modified_beam_search"

# ⚖️ Balanced (cân bằng - khuyến nghị)
num_threads=4, max_active_paths=4, decoding_method="modified_beam_search"

# 🎯 Maximum Accuracy (chính xác nhất)
num_threads=8, max_active_paths=8, decoding_method="modified_beam_search"

# 🚀 Greedy Fast (cực nhanh, accuracy giảm)
num_threads=12, decoding_method="greedy_search"
```

---

## Kiểm Tra Model

### Test 1: Kiểm tra model load thành công

```bash
python3 test_model.py
```

**Expected output:**
```
🔄 Loading model...
✅ Model loaded successfully!
```

### Test 2: Test với sample audio

Tạo file `quick_test.py`:

```python
#!/usr/bin/env python3
import sherpa_onnx
import os

model_dir = "sherpa-onnx-zipformer-vi-2025-04-20"

# Load model
encoder = os.path.join(model_dir, "encoder-epoch-12-avg-8.onnx")
decoder = os.path.join(model_dir, "decoder-epoch-12-avg-8.onnx")
joiner = os.path.join(model_dir, "joiner-epoch-12-avg-8.onnx")
tokens = os.path.join(model_dir, "tokens.txt")

recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=encoder, decoder=decoder, joiner=joiner, tokens=tokens,
    num_threads=4, sample_rate=16000, feature_dim=80,
    decoding_method="modified_beam_search", max_active_paths=4,
)

# Test with sample audio (if available)
test_wav = os.path.join(model_dir, "test_wavs", "0.wav")
if os.path.exists(test_wav):
    print(f"✅ Model ready! Sample audio found: {test_wav}")
else:
    print("✅ Model ready! No sample audio found.")
```

### Test 3: Test với microphone

```bash
python3 quick_test_vietnamese.py
```

Nói tiếng Việt trong 5 giây để kiểm tra.

---

## Troubleshooting

### Lỗi: Model files not found

**Triệu chứng:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'encoder-epoch-12-avg-8.onnx'
```

**Giải pháp:**
1. Kiểm tra model đã được extract:
   ```bash
   ls -lh sherpa-onnx-zipformer-vi-2025-04-20/
   ```

2. Đảm bảo đường dẫn đúng trong code:
   ```python
   model_dir = "sherpa-onnx-zipformer-vi-2025-04-20"  # Relative path
   # hoặc
   model_dir = "/absolute/path/to/sherpa-onnx-zipformer-vi-2025-04-20"
   ```

### Lỗi: sherpa_onnx not found

**Triệu chứng:**
```
ModuleNotFoundError: No module named 'sherpa_onnx'
```

**Giải pháp:**
```bash
pip install --upgrade sherpa-onnx
```

### Lỗi: sounddevice input overflow

**Triệu chứng:**
```
sounddevice.PortAudioError: Input overflowed
```

**Giải pháp:**
```python
# Tăng blocksize
sd.InputStream(blocksize=2048)  # Thay vì 1024
```

### Model chạy chậm

**Nguyên nhân:** CPU threads không tối ưu

**Giải pháp:**
1. Check số CPU cores:
   ```bash
   nproc  # Linux/Mac
   ```

2. Điều chỉnh `num_threads`:
   ```python
   num_threads=8  # Set = số cores hoặc cores/2
   ```

### Accuracy không cao

**Giải pháp:**
1. Tăng `max_active_paths`:
   ```python
   max_active_paths=8  # Thay vì 4
   ```

2. Sử dụng audio chất lượng tốt:
   - Sample rate: 16kHz
   - Mono channel
   - Ít background noise

---

## So Sánh Với Model Cũ

### Model Cũ: zipformer-30m-rnnt-6000h

```python
# OLD configuration
model_dir = "zipformer-30m-rnnt-6000h"
encoder = "encoder-epoch-20-avg-10.int8.onnx"
decoder = "decoder-epoch-20-avg-10.int8.onnx"
joiner = "joiner-epoch-20-avg-10.int8.onnx"
tokens = "config.json"
decoding_method = "greedy_search"
```

### Model Mới: sherpa-onnx-zipformer-vi-2025-04-20

```python
# NEW configuration
model_dir = "sherpa-onnx-zipformer-vi-2025-04-20"
encoder = "encoder-epoch-12-avg-8.onnx"
decoder = "decoder-epoch-12-avg-8.onnx"
joiner = "joiner-epoch-12-avg-8.onnx"
tokens = "tokens.txt"
decoding_method = "modified_beam_search"
max_active_paths = 4
```

### Sự Khác Biệt

| Aspect | Model Cũ | Model Mới |
|--------|----------|-----------|
| Training data | 6,000 giờ | **70,000 giờ** |
| Precision | INT8 quantized | **Float32** |
| Decoding | Greedy search | **Modified beam search** |
| Speed | Nhanh hơn | Chậm hơn ~2x |
| Accuracy | Tốt | **Rất tốt** |
| WER | ~12-17% | **~7-14%** |

---

## Scripts Có Sẵn

Sau khi setup xong, bạn có thể sử dụng các scripts sau:

### 1. Quick Test (5 seconds)
```bash
python3 quick_test_vietnamese.py
```

### 2. Interactive Test Menu
```bash
python3 test_vietnamese_model.py
```

### 3. Real-time VAD Transcription
```bash
python3 realtime_transcript.py
```

### 4. Configuration Optimizer
```bash
python3 optimize_config.py
```

---

## Tối Ưu Hiệu Năng

### For CPU (Recommended)

```python
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=encoder,
    decoder=decoder,
    joiner=joiner,
    tokens=tokens,
    num_threads=4,                          # ✅ Optimal cho CPU
    sample_rate=16000,
    feature_dim=80,
    decoding_method="modified_beam_search", # ✅ Best accuracy
    max_active_paths=4,                     # ✅ Good balance
)
```

**Performance:** RTF ~0.02x (50x faster than real-time)

### For GPU (Optional - nếu có NVIDIA GPU)

```bash
# Install CUDA version
pip install sherpa-onnx-cuda

# Use in code
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    ...,
    provider="cuda",  # Enable GPU
)
```

**Performance:** RTF ~0.005x (200x faster than real-time)

> [!NOTE]
> GPU không bắt buộc! CPU đã đủ nhanh cho real-time transcription.

---

## Tài Liệu Tham Khảo

- **Model Repository**: https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20
- **Sherpa-ONNX Docs**: https://k2-fsa.github.io/sherpa/
- **GitHub Release**: https://github.com/k2-fsa/sherpa-onnx/releases

---

## Changelog

### v1.0 (2025-04-20)
- ✅ Model release với 70,000 giờ training
- ✅ Float32 precision
- ✅ Modified beam search support
- ✅ Best Vietnamese ASR accuracy

---

## Hỗ Trợ

Nếu gặp vấn đề:

1. Check [VIETNAMESE_MODEL_TESTING.md](VIETNAMESE_MODEL_TESTING.md) để biết cách test
2. Xem [Troubleshooting](#troubleshooting) section
3. Check sherpa-onnx documentation: https://k2-fsa.github.io/sherpa/

---

**Happy transcribing! 🎉**
