# QUICKSTART - PhoWhisper Test

## Setup đã hoàn tất ✓

Environment `stt` đã được tạo và đang cài đặt dependencies.

## Cách sử dụng

### 1. Kích hoạt environment

```bash
conda activate stt
```

### 2. Kiểm tra setup

```bash
python quick_test.py
```

### 3. Test với file audio

```bash
# Nếu bạn có file audio tiếng Việt
python test_phowhisper.py path/to/audio.wav

# Hoặc
python test_phowhisper.py path/to/audio.mp3
```

### 4. Ghi âm và test trực tiếp

```bash
# Ghi âm 5 giây và transcribe luôn
python record_audio.py --transcribe

# Ghi âm 10 giây
python record_audio.py --duration 10 --transcribe

# Ghi âm và lưu file
python record_audio.py --output my_recording.wav --duration 10
```

## Ví dụ Output

```
==================================================
TRANSCRIPTION:
==================================================
Xin chào, đây là bài test speech to text cho tiếng Việt
==================================================
```

## Lưu ý

1. **Lần đầu chạy**: Model sẽ tự động download từ Hugging Face (~290MB)
2. **GPU**: Nếu có CUDA, script sẽ tự động sử dụng GPU
3. **Tốc độ**: CPU cũng chạy được nhưng chậm hơn GPU
4. **Định dạng**: Hỗ trợ WAV, MP3, M4A, FLAC, OGG, etc.

## Commands hữu ích

```bash
# Xem tất cả options
python test_phowhisper.py --help

# Force sử dụng CPU
python test_phowhisper.py audio.wav --device cpu

# Sử dụng model lớn hơn (chất lượng tốt hơn)
python test_phowhisper.py audio.wav --model vinai/PhoWhisper-medium
```

## Troubleshooting

### Nếu gặp lỗi import
```bash
conda activate stt
pip install torch transformers accelerate
```

### Nếu không nhận microphone
```bash
pip install sounddevice soundfile
```

### Kiểm tra CUDA
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
