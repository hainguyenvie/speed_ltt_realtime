# PhoWhisper-base Test Script

Script để test model Speech-to-Text tiếng Việt từ VinAI Research.

## Model Information

- **Model**: [vinai/PhoWhisper-base](https://huggingface.co/vinai/PhoWhisper-base)
- **Task**: Automatic Speech Recognition (ASR) cho tiếng Việt
- **Base Model**: Whisper (OpenAI)
- **Training Data**: 844 giờ audio với nhiều giọng Việt Nam khác nhau

## Cài đặt

### Phương pháp 1: Sử dụng Conda (Khuyến nghị)

```bash
# Tạo environment mới
conda create -n stt python=3.10 -y

# Kích hoạt environment
conda activate stt

# Cài đặt dependencies
pip install torch torchaudio transformers accelerate librosa soundfile sounddevice
```

Hoặc sử dụng script tự động:

```bash
./setup_conda.sh
```

### Phương pháp 2: Sử dụng pip

```bash
pip install -r requirements.txt
```

### 2. (Optional) Cài đặt FFmpeg nếu chưa có

FFmpeg cần thiết để xử lý các định dạng audio khác nhau:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## Sử dụng

### Real-time Transcription (Khuyến nghị)

Ghi âm và transcribe liên tục (real-time):

```bash
# Kích hoạt environment
conda activate stt

# Chạy real-time (model base - cân bằng tốc độ/chất lượng)
python realtime_transcript.py

# Với model khác
python realtime_transcript.py --model vinai/PhoWhisper-small

# Tùy chỉnh chunk duration (giây)
python realtime_transcript.py --chunk-duration 2.0
```

**Xem chi tiết**: [REALTIME_GUIDE.md](REALTIME_GUIDE.md)

---

### Cách 1: Sử dụng script CLI

```bash
python test_phowhisper.py path/to/audio.wav
```

Với các options:

```bash
# Chỉ định model khác (base, small, medium, large)
python test_phowhisper.py audio.wav --model vinai/PhoWhisper-small

# Chỉ định device (cuda/cpu)
python test_phowhisper.py audio.wav --device cuda
python test_phowhisper.py audio.wav --device cpu
```

### Cách 2: Import trong code của bạn

```python
from test_phowhisper import transcribe_audio

# Transcribe một file audio
text = transcribe_audio("path/to/audio.wav")
print(text)

# Với GPU
text = transcribe_audio("path/to/audio.wav", device="cuda")

# Với model khác
text = transcribe_audio("path/to/audio.wav", model_name="vinai/PhoWhisper-small")
```

## Các phiên bản model khả dụng

- `vinai/PhoWhisper-tiny` - Nhỏ nhất, nhanh nhất
- `vinai/PhoWhisper-base` - Cân bằng (mặc định)
- `vinai/PhoWhisper-small` - Tốt hơn base
- `vinai/PhoWhisper-medium` - Chất lượng cao
- `vinai/PhoWhisper-large` - Tốt nhất, chậm nhất

## Định dạng audio hỗ trợ

Script hỗ trợ nhiều định dạng audio:
- WAV
- MP3
- M4A
- FLAC
- OGG
- và nhiều format khác

## Ví dụ

```bash
# Test với file WAV
python test_phowhisper.py sample.wav

# Test với file MP3
python test_phowhisper.py recording.mp3

# Sử dụng GPU để xử lý nhanh hơn
python test_phowhisper.py long_audio.wav --device cuda

# Sử dụng model lớn hơn để có kết quả tốt hơn
python test_phowhisper.py audio.wav --model vinai/PhoWhisper-medium
```

## Lưu ý

1. Lần đầu chạy sẽ tải model từ Hugging Face (khoảng 290MB cho base model)
2. Nếu có GPU, script sẽ tự động sử dụng để xử lý nhanh hơn
3. Kết quả tốt nhất với audio rõ ràng, ít nhiễu
4. Model được train với nhiều giọng Việt Nam khác nhau nên có độ chính xác cao

## Citation

```bibtex
@inproceedings{PhoWhisper,
  title = {{PhoWhisper: Automatic Speech Recognition for Vietnamese}},
  author = {Thanh-Thien Le and Linh The Nguyen and Dat Quoc Nguyen},
  booktitle = {Proceedings of the ICLR 2024 Tiny Papers track},
  year = {2024}
}
```

## Reference

- [PhoWhisper GitHub](https://github.com/VinAIResearch/PhoWhisper)
- [PhoWhisper Hugging Face](https://huggingface.co/vinai/PhoWhisper-base)
