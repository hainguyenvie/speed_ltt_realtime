# 🎙️ Voice Recorder + Speech to Text

Ghi âm từ microphone và tự động chuyển thành text bằng Zipformer model.

## Sử dụng

```bash
python voice_recorder.py
```

## Controls

- **SPACE** - Bắt đầu/Dừng ghi âm
- **ESC** - Thoát chương trình

## Luồng hoạt động

1. Nhấn **SPACE** để bắt đầu ghi âm
2. Nói vào microphone
3. Nhấn **SPACE** lại để dừng
4. Model sẽ tự động transcribe và hiển thị kết quả
5. Nhấn **SPACE** để ghi âm tiếp

## Performance

- **Model**: Zipformer-30M-RNNT-6000h (INT8, 30MB)
- **Speed**: ~46x realtime (10s audio → 0.22s processing)
- **Device**: CPU only

## Ví dụ Output

```
================================================================================
🎙️  VOICE RECORDER + SPEECH TO TEXT
================================================================================

Controls:
  SPACE - Start/Stop recording
  ESC   - Quit

================================================================================
Ready! Press SPACE to start recording...
================================================================================

🔴 RECORDING... (Press SPACE to stop)
⏹️  STOPPED (Duration: 5.2s)
🔄 Transcribing...

================================================================================
📝 TRANSCRIPTION:
================================================================================
HÔM NAY TÔI MUA MỘT QUYỂN SÁCH VỀ PYTHON
================================================================================
⏱️  Processing: 0.11s (47.3x realtime)
================================================================================

Ready! Press SPACE to record again...
```

## Requirements

- sounddevice
- pynput
- sherpa-onnx
- k2_sherpa
- kaldifeat

## Troubleshooting

### Microphone không hoạt động
```bash
# List available audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### Model chưa được download
```bash
python download_zipformer.py
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('hynt/Zipformer-30M-RNNT-6000h', 'config.json', local_dir='zipformer-30m-rnnt-6000h')"
```
