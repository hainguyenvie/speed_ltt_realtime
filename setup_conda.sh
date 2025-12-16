#!/bin/bash

echo "================================================"
echo "Tạo Conda Environment cho PhoWhisper"
echo "================================================"

# Create conda environment
echo ""
echo "1. Tạo conda environment 'stt' với Python 3.10..."
conda create -n stt python=3.10 -y

# Activate environment
echo ""
echo "2. Kích hoạt environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stt

# Install dependencies
echo ""
echo "3. Cài đặt PyTorch..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "4. Cài đặt các dependencies khác..."
pip install transformers accelerate librosa soundfile

# Optional: Install audio recording
echo ""
read -p "Bạn có muốn cài thêm sounddevice để ghi âm từ microphone? (y/n): " install_audio
if [ "$install_audio" = "y" ] || [ "$install_audio" = "Y" ]; then
    echo "Đang cài đặt sounddevice..."
    pip install sounddevice
fi

# Test installation
echo ""
echo "5. Kiểm tra cài đặt..."
python quick_test.py

echo ""
echo "================================================"
echo "✓ Hoàn tất!"
echo "================================================"
echo ""
echo "Để sử dụng:"
echo "1. Kích hoạt environment: conda activate stt"
echo "2. Chạy test: python test_phowhisper.py <file-audio>"
echo ""
echo "Hoặc ghi âm trực tiếp:"
echo "   python record_audio.py --transcribe"
echo ""
