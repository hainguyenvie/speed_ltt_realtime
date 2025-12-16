#!/bin/bash

echo "================================================"
echo "Cài đặt PhoWhisper Test Environment"
echo "================================================"

# Check Python version
echo ""
echo "1. Kiểm tra Python version..."
python3 --version

# Upgrade pip
echo ""
echo "2. Upgrade pip..."
python3 -m pip install --upgrade pip

# Install requirements
echo ""
echo "3. Cài đặt dependencies..."
pip install -r requirements.txt

# Optional: Install audio recording dependencies
echo ""
read -p "Bạn có muốn cài thêm sounddevice để ghi âm từ microphone? (y/n): " install_audio
if [ "$install_audio" = "y" ] || [ "$install_audio" = "Y" ]; then
    echo "Đang cài đặt sounddevice..."
    pip install sounddevice soundfile
fi

# Test installation
echo ""
echo "4. Kiểm tra cài đặt..."
python3 quick_test.py

echo ""
echo "================================================"
echo "✓ Hoàn tất!"
echo "================================================"
echo ""
echo "Bước tiếp theo:"
echo "1. Chuẩn bị file audio tiếng Việt"
echo "2. Chạy: python3 test_phowhisper.py <file-audio>"
echo ""
echo "Hoặc ghi âm trực tiếp:"
echo "   python3 record_audio.py --transcribe"
echo ""
