#!/bin/bash
# Guitar Tab Retriever - 環境設定腳本
# 用法: bash setup.sh

set -e

echo "=== Guitar Tab Retriever 環境設定 ==="
echo ""

# 檢查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

required_major=3
required_minor=10
actual_major=$(echo "$python_version" | cut -d. -f1)
actual_minor=$(echo "$python_version" | cut -d. -f2)

if [ "$actual_major" -lt "$required_major" ] || { [ "$actual_major" -eq "$required_major" ] && [ "$actual_minor" -lt "$required_minor" ]; }; then
    echo "錯誤: 需要 Python >= 3.10，目前為 $python_version"
    exit 1
fi

# 建立虛擬環境
echo ""
echo "--- 建立虛擬環境 ---"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "虛擬環境已建立: ./venv"
else
    echo "虛擬環境已存在，跳過建立"
fi

# 啟動虛擬環境
source venv/bin/activate
echo "虛擬環境已啟動"

# 升級 pip
pip install --upgrade pip

# 安裝依賴
echo ""
echo "--- 安裝 Python 套件 ---"
echo "注意: 這會安裝 PyTorch，可能需要幾分鐘..."
pip install -r requirements.txt

# 檢查 ffmpeg
echo ""
echo "--- 檢查 ffmpeg ---"
if command -v ffmpeg &> /dev/null; then
    ffmpeg_version=$(ffmpeg -version 2>&1 | head -n1)
    echo "ffmpeg 已安裝: $ffmpeg_version"
else
    echo "警告: ffmpeg 未安裝！"
    echo "請依照你的系統安裝 ffmpeg:"
    echo "  macOS:   brew install ffmpeg"
    echo "  Ubuntu:  sudo apt install ffmpeg"
    echo "  Windows: https://ffmpeg.org/download.html"
fi

# 檢查 yt-dlp
echo ""
echo "--- 驗證安裝 ---"
python3 -c "import demucs; print(f'  demucs: OK')" 2>/dev/null || echo "  demucs: 安裝失敗"
python3 -c "import basic_pitch; print(f'  basic_pitch: OK')" 2>/dev/null || echo "  basic_pitch: 安裝失敗"
python3 -c "import librosa; print(f'  librosa: OK')" 2>/dev/null || echo "  librosa: 安裝失敗"
python3 -c "import pretty_midi; print(f'  pretty_midi: OK')" 2>/dev/null || echo "  pretty_midi: 安裝失敗"

which yt-dlp &> /dev/null && echo "  yt-dlp: OK" || echo "  yt-dlp: 未找到"

echo ""
echo "=== 設定完成 ==="
echo ""
echo "使用方式:"
echo "  source venv/bin/activate"
echo "  python main.py <YouTube_URL>"
echo ""
echo "範例:"
echo "  python main.py https://www.youtube.com/watch?v=VIf6U2QrjCs"
echo "  python main.py https://www.youtube.com/watch?v=VIf6U2QrjCs --skip-separation"
