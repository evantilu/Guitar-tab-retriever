"""
模組 1: YouTube 音訊擷取
使用 yt-dlp 從 YouTube 下載並轉換為 WAV 格式
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional


def check_ytdlp_installed() -> bool:
    """檢查 yt-dlp 是否已安裝"""
    return shutil.which("yt-dlp") is not None


def check_ffmpeg_installed() -> bool:
    """檢查 ffmpeg 是否已安裝（yt-dlp 轉檔需要）"""
    return shutil.which("ffmpeg") is not None


def extract_audio(
    url: str,
    output_dir: str | Path,
    filename: str = "audio",
    sample_rate: int = 44100,
) -> Path:
    """
    從 YouTube 影片擷取音訊並轉為 WAV 格式。

    Args:
        url: YouTube 影片網址
        output_dir: 輸出目錄
        filename: 輸出檔名（不含副檔名）
        sample_rate: 取樣率（Hz），預設 44100

    Returns:
        輸出 WAV 檔案的路徑

    Raises:
        RuntimeError: 如果 yt-dlp 或 ffmpeg 未安裝，或下載失敗
    """
    if not check_ytdlp_installed():
        raise RuntimeError(
            "yt-dlp 未安裝。請執行: pip install yt-dlp"
        )
    if not check_ffmpeg_installed():
        raise RuntimeError(
            "ffmpeg 未安裝。請參考: https://ffmpeg.org/download.html"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{filename}.wav"

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", f"ffmpeg:-ar {sample_rate} -ac 1",
        "--output", str(output_dir / f"{filename}.%(ext)s"),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]

    print(f"正在從 YouTube 下載音訊: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp 下載失敗:\n{result.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"音訊檔案未生成: {output_path}"
        )

    print(f"音訊已儲存至: {output_path}")
    return output_path


def get_video_info(url: str) -> dict:
    """
    取得 YouTube 影片的基本資訊（標題、時長等）。

    Args:
        url: YouTube 影片網址

    Returns:
        包含 title, duration, uploader 等欄位的字典
    """
    import json

    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"無法取得影片資訊:\n{result.stderr}")

    info = json.loads(result.stdout)
    return {
        "title": info.get("title", "Unknown"),
        "duration": info.get("duration", 0),
        "uploader": info.get("uploader", "Unknown"),
        "url": url,
    }
