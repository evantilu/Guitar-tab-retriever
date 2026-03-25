"""
模組 2: 音源分離
使用 Meta 的 Demucs 模型將混合音訊分離出吉他軌道
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional


# Demucs 模型選項（由快到精確）
MODELS = {
    "htdemucs": "HTDemucs（預設，速度與品質平衡）",
    "htdemucs_ft": "HTDemucs Fine-tuned（品質最好，較慢）",
    "mdx_extra": "MDX-Net Extra（替代方案）",
}

# Demucs 分離出的標準音軌名稱
STEMS = ["drums", "bass", "other", "vocals"]


def check_demucs_installed() -> bool:
    """檢查 demucs 是否已安裝"""
    try:
        import demucs
        return True
    except ImportError:
        return False


def separate_sources(
    audio_path: str | Path,
    output_dir: str | Path,
    model: str = "htdemucs",
    device: str = "auto",
    two_stems: Optional[str] = None,
) -> dict[str, Path]:
    """
    使用 Demucs 進行音源分離。

    Args:
        audio_path: 輸入音訊檔案路徑
        output_dir: 輸出目錄
        model: Demucs 模型名稱
        device: 運算裝置 ("cpu", "cuda", "auto")
        two_stems: 如果指定，只分離為兩軌（例如 "vocals" 會分成 vocals + no_vocals）

    Returns:
        字典，key 為音軌名稱，value 為對應檔案路徑
    """
    if not check_demucs_installed():
        raise RuntimeError(
            "demucs 未安裝。請執行: pip install demucs"
        )

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"音訊檔案不存在: {audio_path}")

    # 建構 demucs 命令
    cmd = [
        "python", "-m", "demucs",
        "--out", str(output_dir),
        "--name", model,
        "--filename", "{stem}.{ext}",
    ]

    # 裝置設定
    if device == "auto":
        pass  # demucs 會自動偵測
    elif device == "cpu":
        cmd.extend(["--device", "cpu"])
    elif device == "cuda":
        cmd.extend(["--device", "cuda"])

    # 雙軌模式
    if two_stems:
        cmd.extend(["--two-stems", two_stems])

    cmd.append(str(audio_path))

    print(f"正在進行音源分離（模型: {model}）...")
    print("  這可能需要幾分鐘，取決於音訊長度和你的硬體。")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Demucs 音源分離失敗:\n{result.stderr}"
        )

    # 找到輸出的音軌檔案
    # Demucs 的輸出結構: output_dir / model / stem.wav
    stem_dir = output_dir / model
    stems = {}

    for stem_file in stem_dir.glob("*.wav"):
        stem_name = stem_file.stem
        stems[stem_name] = stem_file

    if not stems:
        raise RuntimeError(
            f"音源分離未產生任何輸出檔案。請檢查 {stem_dir}"
        )

    print(f"音源分離完成，產生了 {len(stems)} 個音軌:")
    for name, path in stems.items():
        print(f"  - {name}: {path}")

    return stems


def get_guitar_track(stems: dict[str, Path]) -> Path:
    """
    從分離出的音軌中取得最可能包含吉他的軌道。

    Demucs 標準四軌模式中，吉他通常在 "other" 軌道中
    （因為標準分離是 drums / bass / vocals / other）。

    Args:
        stems: separate_sources 回傳的音軌字典

    Returns:
        最可能包含吉他的音軌路徑
    """
    # 優先順序：other > no_vocals > 第一個非 drums/bass/vocals 的軌
    priority = ["other", "no_vocals"]

    for name in priority:
        if name in stems:
            print(f"使用 '{name}' 軌道作為吉他音源")
            return stems[name]

    # fallback: 排除 drums, bass, vocals 後取第一個
    excluded = {"drums", "bass", "vocals"}
    for name, path in stems.items():
        if name not in excluded:
            print(f"使用 '{name}' 軌道作為吉他音源（fallback）")
            return path

    # 最後手段：用原始混合音訊
    raise RuntimeError(
        "無法找到適合的吉他音軌。可用音軌: "
        + ", ".join(stems.keys())
    )
