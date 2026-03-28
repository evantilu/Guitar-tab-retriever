"""
模組 3: 音高偵測與 MIDI 轉換
使用 Spotify 的 Basic Pitch 將音訊轉換為 MIDI 音符資料
"""

from pathlib import Path
from typing import Optional
import numpy as np


# 標準吉他調音 (E2, A2, D3, G3, B3, E4) 的 MIDI 編號
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]

# 吉他的 MIDI 音域範圍（E2=40 到約 E6=88，涵蓋 24 格）
GUITAR_MIDI_LOW = 40   # E2 (第六弦空弦)
GUITAR_MIDI_HIGH = 88  # E6 (第一弦約 24 格)


def check_basic_pitch_installed() -> bool:
    """檢查 basic-pitch 是否已安裝"""
    try:
        import basic_pitch
        return True
    except ImportError:
        return False


def audio_to_midi(
    audio_path: str | Path,
    output_dir: str | Path,
    filename: str = "transcription",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 58.0,
    minimum_frequency: Optional[float] = 80.0,
    maximum_frequency: Optional[float] = 1200.0,
) -> dict:
    """
    使用 Basic Pitch 將音訊轉換為 MIDI。

    Args:
        audio_path: 輸入音訊路徑
        output_dir: 輸出目錄
        filename: 輸出檔名（不含副檔名）
        onset_threshold: 音符起始偵測閾值 (0-1)，越高越嚴格
        frame_threshold: 音框偵測閾值 (0-1)，越高越嚴格
        minimum_note_length: 最短音符長度（毫秒）
        minimum_frequency: 最低頻率（Hz），吉他最低約 82Hz (E2)
        maximum_frequency: 最高頻率（Hz），吉他最高約 1175Hz (D6)

    Returns:
        字典包含:
        - midi_path: MIDI 檔案路徑
        - notes: 音符列表 [{start, end, pitch, velocity, confidence}, ...]
        - pitch_bends: 推弦/滑音資料
    """
    if not check_basic_pitch_installed():
        raise RuntimeError(
            "basic-pitch 未安裝。請執行: pip install basic-pitch"
        )

    from basic_pitch.inference import predict
    import pretty_midi

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"音訊檔案不存在: {audio_path}")

    print("正在進行音高偵測（Basic Pitch）...")

    # 取得 Basic Pitch 模型路徑（優先使用 ONNX 格式，相容性最好）
    from basic_pitch import ICASSP_2022_MODEL_PATH
    model_path = ICASSP_2022_MODEL_PATH
    onnx_path = Path(model_path).with_suffix(".onnx")
    if onnx_path.exists():
        model_path = str(onnx_path)

    # Basic Pitch 推論
    model_output, midi_data, note_events = predict(
        audio_path=str(audio_path),
        model_or_model_path=model_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

    # 儲存 MIDI 檔案
    midi_path = output_dir / f"{filename}.mid"
    midi_data.write(str(midi_path))

    # 解析音符資料
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append({
                "start": round(note.start, 4),
                "end": round(note.end, 4),
                "duration": round(note.end - note.start, 4),
                "pitch": note.pitch,
                "pitch_name": pretty_midi.note_number_to_name(note.pitch),
                "velocity": note.velocity,
            })

    # 按開始時間排序
    notes.sort(key=lambda n: (n["start"], n["pitch"]))

    # 過濾出吉他音域內的音符
    guitar_notes = [
        n for n in notes
        if GUITAR_MIDI_LOW <= n["pitch"] <= GUITAR_MIDI_HIGH
    ]

    filtered_count = len(notes) - len(guitar_notes)
    if filtered_count > 0:
        print(f"  已過濾 {filtered_count} 個超出吉他音域的音符")

    print(f"音高偵測完成: 偵測到 {len(guitar_notes)} 個音符")
    print(f"MIDI 已儲存至: {midi_path}")

    return {
        "midi_path": midi_path,
        "notes": guitar_notes,
        "all_notes": notes,
        "midi_data": midi_data,
    }


def quantize_notes(
    notes: list[dict],
    bpm: float = 120.0,
    quantize_to: int = 16,
) -> list[dict]:
    """
    將音符時間量化到最近的節拍格線上。

    Args:
        notes: 音符列表
        bpm: 每分鐘拍數
        quantize_to: 量化精度（16 = 十六分音符）

    Returns:
        量化後的音符列表
    """
    beat_duration = 60.0 / bpm
    grid_duration = beat_duration * (4.0 / quantize_to)

    quantized = []
    for note in notes:
        q_note = note.copy()
        q_note["start"] = round(note["start"] / grid_duration) * grid_duration
        q_note["end"] = round(note["end"] / grid_duration) * grid_duration
        # 確保音符至少有一個格線的長度
        if q_note["end"] <= q_note["start"]:
            q_note["end"] = q_note["start"] + grid_duration
        q_note["duration"] = q_note["end"] - q_note["start"]
        quantized.append(q_note)

    return quantized
