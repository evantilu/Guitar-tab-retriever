"""
模組 4: 和弦辨識
從音訊中辨識和弦進行，支援兩種模式：
1. 基於 librosa 的 chroma 特徵分析
2. 基於 MIDI 音符的和弦推斷
"""

from pathlib import Path
from typing import Optional
import numpy as np


# 12 個音名
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 和弦模板：每個和弦對應一組相對於根音的半音程
CHORD_TEMPLATES = {
    "maj":     [0, 4, 7],
    "min":     [0, 3, 7],
    "7":       [0, 4, 7, 10],
    "maj7":    [0, 4, 7, 11],
    "min7":    [0, 3, 7, 10],
    "dim":     [0, 3, 6],
    "aug":     [0, 4, 8],
    "sus2":    [0, 2, 7],
    "sus4":    [0, 5, 7],
    "add9":    [0, 4, 7, 14],
    "min9":    [0, 3, 7, 10, 14],
    "9":       [0, 4, 7, 10, 14],
    "6":       [0, 4, 7, 9],
    "min6":    [0, 3, 7, 9],
    "dim7":    [0, 3, 6, 9],
    "m7b5":    [0, 3, 6, 10],
    "power":   [0, 7],
}

# 和弦顯示名稱對映
CHORD_DISPLAY = {
    "maj": "",
    "min": "m",
    "7": "7",
    "maj7": "maj7",
    "min7": "m7",
    "dim": "dim",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "add9": "add9",
    "min9": "m9",
    "9": "9",
    "6": "6",
    "min6": "m6",
    "dim7": "dim7",
    "m7b5": "m7b5",
    "power": "5",
}


def _build_chroma_template(root: int, intervals: list[int]) -> np.ndarray:
    """建立 12 維的 chroma 模板向量"""
    template = np.zeros(12)
    for interval in intervals:
        template[(root + interval) % 12] = 1.0
    # 給根音更高的權重
    template[root % 12] *= 1.5
    return template / np.linalg.norm(template)


def recognize_chords_from_audio(
    audio_path: str | Path,
    hop_length: int = 2048,
    segment_duration: float = 0.5,
    min_confidence: float = 0.5,
) -> list[dict]:
    """
    使用 librosa chroma 特徵從音訊辨識和弦。

    Args:
        audio_path: 音訊檔案路徑
        hop_length: STFT hop length
        segment_duration: 每個分析段落的長度（秒）
        min_confidence: 最低信心度閾值

    Returns:
        和弦列表 [{start, end, chord, confidence}, ...]
    """
    try:
        import librosa
    except ImportError:
        raise RuntimeError("librosa 未安裝。請執行: pip install librosa")

    audio_path = Path(audio_path)
    print("正在進行和弦辨識（chroma 分析）...")

    # 載入音訊
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    # 計算 chroma 特徵 (CQT-based，對吉他音色更準確)
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=hop_length, n_chroma=12
    )

    # 計算每個 frame 對應的時間
    times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
    )

    # 將 chroma 分段，每段約 segment_duration 秒
    frames_per_segment = max(1, int(segment_duration * sr / hop_length))
    n_segments = chroma.shape[1] // frames_per_segment

    chords = []
    prev_chord = None

    for i in range(n_segments):
        start_frame = i * frames_per_segment
        end_frame = min((i + 1) * frames_per_segment, chroma.shape[1])

        # 取這個段落的平均 chroma
        segment_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)

        # 如果能量太低，標記為靜音
        if np.max(segment_chroma) < 0.05:
            if prev_chord and prev_chord.get("chord") == "N.C.":
                prev_chord["end"] = times[end_frame - 1] if end_frame <= len(times) else times[-1]
            else:
                chord_info = {
                    "start": times[start_frame],
                    "end": times[end_frame - 1] if end_frame <= len(times) else times[-1],
                    "chord": "N.C.",
                    "confidence": 1.0,
                }
                chords.append(chord_info)
                prev_chord = chord_info
            continue

        # 正規化
        segment_chroma = segment_chroma / (np.linalg.norm(segment_chroma) + 1e-8)

        # 跟所有和弦模板做比對
        best_chord = None
        best_score = -1

        for root in range(12):
            for chord_type, intervals in CHORD_TEMPLATES.items():
                template = _build_chroma_template(root, intervals)
                score = np.dot(segment_chroma, template)

                if score > best_score:
                    best_score = score
                    root_name = PITCH_CLASSES[root]
                    suffix = CHORD_DISPLAY[chord_type]
                    best_chord = f"{root_name}{suffix}"

        if best_score < min_confidence:
            best_chord = "N.C."

        start_time = times[start_frame]
        end_time = times[end_frame - 1] if end_frame <= len(times) else times[-1]

        # 合併連續相同的和弦
        if prev_chord and prev_chord["chord"] == best_chord:
            prev_chord["end"] = end_time
        else:
            chord_info = {
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "chord": best_chord,
                "confidence": round(float(best_score), 3),
            }
            chords.append(chord_info)
            prev_chord = chord_info

    print(f"和弦辨識完成: 偵測到 {len(chords)} 個和弦段落")
    return chords


def recognize_chords_from_notes(
    notes: list[dict],
    segment_duration: float = 0.5,
) -> list[dict]:
    """
    從 MIDI 音符資料推斷和弦。

    這是一個替代方案：當你已經有了音符資料（來自 pitch_detector），
    可以直接從音符組合推斷和弦，不需要重新分析音訊。

    Args:
        notes: 來自 pitch_detector 的音符列表
        segment_duration: 分段長度（秒）

    Returns:
        和弦列表
    """
    if not notes:
        return []

    print("正在從 MIDI 音符推斷和弦...")

    # 找出時間範圍
    max_time = max(n["end"] for n in notes)
    n_segments = int(np.ceil(max_time / segment_duration))

    chords = []
    prev_chord = None

    for i in range(n_segments):
        t_start = i * segment_duration
        t_end = (i + 1) * segment_duration

        # 找出這個時間段內活躍的音符
        active_pitches = set()
        for note in notes:
            if note["start"] < t_end and note["end"] > t_start:
                active_pitches.add(note["pitch"] % 12)

        if len(active_pitches) < 2:
            if active_pitches:
                # 只有一個音，標記為單音
                root_name = PITCH_CLASSES[list(active_pitches)[0]]
                chord_name = f"{root_name}(note)"
            else:
                chord_name = "N.C."
            confidence = 0.3
        else:
            # 用 pitch class 建構 chroma 向量
            chroma = np.zeros(12)
            for pc in active_pitches:
                chroma[pc] = 1.0
            chroma = chroma / (np.linalg.norm(chroma) + 1e-8)

            best_chord = "N.C."
            best_score = -1

            for root in range(12):
                for chord_type, intervals in CHORD_TEMPLATES.items():
                    template = _build_chroma_template(root, intervals)
                    score = np.dot(chroma, template)
                    if score > best_score:
                        best_score = score
                        root_name = PITCH_CLASSES[root]
                        suffix = CHORD_DISPLAY[chord_type]
                        best_chord = f"{root_name}{suffix}"

            chord_name = best_chord
            confidence = float(best_score)

        # 合併連續相同和弦
        if prev_chord and prev_chord["chord"] == chord_name:
            prev_chord["end"] = round(t_end, 3)
        else:
            chord_info = {
                "start": round(t_start, 3),
                "end": round(t_end, 3),
                "chord": chord_name,
                "confidence": round(confidence, 3),
            }
            chords.append(chord_info)
            prev_chord = chord_info

    print(f"和弦推斷完成: {len(chords)} 個和弦段落")
    return chords


def format_chord_progression(chords: list[dict], beats_per_bar: int = 4) -> str:
    """
    將和弦列表格式化為易讀的和弦進行表示。

    Args:
        chords: 和弦列表
        beats_per_bar: 每小節拍數

    Returns:
        格式化的和弦進行字串
    """
    if not chords:
        return "（無和弦資料）"

    lines = []
    lines.append("=== 和弦進行 ===\n")

    for i, chord in enumerate(chords):
        duration = chord["end"] - chord["start"]
        conf_str = f"{chord['confidence']:.0%}" if chord["confidence"] > 0 else ""
        time_str = _format_time(chord["start"])
        lines.append(
            f"  {time_str}  {chord['chord']:<10s}  "
            f"({duration:.1f}s)  {conf_str}"
        )

    # 簡化摘要：只列出不重複的和弦順序
    unique_sequence = []
    for chord in chords:
        if chord["chord"] != "N.C." and (
            not unique_sequence or unique_sequence[-1] != chord["chord"]
        ):
            unique_sequence.append(chord["chord"])

    lines.append(f"\n簡化進行: {' → '.join(unique_sequence)}")
    return "\n".join(lines)


def _format_time(seconds: float) -> str:
    """將秒數格式化為 MM:SS.ms"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"
