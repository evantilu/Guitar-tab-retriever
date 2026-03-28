"""
模組 4: 和弦辨識
從音訊中辨識和弦進行，支援兩種模式：
1. 基於 librosa 的 chroma 特徵分析（改良版：長窗口 + smoothing + 合併）
2. 基於 MIDI 音符的和弦推斷
"""

from pathlib import Path
from typing import Optional
import numpy as np


# 12 個音名
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 精簡和弦模板：只保留最常見的類型，避免過多選擇造成混淆
CHORD_TEMPLATES = {
    "maj":     [0, 4, 7],
    "m":       [0, 3, 7],
    "7":       [0, 4, 7, 10],
    "maj7":    [0, 4, 7, 11],
    "m7":      [0, 3, 7, 10],
    "dim":     [0, 3, 6],
    "aug":     [0, 4, 8],
    "sus2":    [0, 2, 7],
    "sus4":    [0, 5, 7],
    "add9":    [0, 4, 7, 14],
    "m9":      [0, 3, 7, 10, 14],
    "9":       [0, 4, 7, 10, 14],
    "dim7":    [0, 3, 6, 9],
    "5":       [0, 7],
}

# 和弦顯示名稱
CHORD_DISPLAY = {
    "maj": "", "m": "m", "7": "7", "maj7": "maj7", "m7": "m7",
    "dim": "dim", "aug": "aug", "sus2": "sus2", "sus4": "sus4",
    "add9": "add9", "m9": "m9", "9": "9", "dim7": "dim7", "5": "5",
}


def _build_chroma_template(root: int, intervals: list[int]) -> np.ndarray:
    """建立 12 維的 chroma 模板向量"""
    template = np.zeros(12)
    for interval in intervals:
        template[(root + interval) % 12] = 1.0
    # 根音加重
    template[root % 12] *= 2.0
    # 五度音加重
    if 7 in intervals:
        template[(root + 7) % 12] *= 1.3
    return template / (np.linalg.norm(template) + 1e-8)


def recognize_chords_from_audio(
    audio_path: str | Path,
    hop_length: int = 4096,
    segment_duration: float = 2.0,
    min_confidence: float = 0.45,
    smooth_window: int = 3,
) -> list[dict]:
    """
    使用 librosa chroma 特徵從音訊辨識和弦。
    改良版：更長的分析窗口 + chroma smoothing + 合併連續相同和弦。

    Args:
        audio_path: 音訊路徑
        hop_length: STFT hop length（更大 = 更粗的時間解析度）
        segment_duration: 每段分析長度（秒），2.0 秒適合指彈
        min_confidence: 最低信心度
        smooth_window: chroma 平滑窗口大小

    Returns:
        和弦列表 [{start, end, duration, chord, confidence}, ...]
    """
    try:
        import librosa
    except ImportError:
        raise RuntimeError("librosa 未安裝")

    audio_path = Path(audio_path)
    print("正在進行和弦辨識（chroma 分析）...")

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    # CQT chroma — 對吉他泛音列更準確
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=hop_length, n_chroma=12
    )

    # Smoothing: 中位數濾波減少短暫波動
    if smooth_window > 1:
        from scipy.ndimage import median_filter
        chroma = median_filter(chroma, size=(1, smooth_window))

    times = librosa.frames_to_time(
        np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length
    )

    # 按 segment_duration 分段
    frames_per_segment = max(1, int(segment_duration * sr / hop_length))
    n_segments = max(1, chroma.shape[1] // frames_per_segment)

    raw_chords = []

    for i in range(n_segments):
        start_frame = i * frames_per_segment
        end_frame = min((i + 1) * frames_per_segment, chroma.shape[1])

        # 段落平均 chroma
        seg_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)

        # 靜音檢查
        if np.max(seg_chroma) < 0.03:
            continue

        seg_chroma = seg_chroma / (np.linalg.norm(seg_chroma) + 1e-8)

        # 比對和弦模板
        best_chord = "N.C."
        best_score = -1

        for root in range(12):
            for chord_type, intervals in CHORD_TEMPLATES.items():
                template = _build_chroma_template(root, intervals)
                score = float(np.dot(seg_chroma, template))

                if score > best_score:
                    best_score = score
                    root_name = PITCH_CLASSES[root]
                    suffix = CHORD_DISPLAY[chord_type]
                    best_chord = f"{root_name}{suffix}"

        if best_score < min_confidence:
            continue

        start_time = float(times[start_frame])
        end_time = float(times[min(end_frame - 1, len(times) - 1)])

        raw_chords.append({
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "duration": round(end_time - start_time, 3),
            "chord": best_chord,
            "confidence": round(best_score, 3),
        })

    # 合併連續相同根音的和弦
    chords = _merge_similar_chords(raw_chords)

    print(f"和弦辨識完成: 偵測到 {len(chords)} 個和弦段落")
    return chords


def _merge_similar_chords(chords: list[dict], min_duration: float = 1.0) -> list[dict]:
    """
    合併連續的相同/相似和弦，並過濾過短的段落。
    """
    if not chords:
        return []

    merged = [chords[0].copy()]

    for c in chords[1:]:
        prev = merged[-1]
        prev_root = _get_root(prev["chord"])
        curr_root = _get_root(c["chord"])

        # 如果根音相同，合併
        if prev_root == curr_root:
            # 保留信心度更高的和弦名稱
            if c["confidence"] > prev["confidence"]:
                prev["chord"] = c["chord"]
                prev["confidence"] = c["confidence"]
            prev["end"] = c["end"]
            prev["duration"] = round(prev["end"] - prev["start"], 3)
        else:
            merged.append(c.copy())

    # 過濾過短的和弦（< min_duration），吸收到前一個或後一個
    filtered = []
    for c in merged:
        if c["duration"] >= min_duration:
            filtered.append(c)
        elif filtered:
            # 過短的，併入前一個
            filtered[-1]["end"] = c["end"]
            filtered[-1]["duration"] = round(filtered[-1]["end"] - filtered[-1]["start"], 3)

    return filtered


def _get_root(chord_name: str) -> str:
    """提取和弦根音"""
    if not chord_name or chord_name == "N.C.":
        return ""
    r = chord_name[0]
    if len(chord_name) >= 2 and chord_name[1] in "#b":
        r = chord_name[:2]
    return r


def recognize_chords_from_notes(
    notes: list[dict],
    segment_duration: float = 2.0,
) -> list[dict]:
    """
    從 MIDI 音符資料推斷和弦（改良版：更長窗口）。
    """
    if not notes:
        return []

    print("正在從 MIDI 音符推斷和弦...")

    max_time = max(n["end"] for n in notes)
    n_segments = int(np.ceil(max_time / segment_duration))

    raw_chords = []
    prev_chord = None

    for i in range(n_segments):
        t_start = i * segment_duration
        t_end = (i + 1) * segment_duration

        # 收集這個時間段內的所有 pitch class，用持續時間加權
        chroma = np.zeros(12)
        for note in notes:
            if note["start"] < t_end and note["end"] > t_start:
                overlap = min(note["end"], t_end) - max(note["start"], t_start)
                chroma[note["pitch"] % 12] += overlap

        if np.sum(chroma) < 0.01:
            continue

        chroma = chroma / (np.linalg.norm(chroma) + 1e-8)

        best_chord = "N.C."
        best_score = -1

        for root in range(12):
            for chord_type, intervals in CHORD_TEMPLATES.items():
                template = _build_chroma_template(root, intervals)
                score = float(np.dot(chroma, template))
                if score > best_score:
                    best_score = score
                    root_name = PITCH_CLASSES[root]
                    suffix = CHORD_DISPLAY[chord_type]
                    best_chord = f"{root_name}{suffix}"

        if best_score < 0.4:
            continue

        chord_info = {
            "start": round(t_start, 3),
            "end": round(t_end, 3),
            "duration": round(t_end - t_start, 3),
            "chord": best_chord,
            "confidence": round(best_score, 3),
        }

        # 合併連續相同
        if prev_chord and _get_root(prev_chord["chord"]) == _get_root(best_chord):
            if best_score > prev_chord["confidence"]:
                prev_chord["chord"] = best_chord
                prev_chord["confidence"] = best_score
            prev_chord["end"] = round(t_end, 3)
            prev_chord["duration"] = round(prev_chord["end"] - prev_chord["start"], 3)
        else:
            raw_chords.append(chord_info)
            prev_chord = chord_info

    print(f"和弦推斷完成: {len(raw_chords)} 個和弦段落")
    return raw_chords


def format_chord_progression(chords: list[dict], beats_per_bar: int = 4) -> str:
    """將和弦列表格式化為易讀的和弦進行。"""
    if not chords:
        return "（無和弦資料）"

    lines = ["=== 和弦進行 ===\n"]

    for chord in chords:
        duration = chord.get("duration", chord["end"] - chord["start"])
        conf_str = f"{chord['confidence']:.0%}" if chord.get("confidence", 0) > 0 else ""
        time_str = _format_time(chord["start"])
        lines.append(
            f"  {time_str}  {chord['chord']:<10s}  ({duration:.1f}s)  {conf_str}"
        )

    unique_sequence = []
    for chord in chords:
        name = chord.get("chord", "")
        if name and name != "N.C." and (not unique_sequence or unique_sequence[-1] != name):
            unique_sequence.append(name)

    lines.append(f"\n簡化進行: {' → '.join(unique_sequence)}")
    return "\n".join(lines)


def _format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"
