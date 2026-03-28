"""
模組 3.5: 泛音偵測
分析音訊頻譜特徵，辨識自然泛音 (natural harmonics) 並修正其指板位置。

原理：
  自然泛音的頻譜接近純正弦波（極少頻譜峰值），
  而一般按弦音有豐富的泛音列（很多頻譜峰值）。
  利用 spectral flatness 和頻譜峰值數量來區分。
"""

import numpy as np
from pathlib import Path
from typing import Optional


# 自然泛音：sounding pitch 相對於空弦的 MIDI 半音差 → 對應的泛音格位
# (semitones_above_open, harmonic_fret_positions)
HARMONIC_INTERVALS = [
    (12, [12]),       # 2nd harmonic: octave
    (19, [7, 19]),    # 3rd harmonic: octave + P5
    (24, [5]),        # 4th harmonic: 2 octaves
    (28, [4, 9]),     # 5th harmonic: 2 octaves + M3
    (31, [3]),        # 6th harmonic: 2 octaves + P5
]

# 標準吉他調音
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]


def detect_harmonics(
    audio_path: str | Path,
    notes: list[dict],
    tuning: list[int] | None = None,
    sr: int = 44100,
    flatness_threshold: float = 0.008,
    max_peaks: int = 3,
    min_confidence: float = 0.82,
    min_duration: float = 0.15,
) -> list[dict]:
    """
    分析每個偵測到的音符，判斷是否為自然泛音。

    偵測邏輯：
      1. 取出每個音符對應時間段的音訊片段
      2. 計算 spectral flatness（越低越像純音/泛音）
      3. 計算頻譜中的顯著峰值數量（泛音通常 < 4 個）
      4. 確認該音高可以由某條弦的自然泛音產生

    Args:
        audio_path: 音訊檔案路徑
        notes: 來自 pitch_detector 的音符列表
        tuning: 吉他調弦 MIDI 列表，預設標準調弦
        sr: 取樣率
        flatness_threshold: spectral flatness 閾值，低於此值視為泛音候選
        max_peaks: 頻譜峰值數上限，低於此值視為泛音候選
        min_confidence: 泛音判定的最低信心分數

    Returns:
        更新後的音符列表，泛音音符會新增：
        - is_harmonic: True
        - harmonic_fret: 泛音格位（如 12, 7, 5）
        - harmonic_string: 泛音所在弦（0-5）
        - harmonic_confidence: 信心分數
    """
    import librosa
    from scipy.signal import find_peaks

    if tuning is None:
        tuning = STANDARD_TUNING

    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"  警告: 音訊檔案不存在，跳過泛音偵測: {audio_path}")
        return notes

    # 載入音訊
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    duration = len(y) / sr

    print("正在進行泛音偵測...")
    harmonic_count = 0

    updated_notes = []
    for note in notes:
        note_copy = note.copy()
        note_copy["is_harmonic"] = False

        # 短音符不太可能是泛音（泛音有較長的 bell-like decay）
        note_dur = note.get("duration", note["end"] - note["start"])
        if note_dur < min_duration:
            updated_notes.append(note_copy)
            continue

        # 取出音符時間段的音訊片段
        start_sample = int(note["start"] * sr)
        end_sample = int(note["end"] * sr)

        # 確保片段有效
        if start_sample >= len(y) or end_sample <= start_sample:
            updated_notes.append(note_copy)
            continue

        # 取音符前段（onset 後的穩定段較有代表性）
        note_duration_samples = end_sample - start_sample
        # 使用音符開始後 10%-60% 的區段（跳過 attack，避免 release）
        analysis_start = start_sample + int(note_duration_samples * 0.1)
        analysis_end = start_sample + int(note_duration_samples * 0.6)
        analysis_end = min(analysis_end, len(y))

        if analysis_end - analysis_start < 512:
            updated_notes.append(note_copy)
            continue

        segment = y[analysis_start:analysis_end]

        # --- 特徵 1: Spectral Flatness ---
        flatness = librosa.feature.spectral_flatness(y=segment, n_fft=2048)
        mean_flatness = float(np.mean(flatness))

        # --- 特徵 2: 頻譜峰值數量 ---
        fft_mag = np.abs(np.fft.rfft(segment, n=4096))
        fft_mag_db = 20 * np.log10(fft_mag + 1e-10)
        # 只看有意義的峰值（高於最大值 - 20dB）
        peak_threshold = np.max(fft_mag_db) - 20.0
        peaks, _ = find_peaks(fft_mag_db, height=peak_threshold, distance=10)
        num_peaks = len(peaks)

        # --- 判斷是否為泛音候選 ---
        is_candidate = (mean_flatness < flatness_threshold) and (num_peaks <= max_peaks)

        if not is_candidate:
            updated_notes.append(note_copy)
            continue

        # --- 確認音高是否匹配某條弦的自然泛音 ---
        midi_pitch = note["pitch"]
        best_match = _find_harmonic_match(midi_pitch, tuning)

        if best_match is not None:
            string_idx, harm_fret, semitones = best_match

            # 計算信心分數
            # flatness 越低、peaks 越少 → 越有信心
            flatness_score = max(0, 1.0 - mean_flatness / flatness_threshold)
            peaks_score = max(0, 1.0 - num_peaks / (max_peaks + 1))
            confidence = (flatness_score * 0.6 + peaks_score * 0.4)

            if confidence >= min_confidence:
                note_copy["is_harmonic"] = True
                note_copy["harmonic_fret"] = harm_fret
                note_copy["harmonic_string"] = string_idx
                note_copy["harmonic_confidence"] = round(confidence, 3)
                harmonic_count += 1

        updated_notes.append(note_copy)

    print(f"泛音偵測完成: 偵測到 {harmonic_count} 個自然泛音")
    return updated_notes


def _find_harmonic_match(
    midi_pitch: int,
    tuning: list[int],
) -> Optional[tuple[int, int, int]]:
    """
    檢查一個 MIDI 音高是否可以由某條弦的自然泛音產生。

    Args:
        midi_pitch: 偵測到的 MIDI 音高
        tuning: 吉他調弦

    Returns:
        (string_index, harmonic_fret, semitones_above_open) 或 None
        若有多個匹配，優先選擇低弦（厚弦泛音較常用）
    """
    matches = []

    for string_idx, open_pitch in enumerate(tuning):
        interval = midi_pitch - open_pitch
        for semitones, fret_positions in HARMONIC_INTERVALS:
            if interval == semitones:
                # 優先選最常用的格位（12 > 7 > 5 > ...）
                preferred_fret = fret_positions[0]
                matches.append((string_idx, preferred_fret, semitones))

    if not matches:
        return None

    # 優先低弦（string_idx 小的 = 低音弦），更常用於泛音
    # 但也考慮常見泛音格位的優先度
    FRET_PRIORITY = {12: 0, 7: 1, 5: 2, 19: 3, 4: 4, 9: 5, 3: 6}
    matches.sort(key=lambda m: (FRET_PRIORITY.get(m[1], 99), m[0]))

    return matches[0]
