"""
模組 3B: SynthTab 吉他轉譜
使用 SynthTab 預訓練模型直接從音訊輸出 string+fret tablature。
比 Basic Pitch + 啟發式弦分配更準確。
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np


# SynthTab repo 路徑
_SYNTHTAB_DIR = Path(__file__).parent.parent / "SynthTab" / "demo_embedding"
_MODEL_PATH = _SYNTHTAB_DIR / "pretrained_models" / "finetuned" / "GuitarSet.pt"

# 標準吉他調音 MIDI
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
STRING_NAMES = ["E", "A", "D", "G", "B", "e"]


def check_synthtab_available() -> bool:
    """檢查 SynthTab 模型是否可用"""
    return _MODEL_PATH.exists()


def synthtab_transcribe(
    audio_path: str | Path,
    tuning: list[int] | None = None,
    device_str: str = "cpu",
) -> dict:
    """
    使用 SynthTab 從音訊直接轉譜為 tablature。

    Args:
        audio_path: WAV 音訊路徑
        tuning: 吉他調弦 MIDI (用於計算 pitch)
        device_str: 運算裝置

    Returns:
        dict with:
        - notes: [{string, fret, start, end, pitch, pitch_name, duration}, ...]
        - tablature_raw: 原始 tablature 矩陣 (6, T)
    """
    import torch
    import pretty_midi

    # 加入 SynthTab 到 path
    if str(_SYNTHTAB_DIR) not in sys.path:
        sys.path.insert(0, str(_SYNTHTAB_DIR))

    import amt_tools.tools as tools
    from amt_tools.features import CQT
    from amt_tools.transcribe import ComboEstimator, TablatureWrapper, StackedMultiPitchCollapser
    from amt_tools.evaluate import run_offline

    if tuning is None:
        tuning = STANDARD_TUNING

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"音訊檔案不存在: {audio_path}")

    if not _MODEL_PATH.exists():
        raise RuntimeError(f"SynthTab 模型不存在: {_MODEL_PATH}")

    print("正在進行吉他轉譜（SynthTab）...")

    # 載入模型
    device = torch.device(device_str)
    model = torch.load(str(_MODEL_PATH), map_location=device, weights_only=False)
    model.change_device(device)
    model.eval()

    # 設定特徵提取
    sample_rate = 22050
    hop_length = 512

    data_proc = CQT(sample_rate=sample_rate, hop_length=hop_length,
                    n_bins=192, bins_per_octave=24)

    estimator = ComboEstimator([
        TablatureWrapper(profile=model.profile),
        StackedMultiPitchCollapser(profile=model.profile),
    ])

    # 載入音訊
    audio, sr = tools.load_normalize_audio(str(audio_path), sample_rate)

    # 提取特徵
    features = {
        tools.KEY_FEATS: data_proc.process_audio(audio),
        tools.KEY_TIMES: data_proc.get_times(audio),
    }

    # 推論
    predictions = run_offline(features, model, estimator)

    tab = predictions[tools.KEY_TABLATURE]  # (6, T)
    times = predictions[tools.KEY_TIMES]    # (T,)

    # 轉換為音符列表
    notes = _tablature_to_notes(tab, times, tuning)

    print(f"轉譜完成: 偵測到 {len(notes)} 個音符")

    return {
        "notes": notes,
        "tablature_raw": tab,
        "times": times,
    }


def _tablature_to_notes(
    tab: np.ndarray,
    times: np.ndarray,
    tuning: list[int],
) -> list[dict]:
    """
    將 SynthTab 的 tablature 矩陣轉換為音符列表。

    tab: (6, T) — 每個值: -1=inactive, 0=open, 1=fret1, ...
    times: (T,) — 每幀的時間戳（秒）
    """
    import pretty_midi

    notes = []

    for s in range(6):
        prev_fret = -1
        note_start_idx = None

        for t in range(tab.shape[1]):
            fret = int(tab[s, t])

            if fret >= 0 and prev_fret < 0:
                # 音符開始
                note_start_idx = t
            elif fret < 0 and prev_fret >= 0 and note_start_idx is not None:
                # 音符結束
                _add_note(notes, s, prev_fret, note_start_idx, t, times, tuning)
                note_start_idx = None
            elif fret >= 0 and fret != prev_fret and note_start_idx is not None:
                # 格位變化（同弦換音）
                _add_note(notes, s, prev_fret, note_start_idx, t, times, tuning)
                note_start_idx = t

            prev_fret = fret

        # 處理尾端未結束的音符
        if prev_fret >= 0 and note_start_idx is not None:
            _add_note(notes, s, prev_fret, note_start_idx, tab.shape[1] - 1, times, tuning)

    # 按時間排序
    notes.sort(key=lambda n: (n["start"], n["string"]))

    # 合併過短的音符（< 30ms 的雜訊）
    notes = [n for n in notes if n["duration"] >= 0.03]

    # 過濾孤立的超短音符（< 50ms 且前後 100ms 內同弦無其他音）
    notes = _filter_isolated_noise(notes)

    return notes


def _filter_isolated_noise(notes: list[dict], min_dur: float = 0.05, gap: float = 0.1) -> list[dict]:
    """過濾孤立的短雜訊音符"""
    if len(notes) < 3:
        return notes

    filtered = []
    for i, n in enumerate(notes):
        if n["duration"] >= min_dur:
            filtered.append(n)
            continue
        # 檢查同弦前後是否有相鄰音符
        has_neighbor = False
        for j in range(max(0, i - 3), min(len(notes), i + 4)):
            if j != i and notes[j]["string"] == n["string"]:
                if abs(notes[j]["start"] - n["start"]) < gap:
                    has_neighbor = True
                    break
        if has_neighbor:
            filtered.append(n)
    return filtered


def _add_note(notes, string, fret, start_idx, end_idx, times, tuning):
    """添加一個音符到列表"""
    import pretty_midi

    start_time = float(times[start_idx])
    end_time = float(times[min(end_idx, len(times) - 1)])
    duration = end_time - start_time

    if duration < 0.01:
        return

    # 計算 MIDI pitch
    open_string_pitch = tuning[string]
    midi_pitch = open_string_pitch + fret

    try:
        pitch_name = pretty_midi.note_number_to_name(midi_pitch)
    except Exception:
        pitch_name = "?"

    notes.append({
        "start": round(start_time, 4),
        "end": round(end_time, 4),
        "duration": round(duration, 4),
        "pitch": midi_pitch,
        "pitch_name": pitch_name,
        "string": string,
        "fret": fret,
        "velocity": 80,
        "is_harmonic": False,
    })
