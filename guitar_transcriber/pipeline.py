"""
Pipeline: 將所有模組串接成完整的轉譜流程
"""

from pathlib import Path
from typing import Optional
import json
import time

from .audio_extractor import extract_audio, get_video_info
from .source_separator import separate_sources, get_guitar_track
from .pitch_detector import audio_to_midi, quantize_notes
from .chord_recognizer import (
    recognize_chords_from_audio,
    recognize_chords_from_notes,
    format_chord_progression,
)
from .tab_generator import generate_tab_from_notes


class TranscriptionResult:
    """轉譜結果的容器"""

    def __init__(self):
        self.video_info: dict = {}
        self.audio_path: Optional[Path] = None
        self.guitar_track_path: Optional[Path] = None
        self.midi_path: Optional[Path] = None
        self.notes: list[dict] = []
        self.chords: list[dict] = []
        self.tab: str = ""
        self.chord_progression: str = ""
        self.steps_completed: list[str] = []
        self.errors: list[str] = []
        self.elapsed_time: float = 0.0

    def summary(self) -> str:
        """輸出結果摘要"""
        lines = [
            "=" * 60,
            "  吉他轉譜結果",
            "=" * 60,
        ]

        if self.video_info:
            lines.append(f"  影片: {self.video_info.get('title', 'N/A')}")
            lines.append(f"  上傳者: {self.video_info.get('uploader', 'N/A')}")
            dur = self.video_info.get("duration", 0)
            lines.append(f"  時長: {dur // 60}:{dur % 60:02d}")

        lines.append(f"  偵測音符數: {len(self.notes)}")
        lines.append(f"  和弦段落數: {len(self.chords)}")
        lines.append(f"  處理時間: {self.elapsed_time:.1f} 秒")
        lines.append("")

        if self.steps_completed:
            lines.append("  完成步驟:")
            for step in self.steps_completed:
                lines.append(f"    ✓ {step}")
            lines.append("")

        if self.errors:
            lines.append("  錯誤/警告:")
            for err in self.errors:
                lines.append(f"    ✗ {err}")
            lines.append("")

        if self.chord_progression:
            lines.append(self.chord_progression)
            lines.append("")

        if self.tab:
            lines.append(self.tab)

        lines.append("=" * 60)
        return "\n".join(lines)


def transcribe(
    url: str,
    output_dir: str | Path = "./output",
    tuning: str = "standard",
    skip_separation: bool = False,
    demucs_model: str = "htdemucs",
    device: str = "auto",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    chord_method: str = "both",
    chars_per_second: float = 8.0,
    line_width: int = 80,
    save_intermediate: bool = True,
) -> TranscriptionResult:
    """
    完整的轉譜 pipeline。

    Args:
        url: YouTube 影片網址
        output_dir: 輸出目錄
        tuning: 吉他調弦
        skip_separation: 是否跳過音源分離（直接用混合音訊）
        demucs_model: Demucs 模型名稱
        device: 運算裝置
        onset_threshold: 音符起始偵測閾值
        frame_threshold: 音框偵測閾值
        chord_method: 和弦辨識方式 ("audio", "midi", "both")
        chars_per_second: Tab 譜面密度
        line_width: Tab 行寬
        save_intermediate: 是否儲存中間檔案

    Returns:
        TranscriptionResult 物件
    """
    result = TranscriptionResult()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # ─── Step 1: 取得影片資訊 ───
    print("\n" + "=" * 50)
    print("  Step 1/5: 取得影片資訊")
    print("=" * 50)
    try:
        result.video_info = get_video_info(url)
        title = result.video_info.get("title", "Unknown")
        print(f"  標題: {title}")
        result.steps_completed.append("取得影片資訊")
    except Exception as e:
        result.errors.append(f"取得影片資訊失敗: {e}")
        title = "Unknown"

    # ─── Step 2: 擷取音訊 ───
    print("\n" + "=" * 50)
    print("  Step 2/5: 擷取音訊")
    print("=" * 50)
    try:
        result.audio_path = extract_audio(
            url=url,
            output_dir=output_dir / "audio",
            filename="source",
        )
        result.steps_completed.append("YouTube 音訊擷取")
    except Exception as e:
        result.errors.append(f"音訊擷取失敗: {e}")
        result.elapsed_time = time.time() - start_time
        return result

    # ─── Step 3: 音源分離 ───
    print("\n" + "=" * 50)
    print("  Step 3/5: 音源分離")
    print("=" * 50)

    if skip_separation:
        print("  跳過音源分離，直接使用混合音訊")
        result.guitar_track_path = result.audio_path
        result.steps_completed.append("音源分離（已跳過）")
    else:
        try:
            stems = separate_sources(
                audio_path=result.audio_path,
                output_dir=output_dir / "separated",
                model=demucs_model,
                device=device,
            )
            result.guitar_track_path = get_guitar_track(stems)
            result.steps_completed.append(
                f"音源分離（{demucs_model}）"
            )
        except Exception as e:
            result.errors.append(
                f"音源分離失敗，改用混合音訊: {e}"
            )
            result.guitar_track_path = result.audio_path

    # ─── Step 4: 音高偵測 ───
    print("\n" + "=" * 50)
    print("  Step 4/5: 音高偵測 → MIDI")
    print("=" * 50)
    try:
        midi_result = audio_to_midi(
            audio_path=result.guitar_track_path,
            output_dir=output_dir / "midi",
            filename="guitar",
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
        )
        result.midi_path = midi_result["midi_path"]
        result.notes = midi_result["notes"]
        result.steps_completed.append(
            f"音高偵測（{len(result.notes)} 個音符）"
        )
    except Exception as e:
        result.errors.append(f"音高偵測失敗: {e}")
        result.elapsed_time = time.time() - start_time
        return result

    # ─── Step 5: 和弦辨識 + Tab 生成 ───
    print("\n" + "=" * 50)
    print("  Step 5/5: 和弦辨識 & Tab 生成")
    print("=" * 50)

    # 和弦辨識
    try:
        if chord_method in ("audio", "both"):
            audio_for_chords = (
                result.guitar_track_path or result.audio_path
            )
            result.chords = recognize_chords_from_audio(audio_for_chords)

        if chord_method in ("midi", "both") and result.notes:
            midi_chords = recognize_chords_from_notes(result.notes)
            if chord_method == "midi":
                result.chords = midi_chords
            elif chord_method == "both" and not result.chords:
                result.chords = midi_chords

        result.chord_progression = format_chord_progression(result.chords)
        result.steps_completed.append("和弦辨識")
    except Exception as e:
        result.errors.append(f"和弦辨識失敗: {e}")

    # Tab 生成
    try:
        result.tab = generate_tab_from_notes(
            notes=result.notes,
            tuning=tuning,
            chords=result.chords,
            title=title,
            chars_per_second=chars_per_second,
            line_width=line_width,
        )
        result.steps_completed.append("六線譜生成")
    except Exception as e:
        result.errors.append(f"Tab 生成失敗: {e}")

    # ─── 儲存結果 ───
    result.elapsed_time = time.time() - start_time

    if save_intermediate:
        _save_results(result, output_dir)

    return result


def _save_results(result: TranscriptionResult, output_dir: Path):
    """儲存所有結果到檔案"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 儲存完整摘要
    summary_path = output_dir / "transcription.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(result.summary())
    print(f"\n完整結果已儲存至: {summary_path}")

    # 儲存和弦進行（JSON）
    if result.chords:
        chords_path = output_dir / "chords.json"
        with open(chords_path, "w", encoding="utf-8") as f:
            json.dump(result.chords, f, ensure_ascii=False, indent=2)
        print(f"和弦資料已儲存至: {chords_path}")

    # 儲存音符資料（JSON）
    if result.notes:
        notes_path = output_dir / "notes.json"
        # 去除不可序列化的欄位
        serializable_notes = []
        for n in result.notes:
            sn = {k: v for k, v in n.items() if k != "midi_data"}
            serializable_notes.append(sn)
        with open(notes_path, "w", encoding="utf-8") as f:
            json.dump(serializable_notes, f, ensure_ascii=False, indent=2)
        print(f"音符資料已儲存至: {notes_path}")

    # 儲存 Tab
    if result.tab:
        tab_path = output_dir / "guitar_tab.txt"
        with open(tab_path, "w", encoding="utf-8") as f:
            f.write(result.tab)
        print(f"六線譜已儲存至: {tab_path}")
