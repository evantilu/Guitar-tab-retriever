"""
Pipeline: 將所有模組串接成完整的轉譜流程
"""

from pathlib import Path
from typing import Optional
import json
import re
import time
from urllib.parse import urlparse, parse_qs

from .audio_extractor import extract_audio, get_video_info
from .source_separator import separate_sources, get_guitar_track
from .pitch_detector import audio_to_midi, quantize_notes
from .harmonics_detector import detect_harmonics
from .synthtab_detector import check_synthtab_available, synthtab_transcribe
from .chord_recognizer import (
    recognize_chords_from_audio,
    recognize_chords_from_notes,
    format_chord_progression,
)
from .tab_generator import generate_tab_from_notes
from .pdf_generator import generate_pdf_tab


class TranscriptionResult:
    """轉譜結果的容器"""

    def __init__(self):
        self.video_info: dict = {}
        self.audio_path: Optional[Path] = None
        self.guitar_track_path: Optional[Path] = None
        self.midi_path: Optional[Path] = None
        self.pdf_path: Optional[Path] = None
        self.notes: list[dict] = []
        self.chords: list[dict] = []
        self.tab: str = ""
        self.chord_progression: str = ""
        self.steps_completed: list[str] = []
        self.errors: list[str] = []
        self.elapsed_time: float = 0.0
        self.capo: int = 0
        self.bpm: float = 120.0

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
            url = self.video_info.get("url", "")
            if url:
                lines.append(f"  來源: {url}")

        if self.capo > 0:
            lines.append(f"  Capo: {self.capo}")
        lines.append(f"  BPM: {self.bpm:.0f}")

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
    capo: int = 0,
    skip_separation: bool = True,
    demucs_model: str = "htdemucs",
    device: str = "auto",
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    chord_method: str = "both",
    chars_per_second: float = 8.0,
    line_width: int = 80,
    save_intermediate: bool = True,
    manual_bpm: float | None = None,
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
    result.capo = capo
    base_output_dir = Path(output_dir)
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

    # 建立以影片 ID + 標題命名的獨立資料夾
    video_id = _extract_video_id(url)
    folder_name = _make_folder_name(title, video_id)
    output_dir = base_output_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  輸出目錄: {output_dir}")

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

        # BPM 偵測
        if manual_bpm:
            result.bpm = manual_bpm
            print(f"  BPM（手動指定）: {result.bpm:.0f}")
        else:
            try:
                import librosa
                y_bpm, sr_bpm = librosa.load(str(result.audio_path), sr=22050, mono=True)
                tempo, _ = librosa.beat.beat_track(y=y_bpm, sr=sr_bpm)
                if hasattr(tempo, '__len__'):
                    tempo = tempo[0]
                if 40 <= tempo <= 220:
                    result.bpm = float(tempo)
                print(f"  BPM 偵測: {result.bpm:.0f}")
            except Exception:
                result.bpm = 120.0
                print(f"  BPM 偵測失敗，使用預設: {result.bpm:.0f}")

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

    # ─── Step 4: 吉他轉譜 ───
    print("\n" + "=" * 50)
    from .tab_generator import TUNINGS
    tuning_midi = TUNINGS.get(tuning, TUNINGS["standard"])

    use_synthtab = check_synthtab_available()

    if use_synthtab:
        print("  Step 4/5: 吉他轉譜（SynthTab — 吉他專用模型）")
        print("=" * 50)
        try:
            st_result = synthtab_transcribe(
                audio_path=result.guitar_track_path,
                tuning=tuning_midi,
            )
            result.notes = st_result["notes"]
            result.steps_completed.append(
                f"SynthTab 轉譜（{len(result.notes)} 個音符）"
            )
        except Exception as e:
            result.errors.append(f"SynthTab 轉譜失敗，退回 Basic Pitch: {e}")
            use_synthtab = False

    if not use_synthtab:
        print("  Step 4/5: 音高偵測 → MIDI（Basic Pitch）")
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
                f"Basic Pitch 偵測（{len(result.notes)} 個音符）"
            )
        except Exception as e:
            result.errors.append(f"音高偵測失敗: {e}")
            result.elapsed_time = time.time() - start_time
            return result

        # 泛音偵測（只在 Basic Pitch 模式下執行）
        print("\n  泛音偵測...")
        try:
            result.notes = detect_harmonics(
                audio_path=result.guitar_track_path,
                notes=result.notes,
                tuning=tuning_midi,
            )
            harmonic_count = sum(1 for n in result.notes if n.get("is_harmonic"))
            result.steps_completed.append(f"泛音偵測（{harmonic_count} 個泛音）")
        except Exception as e:
            result.errors.append(f"泛音偵測失敗: {e}")

    # ─── Capo 移調（泛音偵測之後，Tab 生成之前）───
    if capo > 0:
        print(f"\n  Capo {capo}: 非泛音音符降 {capo} 半音（對應 Capo 相對格位）")
        import pretty_midi as pm
        for note in result.notes:
            if not note.get("is_harmonic"):
                note["pitch"] -= capo
                try:
                    note["pitch_name"] = pm.note_number_to_name(note["pitch"])
                except Exception:
                    pass
        # 過濾掉移調後低於調弦最低音的音符
        from .tab_generator import TUNINGS
        tuning_midi = TUNINGS.get(tuning, TUNINGS["standard"])
        lowest_note = min(tuning_midi)  # 調弦的最低音
        before = len(result.notes)
        result.notes = [n for n in result.notes if n["pitch"] >= lowest_note]
        filtered = before - len(result.notes)
        if filtered > 0:
            print(f"  過濾 {filtered} 個超出吉他音域的音符")

    # ─── Step 6: 和弦辨識 + Tab 生成 ───
    print("\n" + "=" * 50)
    print("  Step 5/5: 和弦辨識 & Tab 生成")
    print("=" * 50)

    # 和弦辨識（SynthTab 模式優先用 notes，音高更準確）
    try:
        if chord_method == "audio":
            audio_for_chords = result.guitar_track_path or result.audio_path
            result.chords = recognize_chords_from_audio(audio_for_chords)
        elif chord_method == "midi":
            if result.notes:
                result.chords = recognize_chords_from_notes(result.notes)
        else:  # "both"
            # 優先用 notes（特別是 SynthTab 的結果更可靠）
            if result.notes:
                result.chords = recognize_chords_from_notes(result.notes)
            if not result.chords:
                audio_for_chords = result.guitar_track_path or result.audio_path
                result.chords = recognize_chords_from_audio(audio_for_chords)

        # Capo + 調弦偏移：移調和弦名稱（顯示相對 Capo 的和弦手型）
        if result.chords:
            from .tab_generator import TUNINGS
            tuning_midi = TUNINGS.get(tuning, TUNINGS["standard"])
            standard_midi = TUNINGS["standard"]
            # 調弦偏移 = 當前調弦最低音 - 標準調弦最低音
            tuning_offset = min(tuning_midi) - min(standard_midi)
            total_transpose = -(capo + tuning_offset)
            if total_transpose != 0:
                result.chords = _transpose_chords(result.chords, total_transpose)

        result.chord_progression = format_chord_progression(result.chords)
        result.steps_completed.append("和弦辨識")
    except Exception as e:
        result.errors.append(f"和弦辨識失敗: {e}")

    # Tab 生成
    try:
        from .tab_generator import TabGenerator
        tab_gen = TabGenerator(tuning=tuning)

        # SynthTab 已有 string/fret，不需要重新分配
        has_frets = all(n.get("string", -1) >= 0 and n.get("fret", -1) >= 0
                       for n in result.notes[:10]) if result.notes else False
        if has_frets:
            assigned_notes = result.notes
        else:
            assigned_notes = tab_gen.assign_fret_positions(result.notes)

        result.tab = tab_gen.generate_ascii_tab(
            notes=assigned_notes,
            chars_per_second=chars_per_second,
            line_width=line_width,
            title=title,
            chords=result.chords,
        )
        result.assigned_notes = assigned_notes
        result.steps_completed.append("六線譜生成")
    except Exception as e:
        result.errors.append(f"Tab 生成失敗: {e}")
        result.assigned_notes = []

    # ─── 儲存結果 ───
    result.elapsed_time = time.time() - start_time

    if save_intermediate:
        _save_results(result, output_dir, tuning)

    return result


def _save_results(result: TranscriptionResult, output_dir: Path, tuning: str = "standard"):
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
        # 去除不可序列化的欄位，並轉換 numpy 型別
        import numpy as np
        serializable_notes = []
        for n in result.notes:
            sn = {}
            for k, v in n.items():
                if k == "midi_data":
                    continue
                if isinstance(v, (np.integer,)):
                    sn[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    sn[k] = float(v)
                else:
                    sn[k] = v
            serializable_notes.append(sn)
        with open(notes_path, "w", encoding="utf-8") as f:
            json.dump(serializable_notes, f, ensure_ascii=False, indent=2)
        print(f"音符資料已儲存至: {notes_path}")

    # 儲存 Tab (TXT)
    if result.tab:
        tab_path = output_dir / "guitar_tab.txt"
        with open(tab_path, "w", encoding="utf-8") as f:
            f.write(result.tab)
        print(f"六線譜已儲存至: {tab_path}")

    # 儲存 Tab (PDF)
    assigned = getattr(result, "assigned_notes", [])
    if assigned:
        try:
            from .tab_generator import TUNINGS
            tuning_midi = TUNINGS.get(tuning, TUNINGS["standard"])
            pdf_path = output_dir / "guitar_tab.pdf"
            result.pdf_path = generate_pdf_tab(
                notes=assigned,
                output_path=pdf_path,
                tuning_name=tuning,
                tuning_midi=tuning_midi,
                title=result.video_info.get("title", ""),
                chords=result.chords,
                capo=result.capo,
                bpm=result.bpm,
            )
        except ImportError:
            print("  注意: reportlab 未安裝，跳過 PDF 生成。")
            print("  安裝方式: pip install reportlab")
        except Exception as e:
            print(f"  PDF 生成失敗: {e}")


_CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_CHROMATIC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def _transpose_chords(chords: list[dict], semitones: int) -> list[dict]:
    """
    將和弦名稱移調指定半音數。
    例: semitones=-4 → B → G, F# → D, D#m → Bm
    """
    transposed = []
    for chord in chords:
        c = chord.copy()
        name = c.get("chord", "")
        if name and name != "N.C.":
            c["chord"] = _transpose_chord_name(name, semitones)
        transposed.append(c)
    return transposed


def _transpose_chord_name(name: str, semitones: int) -> str:
    """移調單一和弦名稱"""
    # 解析根音（可能是 1 或 2 字元，如 C, C#, Db, Ab）
    root = ""
    suffix = ""
    if len(name) >= 2 and name[1] in ("#", "b"):
        root = name[:2]
        suffix = name[2:]
    elif len(name) >= 1:
        root = name[0]
        suffix = name[1:]
    else:
        return name

    # 找到根音在 chromatic scale 中的位置
    root_idx = None
    for i, n in enumerate(_CHROMATIC):
        if root == n:
            root_idx = i
            break
    if root_idx is None:
        for i, n in enumerate(_CHROMATIC_FLAT):
            if root == n:
                root_idx = i
                break
    if root_idx is None:
        return name  # 無法辨識的根音，原樣返回

    new_idx = (root_idx + semitones) % 12
    new_root = _CHROMATIC[new_idx]
    return new_root + suffix


def _extract_video_id(url: str) -> str:
    """從 YouTube URL 提取影片 ID"""
    parsed = urlparse(url)
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(parsed.query).get("v", ["unknown"])[0]
    elif parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    return "unknown"


def _make_folder_name(title: str, video_id: str) -> str:
    """
    用影片標題 + ID 建立安全的資料夾名稱。
    格式: {sanitized_title}_{video_id}
    例如: 告白气球_泛音版_VIf6U2QrjCs
    """
    # 移除不安全的檔名字元，保留中文、英文、數字
    safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', title)
    # 截斷太長的標題
    safe_title = safe_title[:60].strip().rstrip(".")
    if not safe_title:
        safe_title = "untitled"
    return f"{safe_title}_{video_id}"
