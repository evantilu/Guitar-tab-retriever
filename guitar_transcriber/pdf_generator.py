"""
模組 6: PDF 六線譜 + 五線譜輸出
使用 music21 做節奏量化，LilyPond 渲染專業 PDF。
五線譜 (Staff) + 六線譜 (TAB) 共用同一份音樂資料。
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

_NOTE_NAMES = ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]


def _midi_to_ly(midi_pitch: int) -> str:
    """MIDI pitch → LilyPond absolute pitch (e.g. 60→c', 64→e', 48→c, 40→e,)"""
    name = _NOTE_NAMES[midi_pitch % 12]
    ly_oct = midi_pitch // 12 - 4  # c' is MIDI octave 5 (60-71)
    if ly_oct > 0:
        name += "'" * ly_oct
    elif ly_oct < 0:
        name += "," * (-ly_oct)
    return name


def _ql_to_ly_dur(ql: float) -> str:
    """music21 quarterLength → LilyPond duration"""
    mapping = [
        (4.0, "1"), (3.0, "2."), (2.0, "2"), (1.5, "4."), (1.0, "4"),
        (0.75, "8."), (0.5, "8"), (0.375, "16."), (0.25, "16"), (0.125, "32"),
    ]
    if ql <= 0:
        return "16"
    best = min(mapping, key=lambda x: abs(x[0] - ql))
    return best[1]


def _string_ly(string_idx: int) -> str:
    """Guitar string (0=low E) → LilyPond \\N (6=low E, 1=high e)"""
    return str(6 - string_idx)


def _escape_ly(text: str) -> str:
    return text.replace('"', "'").replace("\\", "")


def generate_pdf_tab(
    notes: list[dict],
    output_path: str | Path,
    tuning_name: str = "standard",
    tuning_midi: list[int] | None = None,
    title: str = "",
    chords: Optional[list[dict]] = None,
    chars_per_second: float = 8.0,
    line_width: int = 72,
    capo: int = 0,
    bpm: float = 120.0,
) -> Path:
    output_path = Path(output_path)
    if shutil.which("lilypond"):
        return _generate_lilypond_pdf(
            notes, output_path, tuning_name, tuning_midi, title, chords, capo, bpm)
    else:
        return _generate_reportlab_pdf(
            notes, output_path, tuning_name, title, chords,
            chars_per_second, line_width, capo)


def _quantize_notes(notes: list[dict], bpm: float) -> list[dict]:
    """
    用 music21 量化音符到節拍格線，並加入休止符填滿小節。
    返回量化後的事件列表（音符和休止符）。
    """
    import music21

    if not notes:
        return []

    beat_dur = 60.0 / bpm

    # 建立 music21 Part
    part = music21.stream.Part()
    part.insert(0, music21.meter.TimeSignature('4/4'))

    # 將音符按時間分組
    sorted_notes = sorted(notes, key=lambda n: (n["start"], n["pitch"]))
    groups = []
    current = [sorted_notes[0]]
    for n in sorted_notes[1:]:
        if n["start"] - current[0]["start"] <= 0.05:
            current.append(n)
        else:
            groups.append(current)
            current = [n]
    groups.append(current)

    for group in groups:
        offset_beats = group[0]["start"] / beat_dur
        dur_beats = max(group[0].get("duration", 0.25) / beat_dur, 0.125)

        if len(group) == 1:
            n = group[0]
            m_note = music21.note.Note(n["pitch"])
            m_note.quarterLength = dur_beats
            m_note.guitar_data = n
            part.insert(offset_beats, m_note)
        else:
            pitches = [music21.pitch.Pitch(midi=n["pitch"]) for n in group]
            m_chord = music21.chord.Chord(pitches)
            m_chord.quarterLength = dur_beats
            m_chord.guitar_data = group
            part.insert(offset_beats, m_chord)

    # 量化
    part.quantize(quarterLengthDivisors=(4, 3), inPlace=True)

    # 加入小節結構
    measured = part.makeMeasures(inPlace=False)

    # 提取量化結果為列表
    events = []
    for el in measured.recurse().notesAndRests:
        ql = el.quarterLength
        if ql <= 0:
            continue

        if isinstance(el, music21.note.Rest):
            events.append({"type": "rest", "duration": ql})
        elif isinstance(el, music21.note.Note):
            gd = getattr(el, 'guitar_data', {})
            events.append({
                "type": "note",
                "pitch": el.pitch.midi,
                "duration": ql,
                "string": gd.get("string", -1),
                "fret": gd.get("fret", -1),
                "is_harmonic": gd.get("is_harmonic", False),
                "harmonic_fret": gd.get("harmonic_fret", -1),
                "harmonic_string": gd.get("harmonic_string", -1),
            })
        elif isinstance(el, music21.chord.Chord):
            gd_list = getattr(el, 'guitar_data', [])
            chord_notes = []
            for i, p in enumerate(el.pitches):
                gd = gd_list[i] if isinstance(gd_list, list) and i < len(gd_list) else {}
                chord_notes.append({
                    "pitch": p.midi,
                    "string": gd.get("string", -1),
                    "fret": gd.get("fret", -1),
                    "is_harmonic": gd.get("is_harmonic", False),
                    "harmonic_fret": gd.get("harmonic_fret", -1),
                    "harmonic_string": gd.get("harmonic_string", -1),
                })
            events.append({
                "type": "chord",
                "notes": chord_notes,
                "duration": ql,
            })

    return events


_STANDARD_TUNING = [40, 45, 50, 55, 59, 64]


def _harmonic_tab_pitch(note: dict) -> int:
    """
    對泛音音符，計算 TAB 上應顯示的「等效按弦音高」。
    例：第 5 弦(A=45) 的 7 格泛音 → TAB 顯示 45+7=52 (即 D3 的 7 格位置)
    """
    h_string = note.get("harmonic_string", -1)
    h_fret = note.get("harmonic_fret", -1)
    if h_string >= 0 and h_fret >= 0:
        open_pitch = _STANDARD_TUNING[h_string] if h_string < len(_STANDARD_TUNING) else 40
        return open_pitch + h_fret
    return note["pitch"]


def _events_to_lilypond(events: list[dict], include_strings: bool = False) -> str:
    """
    將量化事件轉為 LilyPond 音樂字串。
    include_strings=True: TAB 模式，泛音用等效按弦音高+弦號。
    include_strings=False: Staff 模式，用實際音高，泛音加 flageolet。
    """
    parts = []

    for ev in events:
        dur = _ql_to_ly_dur(ev["duration"])

        if ev["type"] == "rest":
            parts.append(f"r{dur}")

        elif ev["type"] == "note":
            is_harm = ev.get("is_harmonic", False)

            if include_strings:
                # TAB: 泛音用等效按弦音高
                midi = _harmonic_tab_pitch(ev) if is_harm else ev["pitch"]
                pitch = _midi_to_ly(midi)
                s = ev.get("harmonic_string", -1) if is_harm else ev.get("string", -1)
                string_str = f"\\{_string_ly(s)}" if s >= 0 else ""
                parts.append(f"{pitch}{dur}{string_str}")
            else:
                # Staff: 用實際音高
                pitch = _midi_to_ly(ev["pitch"])
                flag = "\\flageolet" if is_harm else ""
                parts.append(f"{pitch}{dur}{flag}")

        elif ev["type"] == "chord":
            note_strs = []
            for n in ev["notes"]:
                is_harm = n.get("is_harmonic", False)

                if include_strings:
                    midi = _harmonic_tab_pitch(n) if is_harm else n["pitch"]
                    p = _midi_to_ly(midi)
                    s = n.get("harmonic_string", -1) if is_harm else n.get("string", -1)
                    if s >= 0:
                        p += f"\\{_string_ly(s)}"
                else:
                    p = _midi_to_ly(n["pitch"])

                note_strs.append(p)
            parts.append(f"<{' '.join(note_strs)}>{dur}")

    return "\n  ".join(parts)


def _chords_to_lilypond(chords: list[dict], bpm: float, total_beats: float) -> str:
    """
    和弦列表 → LilyPond \\chordmode 字串。
    按時間位置對齊，用休止符填滿間隙，確保和弦與音樂同步。
    """
    if not chords:
        return ""

    beat_dur = 60.0 / bpm

    _roots = {
        "C": "c", "C#": "cis", "Db": "des", "D": "d", "D#": "dis", "Eb": "ees",
        "E": "e", "F": "f", "F#": "fis", "Gb": "ges", "G": "g", "G#": "gis",
        "Ab": "aes", "A": "a", "A#": "ais", "Bb": "bes", "B": "b",
    }
    _quality = {
        "": "", "m": ":m", "7": ":7", "maj7": ":maj7", "m7": ":m7",
        "dim": ":dim", "aug": ":aug", "sus2": ":sus2", "sus4": ":sus4",
        "5": ":5", "6": ":6", "9": ":9", "m9": ":m9", "m6": ":m7",
        "add9": ":maj9", "maj9": ":maj9",
    }

    def _chord_name_to_ly(name, dur_str):
        root_str, suffix = "", ""
        if len(name) >= 2 and name[1] in ("#", "b"):
            root_str, suffix = name[:2], name[2:]
        elif len(name) >= 1:
            root_str, suffix = name[0], name[1:]
        ly_root = _roots.get(root_str, "c")
        q = ""
        for k in sorted(_quality.keys(), key=lambda x: -len(x)):
            if suffix == k:
                q = _quality[k]
                break
        return f"{ly_root}{dur_str}{q}"

    # 簡化和弦：量化到每 2 拍一個和弦，合併重複
    # 先建立每 2 拍的和弦 grid
    total_half_notes = int(total_beats / 2) + 1
    chord_grid = [""] * total_half_notes

    for ch in chords:
        name = ch.get("chord", "")
        if not name or name == "N.C.":
            continue
        beat = ch["start"] / beat_dur
        grid_idx = int(beat / 2)
        if 0 <= grid_idx < len(chord_grid) and not chord_grid[grid_idx]:
            chord_grid[grid_idx] = name

    # 生成 LilyPond 和弦序列（每個和弦佔 2 拍 = half note）
    parts = []
    for name in chord_grid:
        if name:
            parts.append(_chord_name_to_ly(name, "2"))
        else:
            parts.append("r2")

    return "chordSeq = \\chordmode { " + " ".join(parts) + " }"


def _generate_lilypond_pdf(
    notes, output_path, tuning_name, tuning_midi, title, chords, capo, bpm=120.0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not notes:
        ly = f'\\header {{ title = "{_escape_ly(title)}" }}\n{{ s1 }}'
        return _compile_lilypond(ly, output_path)

    # 量化
    events = _quantize_notes(notes, bpm)

    # 計算總拍數
    beat_dur = 60.0 / bpm
    max_time = max(n["end"] for n in notes)
    total_beats = max_time / beat_dur

    # 生成音樂字串（五線譜不含弦號，TAB 含弦號）
    staff_music_str = _events_to_lilypond(events, include_strings=False)
    tab_music_str = _events_to_lilypond(events, include_strings=True)

    # 和弦
    chord_def = _chords_to_lilypond(chords, bpm, total_beats) if chords else ""

    # 調弦
    if tuning_midi is None:
        tuning_midi = [40, 45, 50, 55, 59, 64]
    tuning_str = " ".join(_midi_to_ly(p) for p in tuning_midi)

    capo_text = f"Capo fret {capo}" if capo > 0 else ""

    ly = f"""\\version "2.24.0"

\\header {{
  title = "{_escape_ly(title)}"
  subtitle = "{_escape_ly(capo_text)}"
  tagline = "Generated by Guitar Tab Retriever"
}}

\\paper {{
  #(set-paper-size "a4")
  top-margin = 12\\mm
  bottom-margin = 12\\mm
  left-margin = 10\\mm
  right-margin = 10\\mm
  indent = 12\\mm
  system-system-spacing.basic-distance = #16
  ragged-last = ##t
  %% 強制每行最多 4 小節
}}

\\layout {{
  \\context {{
    \\Score
    %% 增大最短音符間距，讓每行放更少小節
    \\override SpacingSpanner.base-shortest-duration = #(ly:make-moment 1/8)
    \\override SpacingSpanner.common-shortest-duration = #(ly:make-moment 1/8)
  }}
  \\context {{
    \\Staff
    \\override StringNumber.stencil = ##f
  }}
}}

{chord_def}

staffMusic = {{
  \\clef "treble_8"
  \\key c \\major
  \\time 4/4
  \\tempo 4 = {int(bpm)}
  {staff_music_str}
  \\bar "|."
}}

tabMusic = {{
  \\time 4/4
  {tab_music_str}
  \\bar "|."
}}

\\score {{
  <<
    {"\\new ChordNames { \\chordSeq }" if chord_def else ""}
    \\new StaffGroup <<
      \\new Staff \\with {{
        instrumentName = "Guitar"
      }} {{
        \\staffMusic
      }}
      \\new TabStaff \\with {{
        stringTunings = \\stringTuning <{tuning_str}>
      }} {{
        \\tabMusic
      }}
    >>
  >>
}}
"""
    return _compile_lilypond(ly, output_path)


def _compile_lilypond(ly_content: str, output_path: Path) -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        ly_file = Path(tmpdir) / "score.ly"
        ly_file.write_text(ly_content, encoding="utf-8")

        result = subprocess.run(
            ["lilypond", "-dno-point-and-click",
             "-o", str(Path(tmpdir) / "score"), str(ly_file)],
            capture_output=True, text=True, cwd=tmpdir,
        )

        pdf_result = Path(tmpdir) / "score.pdf"
        import shutil as sh
        ly_out = output_path.with_suffix(".ly")
        sh.copy2(str(ly_file), str(ly_out))

        if pdf_result.exists():
            sh.copy2(str(pdf_result), str(output_path))
            print(f"PDF 六線譜已儲存至: {output_path}")
            print(f"LilyPond 原始碼: {ly_out}")
            return output_path
        else:
            err = result.stderr[-800:] if result.stderr else "unknown"
            print(f"  LilyPond 編譯失敗，原始碼: {ly_out}")
            print(f"  錯誤: {err}")
            raise RuntimeError(f"LilyPond failed: {err}")


# ── Reportlab fallback ──

def _generate_reportlab_pdf(
    notes, output_path, tuning_name, title, chords,
    chars_per_second, line_width, capo,
) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    STRING_NAMES = ["e", "B", "G", "D", "A", "E"]

    cjk = None
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        cjk = "STSong-Light"
    except Exception:
        pass

    if not notes:
        c = canvas.Canvas(str(output_path), pagesize=A4)
        c.drawString(50, 800, "(No data)")
        c.save()
        return output_path

    mt = max(n["end"] for n in notes)
    tc = int(mt * chars_per_second) + 1
    strings = [list("-" * tc) for _ in range(6)]

    for n in notes:
        if n.get("string", -1) < 0 or n.get("fret", -1) < 0:
            continue
        pos = int(n["start"] * chars_per_second)
        if pos >= tc:
            continue
        si = n["string"]
        fs = f"<{n['fret']}>" if n.get("is_harmonic") else str(n["fret"])
        for j, ch in enumerate(fs):
            if pos + j < tc:
                strings[si][pos + j] = ch

    pw, ph = A4
    c = canvas.Canvas(str(output_path), pagesize=A4)
    ml, mt_m, mb = 25 * mm, 25 * mm, 20 * mm
    y = ph - mt_m
    lh = 12.6

    if title:
        c.setFont(cjk or "Helvetica-Bold", 16)
        c.drawString(ml, y, title)
        y -= 22
        c.setFont(cjk or "Helvetica", 10)
        c.drawString(ml, y, f"Tuning: {tuning_name}")
        if capo > 0:
            y -= 14
            c.drawString(ml, y, f"Capo: fret {capo}")
        y -= 20

    c.setFont("Courier", 9)
    for start in range(0, tc, line_width):
        end = min(start + line_width, tc)
        if y - lh * 8 < mb:
            c.showPage()
            c.setFont("Courier", 9)
            y = ph - mt_m
        for di in [5, 4, 3, 2, 1, 0]:
            seg = "".join(strings[di][start:end])
            c.drawString(ml, y, f"{STRING_NAMES[di]}|{seg}|")
            y -= lh
        y -= 10

    c.save()
    print(f"PDF: {output_path}")
    return output_path
