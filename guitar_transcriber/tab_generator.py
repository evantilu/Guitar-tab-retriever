"""
模組 5: MIDI → 吉他六線譜（Tab）轉換
將 MIDI 音符對應到吉他指板上的具體把位，並輸出 ASCII Tab 格式
"""

from typing import Optional
import numpy as np


# 標準吉他調音 (從第六弦到第一弦): E2, A2, D3, G3, B3, E4
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
STRING_NAMES = ["e", "B", "G", "D", "A", "E"]  # 顯示順序：高音到低音

# 常見特殊調弦
TUNINGS = {
    "standard":   [40, 45, 50, 55, 59, 64],
    "drop_d":     [38, 45, 50, 55, 59, 64],
    "open_g":     [38, 43, 50, 55, 59, 62],
    "open_d":     [38, 45, 50, 54, 57, 62],
    "dadgad":     [38, 45, 50, 55, 57, 62],
    "half_step_down": [39, 44, 49, 54, 58, 63],
}

MAX_FRET = 24  # 最高把位


class TabGenerator:
    """吉他六線譜生成器"""

    def __init__(
        self,
        tuning: list[int] | str = "standard",
        max_fret: int = MAX_FRET,
        max_stretch: int = 5,
    ):
        """
        Args:
            tuning: 調弦設定（MIDI 編號列表或預設名稱）
            max_fret: 最高把位
            max_stretch: 同一時間點內手指的最大跨格數
        """
        if isinstance(tuning, str):
            if tuning not in TUNINGS:
                raise ValueError(
                    f"未知調弦: {tuning}。可用: {list(TUNINGS.keys())}"
                )
            self.tuning = TUNINGS[tuning]
        else:
            self.tuning = tuning

        self.max_fret = max_fret
        self.max_stretch = max_stretch
        self.tuning_name = tuning if isinstance(tuning, str) else "custom"

    def midi_to_fret_options(self, midi_pitch: int) -> list[tuple[int, int]]:
        """
        找出一個 MIDI 音高在吉他上所有可能的 (弦, 格) 組合。

        Args:
            midi_pitch: MIDI 音高編號

        Returns:
            [(string_index, fret), ...] 的列表，string_index 0=第六弦
        """
        options = []
        for string_idx, open_pitch in enumerate(self.tuning):
            fret = midi_pitch - open_pitch
            if 0 <= fret <= self.max_fret:
                options.append((string_idx, fret))
        return options

    def assign_fret_positions(
        self, notes: list[dict]
    ) -> list[dict]:
        """
        為一組音符分配最佳的吉他指板位置。

        使用貪心演算法 + 局部最佳化：
        - 泛音音符直接使用泛音偵測結果的弦和格位
        - 優先選擇低把位（對初學者友好）
        - 同時發聲的音符必須在不同弦上
        - 考慮手指跨度限制

        Args:
            notes: 音符列表（來自 pitch_detector，可能含泛音標記）

        Returns:
            增加了 string, fret 欄位的音符列表
        """
        if not notes:
            return []

        # 將音符按時間分組（同時發聲的音符歸為一組）
        groups = self._group_simultaneous_notes(notes)
        result = []

        prev_position = None  # 上一組的平均把位，用於最小化移動

        for group in groups:
            assigned = self._assign_group(group, prev_position)
            result.extend(assigned)

            # 更新把位參考
            frets = [n["fret"] for n in assigned if n["fret"] > 0]
            if frets:
                prev_position = np.mean(frets)

        return result

    def _group_simultaneous_notes(
        self, notes: list[dict], tolerance: float = 0.05
    ) -> list[list[dict]]:
        """
        將幾乎同時開始的音符歸為一組（和弦或琶音）。

        Args:
            notes: 音符列表
            tolerance: 時間容差（秒），小於此值視為同時

        Returns:
            音符組列表
        """
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda n: n["start"])
        groups = []
        current_group = [sorted_notes[0]]

        for note in sorted_notes[1:]:
            if note["start"] - current_group[0]["start"] <= tolerance:
                current_group.append(note)
            else:
                groups.append(current_group)
                current_group = [note]

        groups.append(current_group)
        return groups

    def _assign_group(
        self,
        group: list[dict],
        prev_position: Optional[float] = None,
    ) -> list[dict]:
        """
        為一組同時發聲的音符分配弦和格。

        使用回溯法找到最佳分配：
        - 泛音音符直接使用預先偵測的弦/格位
        - 每個音符必須在不同弦上
        - 最小化手指跨度
        - 偏好接近上一個位置的把位
        """
        # 先處理泛音音符（已有確定位置）
        pre_assigned = {}  # idx -> (string, fret)
        used_strings = set()
        for idx, note in enumerate(group):
            if note.get("is_harmonic"):
                s = note["harmonic_string"]
                f = note["harmonic_fret"]
                pre_assigned[idx] = (s, f)
                used_strings.add(s)

        # 為非泛音音符找出所有可能的選項
        all_options = []
        for idx, note in enumerate(group):
            if idx in pre_assigned:
                all_options.append([pre_assigned[idx]])
            else:
                options = self.midi_to_fret_options(note["pitch"])
                if not options:
                    all_options.append([])
                else:
                    all_options.append(options)

        # 用回溯法找最佳分配
        best_assignment = None
        best_score = float("inf")

        def backtrack(idx, bt_used_strings, current_assignment):
            nonlocal best_assignment, best_score

            if idx == len(group):
                score = self._score_assignment(
                    current_assignment, prev_position
                )
                if score < best_score:
                    best_score = score
                    best_assignment = current_assignment.copy()
                return

            if not all_options[idx]:
                # 無法分配，用 -1 標記
                current_assignment.append((-1, -1))
                backtrack(idx + 1, bt_used_strings, current_assignment)
                current_assignment.pop()
                return

            for string_idx, fret in all_options[idx]:
                if string_idx in bt_used_strings and idx not in pre_assigned:
                    continue

                current_assignment.append((string_idx, fret))
                bt_used_strings.add(string_idx)
                backtrack(idx + 1, bt_used_strings, current_assignment)
                if idx not in pre_assigned:
                    bt_used_strings.remove(string_idx)
                current_assignment.pop()

        backtrack(0, set(used_strings), [])

        # 將分配結果寫回音符
        result = []
        for i, note in enumerate(group):
            note_copy = note.copy()
            if best_assignment and i < len(best_assignment):
                string_idx, fret = best_assignment[i]
                note_copy["string"] = string_idx  # 0-5
                note_copy["fret"] = fret
            else:
                note_copy["string"] = -1
                note_copy["fret"] = -1
            result.append(note_copy)

        return result

    def _score_assignment(
        self,
        assignment: list[tuple[int, int]],
        prev_position: Optional[float] = None,
    ) -> float:
        """
        評估一組分配的好壞（分數越低越好）。

        考慮因素：
        1. 手指跨度（格之間的距離）
        2. 偏好低把位
        3. 與前一組把位的距離（最小化手部移動）
        """
        frets = [f for _, f in assignment if f >= 0]
        if not frets:
            return 0

        score = 0.0

        # 1. 手指跨度
        if len(frets) > 1:
            span = max(frets) - min(frets)
            if span > self.max_stretch:
                score += 100  # 嚴重懲罰超過跨度的分配
            else:
                score += span * 2

        # 2. 偏好低把位（但不過度偏好空弦）
        avg_fret = np.mean(frets)
        score += avg_fret * 0.5

        # 3. 與前一位置的距離
        if prev_position is not None:
            distance = abs(avg_fret - prev_position)
            score += distance * 1.5

        return score

    def generate_ascii_tab(
        self,
        notes: list[dict],
        chars_per_second: float = 8.0,
        line_width: int = 80,
        title: str = "",
        chords: Optional[list[dict]] = None,
    ) -> str:
        """
        生成 ASCII 格式的吉他六線譜。

        Args:
            notes: 已分配 string/fret 的音符列表
            chars_per_second: 每秒對應的字元數（控制譜面密度）
            line_width: 每行最大字元數
            title: 標題
            chords: 和弦列表（可選，會顯示在 Tab 上方）

        Returns:
            ASCII Tab 字串
        """
        if not notes:
            return "（無音符資料）"

        # 計算總時間
        max_time = max(n["end"] for n in notes)
        total_chars = int(max_time * chars_per_second) + 1

        # 初始化六條弦（用 - 填充）
        # 索引 0 = 第六弦 (E), ..., 5 = 第一弦 (e)
        strings = [list("-" * total_chars) for _ in range(6)]

        # 和弦行
        chord_line = list(" " * total_chars) if chords else None

        # 放入音符
        for note in notes:
            if note.get("string", -1) < 0 or note.get("fret", -1) < 0:
                continue

            pos = int(note["start"] * chars_per_second)
            if pos >= total_chars:
                continue

            string_idx = note["string"]

            # 泛音用 <N> 標記，一般音符只顯示數字
            if note.get("is_harmonic"):
                fret_str = f"<{note['fret']}>"
            else:
                fret_str = str(note["fret"])

            # 寫入格數（可能是多位字元）
            for j, ch in enumerate(fret_str):
                if pos + j < total_chars:
                    strings[string_idx][pos + j] = ch

        # 放入和弦標記
        if chords and chord_line:
            for chord in chords:
                if chord["chord"] == "N.C.":
                    continue
                pos = int(chord["start"] * chars_per_second)
                chord_name = chord["chord"]
                for j, ch in enumerate(chord_name):
                    if pos + j < total_chars:
                        chord_line[pos + j] = ch

        # 組裝輸出
        output_lines = []

        if title:
            output_lines.append(f"  {title}")
            output_lines.append(f"  調弦: {self.tuning_name}")
            output_lines.append("")

        # 將長的 tab 拆成多行
        usable_width = line_width - 4  # 減去弦名和前綴

        for start in range(0, total_chars, usable_width):
            end = min(start + usable_width, total_chars)

            # 時間標記
            time_start = start / chars_per_second
            time_end = end / chars_per_second
            output_lines.append(
                f"  [{self._format_time(time_start)} - "
                f"{self._format_time(time_end)}]"
            )

            # 和弦行
            if chord_line:
                chord_segment = "".join(chord_line[start:end])
                output_lines.append(f"    {chord_segment}")

            # 六條弦（從高音到低音顯示）
            display_order = [5, 4, 3, 2, 1, 0]  # e, B, G, D, A, E
            for disp_idx in display_order:
                string_segment = "".join(strings[disp_idx][start:end])
                name = STRING_NAMES[disp_idx]
                output_lines.append(f"  {name}|{string_segment}|")

            output_lines.append("")

        return "\n".join(output_lines)

    def _format_time(self, seconds: float) -> str:
        """將秒數格式化為 MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


def generate_tab_from_notes(
    notes: list[dict],
    tuning: str = "standard",
    chords: Optional[list[dict]] = None,
    title: str = "",
    chars_per_second: float = 8.0,
    line_width: int = 80,
) -> str:
    """
    便捷函式：從音符列表直接生成六線譜。

    Args:
        notes: 來自 pitch_detector 的音符列表
        tuning: 調弦名稱
        chords: 和弦列表（可選）
        title: 標題
        chars_per_second: 譜面密度
        line_width: 行寬

    Returns:
        ASCII Tab 字串
    """
    generator = TabGenerator(tuning=tuning)

    # 分配指板位置
    assigned_notes = generator.assign_fret_positions(notes)

    # 生成 Tab
    tab = generator.generate_ascii_tab(
        notes=assigned_notes,
        chars_per_second=chars_per_second,
        line_width=line_width,
        title=title,
        chords=chords,
    )

    return tab
