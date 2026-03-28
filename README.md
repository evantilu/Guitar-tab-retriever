# Guitar Tab Retriever

從 YouTube 影片自動擷取吉他六線譜（Tab）與和弦進行的 CLI 工具。

---

## 系統架構

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    Guitar Tab Retriever                         │
  │                                                                 │
  │   YouTube URL                                                   │
  │       │                                                         │
  │       ▼                                                         │
  │  ┌──────────┐    ┌──────────────┐    ┌───────────────┐          │
  │  │  yt-dlp  │───▶│   Demucs     │───▶│  Basic Pitch  │          │
  │  │ 音訊擷取  │    │  音源分離     │    │  音高偵測      │          │
  │  └──────────┘    └──────────────┘    └───────┬───────┘          │
  │       │                │                     │                  │
  │       │           WAV (各軌道)            MIDI 音符              │
  │       │                │                     │                  │
  │       │                ▼                     ▼                  │
  │       │         ┌──────────────┐    ┌───────────────┐           │
  │       │         │   librosa    │    │ Tab Generator │           │
  │       │         │  和弦辨識     │    │  六線譜生成    │           │
  │       │         └──────┬───────┘    └───────┬───────┘           │
  │       │                │                     │                  │
  │       │                ▼                     ▼                  │
  │       │          和弦進行表            ASCII 六線譜               │
  │       │           (JSON)               (TXT)                    │
  │       │                │                     │                  │
  │       │                └────────┬────────────┘                  │
  │       │                         ▼                               │
  │       │              ┌─────────────────┐                        │
  │       └─────────────▶│  transcription  │                        │
  │          影片資訊      │    .txt 完整報告 │                        │
  │                       └─────────────────┘                        │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 各模組說明

```
  guitar_transcriber/
  │
  ├── audio_extractor.py ··· [Step 1] YouTube → WAV
  │   使用 yt-dlp 下載影片音訊，ffmpeg 轉為 44100Hz 單聲道 WAV
  │
  ├── source_separator.py ·· [Step 2] 混合音訊 → 分軌
  │   使用 Meta Demucs 模型分離出 drums / bass / vocals / other
  │   吉他通常在 "other" 軌道中
  │
  │       混合音訊                    分離結果
  │   ┌─────────────┐          ┌─────────────────┐
  │   │ ♪♪♪♪♪♪♪♪♪♪ │  Demucs  │ drums    ♪·♪·♪  │
  │   │ 吉他+人聲+鼓 │ ───────▶ │ bass     ♪~~~♪  │
  │   │ +貝斯+其他   │          │ vocals   ♪♪♪♪♪  │
  │   └─────────────┘          │ other    ♪♪♪♪♪  │◄── 吉他在這裡
  │                            └─────────────────┘
  │
  ├── pitch_detector.py ···· [Step 3] WAV → MIDI 音符
  │   使用 Spotify Basic Pitch 偵測每個音符的音高、起始時間、時值
  │   自動過濾吉他音域外的音符 (E2~E6, MIDI 40~88)
  │
  │       音訊波形                     MIDI 音符
  │   ┌─────────────┐          ┌─────────────────────┐
  │   │ ∿∿∿∿∿∿∿∿∿∿∿ │  Basic   │ C4  ████             │
  │   │ ∿∿∿∿∿∿∿∿∿∿∿ │  Pitch   │ E4     ██████        │
  │   │ ∿∿∿∿∿∿∿∿∿∿∿ │ ───────▶ │ G4        ████       │
  │   │ ∿∿∿∿∿∿∿∿∿∿∿ │          │ C5           ██████  │
  │   └─────────────┘          └─────────────────────┘
  │                              pitch × time
  │
  ├── chord_recognizer.py ·· [Step 4] 和弦辨識
  │   兩種模式：
  │     (a) 從音訊的 chroma 特徵比對和弦模板 (librosa)
  │     (b) 從 MIDI 音符組合推斷和弦
  │   支援 17 種和弦類型 (maj, min, 7, maj7, min7, dim, aug, sus2...)
  │
  │       Chroma 特徵                 和弦比對
  │   C  ███████████              C  ███████████  ← 根音
  │   C# ·                       E  ████████     ← 大三度
  │   D  ··                      G  ██████████   ← 純五度
  │   D# ·                       ─────────────
  │   E  █████████                 = C major
  │   F  ·
  │   F# ·
  │   G  ██████████
  │   ...
  │
  ├── tab_generator.py ····· [Step 5] MIDI → 六線譜
  │   核心挑戰：同一個音高可能出現在不同弦的不同格位
  │   使用回溯法 + 評分函式找最佳指法分配
  │
  │       同一個 E4 (MIDI 64) 在吉他上的可能位置：
  │
  │       e ──0──────────────────  (第1弦 空弦)
  │       B ──5──────────────────  (第2弦 第5格)
  │       G ──9──────────────────  (第3弦 第9格)
  │       D ──14─────────────────  (第4弦 第14格)
  │       A ──19─────────────────  (第5弦 第19格)
  │       E ──24─────────────────  (第6弦 第24格)
  │
  │       演算法選擇最佳位置，考慮：
  │       ✓ 手指跨度 (max_stretch)
  │       ✓ 低把位優先
  │       ✓ 最小化手部移動
  │       ✓ 同時發聲必須在不同弦
  │
  └── pipeline.py ·········· 串接以上所有步驟
```

---

## 前置需求

| 工具 | 用途 | 安裝方式 |
|------|------|----------|
| Python ≥ 3.10 | 執行環境 | [python.org](https://www.python.org/) |
| ffmpeg | 音訊轉檔 | `brew install ffmpeg` (macOS) |
| GPU (選配) | 加速 ML 推論 | 有 NVIDIA GPU + CUDA 會快很多，但 CPU 也能跑 |

---

## 安裝

```bash
cd ~/Documents/Guitar-tab-retriever

# 方法 1：一鍵安裝
bash setup.sh

# 方法 2：手動安裝
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **注意：** 首次安裝會下載 PyTorch（約 2GB），請確保網路穩定。

---

## 使用方式

```bash
# 先啟動虛擬環境
source venv/bin/activate

# 基本用法
python main.py https://www.youtube.com/watch?v=XXXXXXX

# 跳過音源分離（適合純吉他獨奏影片，速度快很多）
python main.py https://www.youtube.com/watch?v=XXXXXXX --skip-separation

# 指定調弦
python main.py https://www.youtube.com/watch?v=XXXXXXX --tuning drop_d

# 使用更精確的模型（較慢）
python main.py https://www.youtube.com/watch?v=XXXXXXX --demucs-model htdemucs_ft

# 強制使用 CPU
python main.py https://www.youtube.com/watch?v=XXXXXXX --device cpu

# 調整偵測靈敏度（閾值越高越嚴格，越低越多音符）
python main.py https://www.youtube.com/watch?v=XXXXXXX --onset-threshold 0.6 --frame-threshold 0.4
```

---

## CLI 參數一覽

```
  必要參數:
    url                    YouTube 影片網址

  選用參數:
    -o, --output           輸出目錄         (預設: ./output)
    -t, --tuning           吉他調弦         (預設: standard)
    --skip-separation      跳過音源分離      (適合獨奏影片)
    --demucs-model         Demucs 模型      (預設: htdemucs)
    --device               運算裝置         (auto / cpu / cuda)
    --onset-threshold      音符起始閾值 0~1  (預設: 0.5)
    --frame-threshold      音框偵測閾值 0~1  (預設: 0.3)
    --chord-method         和弦辨識方式      (audio / midi / both)
    --tab-density          Tab 密度         (預設: 8.0 字元/秒)
    --line-width           Tab 行寬         (預設: 80)
    --no-save              不儲存中間檔案
```

---

## 支援的調弦

```
  standard          E A D G B e     (標準調弦)
  drop_d            D A D G B e     (Drop D)
  open_g            D G D G B D     (Open G)
  open_d            D A D F# A D   (Open D)
  dadgad            D A D G A D    (DADGAD)
  half_step_down    Eb Ab Db Gb Bb Eb  (降半音)
```

---

## 輸出檔案

執行完畢後，`output/` 目錄會包含：

```
  output/
  ├── audio/
  │   └── source.wav            ← 從 YouTube 擷取的原始音訊
  ├── separated/
  │   └── htdemucs/
  │       ├── drums.wav         ← 鼓軌
  │       ├── bass.wav          ← 貝斯軌
  │       ├── vocals.wav        ← 人聲軌
  │       └── other.wav         ← 其他樂器（含吉他）
  ├── midi/
  │   └── guitar.mid            ← MIDI 檔案（可匯入 DAW）
  ├── chords.json               ← 和弦進行資料
  ├── notes.json                ← 所有偵測到的音符資料
  ├── guitar_tab.txt            ← ASCII 六線譜
  └── transcription.txt         ← 完整轉譜報告
```

---

## 輸出範例

### 和弦進行

```
  === 和弦進行 ===

    00:00.00  Am          (2.0s)  87%
    00:02.00  F           (2.0s)  82%
    00:04.00  C           (2.0s)  91%
    00:06.00  G           (1.5s)  78%

  簡化進行: Am → F → C → G
```

### 六線譜

```
    Fingerstyle Guitar
    調弦: standard

    [00:00 - 00:09]
    Am              F               C
  e|-----0-------1-|-------0-------|
  B|---1---1---1---|---1-----1-----|
  G|-2-------2---2-|-0---------0---|
  D|---------------|---------------|
  A|0-----------0--|---------------|
  E|---------------|---1-----------|
```

---

## 處理流程時間參考

```
  以一首 3 分鐘的影片為例（CPU 環境）：

  Step 1  音訊擷取     ████                          ~10 秒
  Step 2  音源分離     ████████████████████████████   ~3-5 分鐘
  Step 3  音高偵測     ████████████                   ~1-2 分鐘
  Step 4  和弦辨識     ███                            ~5 秒
  Step 5  Tab 生成     █                              ~1 秒
                       ─────────────────────────────
                       總計約 5-8 分鐘 (CPU)
                       總計約 1-3 分鐘 (GPU)

  使用 --skip-separation 可跳過 Step 2，總計約 1-3 分鐘 (CPU)
```

---

## 疑難排解

**Q: `yt-dlp` 下載失敗**
```bash
pip install --upgrade yt-dlp    # yt-dlp 需要定期更新以跟上 YouTube 變化
```

**Q: Demucs 記憶體不足 (OOM)**
```bash
# 改用 CPU 模式（較慢但記憶體需求較低）
python main.py <URL> --device cpu

# 或直接跳過音源分離
python main.py <URL> --skip-separation
```

**Q: 偵測到的音符太多/太少**
```bash
# 太多雜音 → 提高閾值
python main.py <URL> --onset-threshold 0.7 --frame-threshold 0.5

# 漏掉音符 → 降低閾值
python main.py <URL> --onset-threshold 0.3 --frame-threshold 0.2
```

**Q: Tab 指法不合理**

目前的指法分配使用貪心 + 回溯法，對於複雜的指彈編曲可能不夠完美。
可以考慮的改進方向：用 MIDI 檔案匯入 Guitar Pro 等專業軟體做後續編輯。

---

## 實驗結果與現狀（2026-03-28）

> **本專案目前為實驗性質，轉譜準確度尚未達到實用水準。**

### 測試結果

使用兩首有標準答案的曲目進行 benchmark：
- *Sunset* (Nathania Jualim) — 指彈，標準調弦
- *River Flows in You* (Sungha Jung) — 指彈，降半音 + Capo 5

| 指標 | 結果 |
|------|------|
| TAB string+fret 完全匹配 | **~9%** |
| TAB fret 值匹配（不計弦，±1） | **~77%** |
| 和弦根音準確率 | **~23%** |
| 和弦完全匹配 | **~0%** |

### 問題分析

1. **音高偵測不夠準確**：無論是 Basic Pitch（Spotify）還是 SynthTab（基於 TabCNN），在 YouTube 壓縮音訊上的音高偵測準確率約 77%。這些模型都是在乾淨的錄音室 DI 錄音上訓練的（GuitarSet = 3.2 小時），對 YouTube 音訊存在嚴重的 domain gap。

2. **弦分配是核心瓶頸**：即使偵測到了正確的音高，同一個音可以在 3-4 條不同的弦上彈出。模型和啟發式演算法都無法準確判斷吉他手實際使用的弦位。

3. **和弦辨識不可靠**：chroma 分析在指彈吉他上表現很差（因為同時有低音、中聲部、高音旋律，chroma 特徵很混亂）。基於音符的和弦推斷也因為音高偵測錯誤而連鎖失敗。

4. **錯誤逐層放大**：音高偵測 → 弦分配 → 和弦推斷，每一層的錯誤都往下傳播並放大。

### 嘗試過的方案

| 方案 | 結果 |
|------|------|
| Basic Pitch (Spotify) + 啟發式弦分配 | fret 匹配 ~40%，弦分配幾乎全錯 |
| SynthTab (TabCNN, 吉他專用模型) | fret 匹配 ~77%，弦分配仍不準確 |
| 泛音偵測 (spectral flatness + peak count) | 泛音版曲目可用，一般曲目 false positive 已降至接近 0 |
| 和弦辨識改良 (長窗口 + smoothing + 合併) | 數量合理了，但內容仍不準確 |

### 要達到實用水準（~85% 準確度）可能需要

- 大量 YouTube 音訊 + 對應 TAB 的配對資料做 fine-tuning
- 商業 API（如 Klangio Guitar2Tabs，宣稱 ~85%）
- 視覺輔助（從影片畫面辨識指板手指位置）
- 或等待更強的開源吉他轉譜模型出現

---

## 技術限制

```
  目前限制                          可能的改進
  ─────────────────────────────────────────────────────
  音高偵測準確度不足                  用 YouTube 音訊 fine-tune 模型
  弦分配幾乎全錯                     視覺輔助 / Fretting-Transformer
  和弦辨識不可靠                     訓練專用 CNN/Transformer 模型
  無法偵測特殊技巧                   加入推弦/滑音/槌弦偵測
  (推弦、滑音、泛音等)
  指法分配不一定最優                 改用動態規劃或 beam search
```

---

## 相關技術與工具

| 技術 | 專案 | 用途 |
|------|------|------|
| 音源分離 | [Demucs](https://github.com/facebookresearch/demucs) (Meta) | 將混合音訊拆成個別樂器 |
| 音高偵測 | [Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify) | 音訊轉 MIDI |
| 音訊分析 | [librosa](https://librosa.org/) | Chroma 特徵、頻譜分析 |
| 音訊下載 | [yt-dlp](https://github.com/yt-dlp/yt-dlp) | YouTube 音訊擷取 |
| MIDI 處理 | [pretty-midi](https://github.com/craffel/pretty-midi) | MIDI 檔案讀寫 |
