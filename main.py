#!/usr/bin/env python3
"""
Guitar Tab Retriever - 從 YouTube 影片自動轉譜吉他六線譜

用法:
    python main.py <YouTube_URL>
    python main.py <YouTube_URL> --skip-separation
    python main.py <YouTube_URL> --tuning drop_d --output ./my_output
"""

import argparse
import sys

from guitar_transcriber.pipeline import transcribe
from guitar_transcriber.tab_generator import TUNINGS


def main():
    parser = argparse.ArgumentParser(
        description="從 YouTube 影片自動擷取吉他六線譜與和弦進行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本用法
  python main.py https://www.youtube.com/watch?v=XXXXXXX

  # 跳過音源分離（較快，但精確度較低）
  python main.py https://www.youtube.com/watch?v=XXXXXXX --skip-separation

  # 使用 Drop D 調弦
  python main.py https://www.youtube.com/watch?v=XXXXXXX --tuning drop_d

  # 使用更精確的 Demucs 模型（較慢）
  python main.py https://www.youtube.com/watch?v=XXXXXXX --demucs-model htdemucs_ft

  # 強制使用 CPU
  python main.py https://www.youtube.com/watch?v=XXXXXXX --device cpu
        """,
    )

    parser.add_argument(
        "url",
        help="YouTube 影片網址",
    )
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="輸出目錄（預設: ./output）",
    )
    parser.add_argument(
        "-t", "--tuning",
        default="standard",
        choices=list(TUNINGS.keys()),
        help="吉他調弦（預設: standard）",
    )
    parser.add_argument(
        "--skip-separation",
        action="store_true",
        help="跳過音源分離，直接分析混合音訊（較快但不精確）",
    )
    parser.add_argument(
        "--demucs-model",
        default="htdemucs",
        choices=["htdemucs", "htdemucs_ft", "mdx_extra"],
        help="Demucs 模型（預設: htdemucs）",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="運算裝置（預設: auto，自動偵測 GPU）",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="音符起始偵測閾值 0-1（預設: 0.5，越高越嚴格）",
    )
    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.3,
        help="音框偵測閾值 0-1（預設: 0.3，越高越嚴格）",
    )
    parser.add_argument(
        "--chord-method",
        default="both",
        choices=["audio", "midi", "both"],
        help="和弦辨識方式（預設: both）",
    )
    parser.add_argument(
        "--tab-density",
        type=float,
        default=8.0,
        help="Tab 譜面密度，每秒字元數（預設: 8.0）",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=80,
        help="Tab 每行寬度（預設: 80）",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不儲存中間檔案",
    )

    args = parser.parse_args()

    # 驗證 URL
    if "youtube.com" not in args.url and "youtu.be" not in args.url:
        print("警告: 網址看起來不像 YouTube 連結，仍然嘗試處理...")

    print("=" * 60)
    print("  Guitar Tab Retriever v0.1.0")
    print("  從 YouTube 影片自動擷取吉他六線譜")
    print("=" * 60)

    # 執行轉譜 pipeline
    result = transcribe(
        url=args.url,
        output_dir=args.output,
        tuning=args.tuning,
        skip_separation=args.skip_separation,
        demucs_model=args.demucs_model,
        device=args.device,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        chord_method=args.chord_method,
        chars_per_second=args.tab_density,
        line_width=args.line_width,
        save_intermediate=not args.no_save,
    )

    # 輸出結果
    print("\n")
    print(result.summary())

    if result.errors:
        print("\n注意: 有些步驟遇到了問題，結果可能不完整。")
        print("嘗試調整參數或使用 --skip-separation 來簡化流程。")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
