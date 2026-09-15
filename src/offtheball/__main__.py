import argparse
import json
from pathlib import Path
from . import ROOT


def main():
    parser = argparse.ArgumentParser(description="OffTheBall")
    sub = parser.add_subparsers(dest="command")
    analyze = sub.add_parser("analyze", help="영상 사람 탐지")
    analyze.add_argument("video", type=Path)
    sub.add_parser("gui", help="프로그램 열기")
    args = parser.parse_args()
    if args.command == "analyze":
        from .detection import PersonDetector
        from .pipeline import analyze_video
        output, summary = analyze_video(args.video, PersonDetector())
        print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))
    else:
        from .gui import launch
        launch()


if __name__ == "__main__":
    main()
