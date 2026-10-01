import argparse
import json
from pathlib import Path
from . import ROOT


def main():
    parser = argparse.ArgumentParser(description="OffTheBall")
    sub = parser.add_subparsers(dest="command")
    analyze = sub.add_parser("analyze", help="영상 사람 탐지")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--calibration", type=Path,
                         help="고정 카메라 보정 JSON 경로")
    analyze.add_argument("--device", default="auto",
                         help="분석 장치: auto, cpu, 0 등 (기본값: auto)")
    analyze.add_argument("--output", type=Path,
                         help="결과를 저장할 새 폴더")
    analyze.add_argument("--image-size", type=int, default=960,
                         help="YOLO 입력 크기 (기본값: 960)")
    sub.add_parser("gui", help="프로그램 열기")
    args = parser.parse_args()
    if args.command == "analyze":
        from .detection import PersonDetector
        from .pipeline import analyze_video
        calibration = None
        if args.calibration:
            from .calibration import ManualCalibration
            calibration = ManualCalibration.load(args.calibration)
        device = int(args.device) if args.device.isdigit() else args.device
        detector = PersonDetector(device=device, image_size=args.image_size)
        output, summary = analyze_video(args.video, detector,
                                        calibration=calibration,
                                        output_dir=args.output)
        print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))
    else:
        from .gui import launch
        launch()


if __name__ == "__main__":
    main()
