"""Timestamped video analysis with bounded UI delivery and recoverable outputs."""
import json
import time
import uuid
from pathlib import Path
from threading import Event
import cv2
import numpy as np
from . import ROOT
from .detection import annotate
from .analysis import AnalysisSession
from .inputs import VideoSource
from .quality import is_black_frame


def create_run_directory():
    path = ROOT / "outputs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    path.mkdir(parents=True, exist_ok=False)
    return path


def analyze_video(path, detector, *, stop=None, on_frame=None, output_dir=None,
                  team_classifier=None, calibration=None, pitch_polygon=None):
    stop = stop or Event()
    output_dir = Path(output_dir) if output_dir else create_run_directory()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "analysis.mp4"
    data_path = output_dir / "frames.jsonl"
    summary_path = output_dir / "summary.json"
    if any(p.exists() for p in (video_path, data_path, summary_path)):
        raise FileExistsError("출력 폴더에 분석 결과가 이미 있습니다.")
    writer = None
    source_size = None
    count = 0
    black_count = 0
    elapsed_samples = []
    started = time.perf_counter()
    status = "failed"
    error = None
    session = AnalysisSession(detector, team_classifier, calibration=calibration, pitch_polygon=pitch_polygon)
    try:
        with VideoSource(path) as source, data_path.open("x", encoding="utf-8") as records:
            fps = source.fps
            for frame in source:
                if stop.is_set():
                    status = "cancelled"
                    break
                tick = time.perf_counter()
                size = (frame.width, frame.height)
                if source_size is not None and size != source_size:
                    raise ValueError("영상 해상도가 바뀌었습니다. 결과 일치를 위해 분석을 중단합니다.")
                source_size = size
                black = is_black_frame(frame.pixels)
                if black:
                    black_count += 1
                    detections = session.process(frame, black=True)
                    rendered = frame.pixels.copy()
                    cv2.putText(rendered, "BLACK INPUT - CHECK VIDEO", (20, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, .8, (80, 180, 255), 2)
                else:
                    detections = session.process(frame)
                    rendered = annotate(frame.pixels, detections)
                # MPEG-4 requires even dimensions; pad rather than silently crop.
                encoded = cv2.copyMakeBorder(rendered, 0, frame.height % 2,
                                             0, frame.width % 2, cv2.BORDER_CONSTANT)
                if writer is None:
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (encoded.shape[1], encoded.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("결과 영상 파일을 열 수 없습니다.")
                writer.write(encoded)
                seconds = time.perf_counter() - tick
                elapsed_samples.append(seconds)
                row = {"frame_index": frame.index, "source_time_s": frame.timestamp_s,
                       "schema_version": 2,
                       "image_size": [frame.width, frame.height],
                       "encoded_image_size": [encoded.shape[1], encoded.shape[0]],
                       "detections": [d.to_dict() for d in detections],
                       "scene_id": session.scene_id,
                       "calibration_status": session.calibration_diagnostics()["status"],
                       "calibration": session.calibration_diagnostics(),
                       "processing_s": seconds,
                       "input_status": "black_frame" if black else "available"}
                records.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
                if on_frame:
                    on_frame(rendered, row, frame.pixels)
            else:
                if count == 0:
                    raise ValueError("읽을 수 있는 영상 프레임이 없습니다.")
                status = "completed"
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        if writer is not None:
            writer.release()
        elapsed = time.perf_counter() - started
        summary = {"schema_version": 2, "status": status, "error": error, "frames": count,
                   "black_frames": black_count,
                   "warnings": ["Black input frames present; player absence cannot be inferred."] if black_count else [],
                   "elapsed_s": elapsed, "processing_fps": count / elapsed if elapsed else 0,
                   "processing_p95_ms": float(np.percentile(elapsed_samples, 95) * 1000)
                   if elapsed_samples else None,
                   "source": str(Path(path).resolve()), "audio_included": False,
                   "scope": "ByteTrack person IDs, temporary team labels, and optional fixed-camera field coordinates; no identity claims"}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                           allow_nan=False), encoding="utf-8")
    return output_dir, summary
