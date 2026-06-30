"""Video processing module for EcoSort AI.

Extracts key frames from uploaded video files and runs waste
classification on each frame.  Supports MP4, AVI, MOV, MKV, and WEBM.
"""

import os
import tempfile
import math
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_key_frames(
    video_path: str,
    max_frames: int = 8,
    min_interval_sec: float = 1.0,
) -> list[dict]:
    """Extract evenly-spaced key frames from a video file.

    Args:
        video_path: Path to the video on disk.
        max_frames: Maximum number of frames to extract.
        min_interval_sec: Minimum time gap (seconds) between frames.

    Returns:
        A list of dicts, each containing:
          - ``pil_image``: PIL Image (RGB)
          - ``timestamp_sec``: float — position in seconds
          - ``frame_index``: int — original frame number
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    # Sample frames across the video
    min_interval_frames = max(1, int(fps * min_interval_sec))
    sampled_frames = []

    for pos in range(0, total_frames, min_interval_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, bgr_frame = cap.read()
        if not ret:
            continue
            
        # Calculate detail/sharpness score using Variance of Laplacian
        # Frames with objects/texture will have a much higher score than blank backgrounds
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        sampled_frames.append({
            "pil_image": pil_image,
            "timestamp_sec": round(pos / fps, 2),
            "frame_index": pos,
            "score": score
        })

    cap.release()
    
    # Select the top `max_frames` with the highest detail score (objects present/in focus)
    sampled_frames.sort(key=lambda x: x["score"], reverse=True)
    best_frames = sampled_frames[:max_frames]
    
    # Re-sort chronologically to maintain the timeline
    best_frames.sort(key=lambda x: x["frame_index"])

    return best_frames


def get_video_metadata(video_path: str) -> dict:
    """Return metadata (resolution, fps, duration, codec) for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4))

    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "duration_str": _format_duration(duration_sec),
        "codec": codec.strip(),
        "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
    }


def save_uploaded_video(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temporary path and return it."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Convert seconds to a human-readable string like '1m 23s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs:02d}s"


def aggregate_frame_results(frame_results: list[list[dict]]) -> dict:
    """Aggregate classification results across multiple frames.

    Args:
        frame_results: A list where each element is a classify_image()
                       result list for one frame.

    Returns:
        A summary dict with:
          - ``dominant_category``: most frequent top-1 category
          - ``category_counts``: {category: count}
          - ``avg_confidence``: mean confidence of the dominant category
          - ``all_detections``: list of per-frame top-1 detections
          - ``category_color``: hex colour for the dominant category
          - ``bin_color``: bin recommendation
    """
    from app.classifier import CATEGORY_COLORS, BIN_COLORS

    category_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    confidence_sums: dict[str, float] = {}
    all_detections = []

    for frame_idx, results in enumerate(frame_results):
        if not results:
            continue
        top = results[0]
        cat = top["waste_category"]
        lbl = top["label"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        confidence_sums[cat] = confidence_sums.get(cat, 0.0) + top["confidence"]
        all_detections.append({
            "frame": frame_idx,
            "label": lbl,
            "waste_category": cat,
            "confidence": top["confidence"],
        })

    if not category_counts:
        return {
            "dominant_category": "Unknown",
            "dominant_label": "Unknown item",
            "category_counts": {},
            "label_counts": {},
            "avg_confidence": 0.0,
            "all_detections": [],
            "category_color": "#64748b",
            "bin_color": "Unknown",
        }

    dominant = max(category_counts, key=category_counts.get)
    dominant_lbl = max(label_counts, key=label_counts.get)
    avg_conf = confidence_sums[dominant] / category_counts[dominant]

    return {
        "dominant_category": dominant,
        "dominant_label": dominant_lbl.replace("_", " ").title(),
        "category_counts": category_counts,
        "label_counts": label_counts,
        "avg_confidence": round(avg_conf, 4),
        "all_detections": all_detections,
        "category_color": CATEGORY_COLORS.get(dominant, "#3B82F6"),
        "bin_color": BIN_COLORS.get(dominant, "Blue 🔵"),
    }
