"""SkinSense AI - Video Frame Sampling & Multi-Frame Consensus Agent
Extracts evenly spaced frames from skin inspection video, evaluates sharpness
using Laplacian variance to discard blurry frames, and aggregates multi-frame predictions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image


class VideoAgent:
    """Agent that samples high-quality frames from skin videos and aggregates multi-frame consensus."""

    def __init__(self, target_frames: int = 4, sharpness_threshold: float = 40.0):
        self.target_frames = target_frames
        self.sharpness_threshold = sharpness_threshold

    def calculate_sharpness(self, frame_bgr: np.ndarray) -> float:
        """Computes frame sharpness using the variance of the Laplacian filter."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def sample_key_frames(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Samples frames from video and filters out blurry or non-informative frames."""
        vid_p = Path(video_path)
        if not vid_p.exists():
            return {"success": False, "error": f"Video file not found: {video_path}"}

        cap = cv2.VideoCapture(str(vid_p))
        if not cap.isOpened():
            return {"success": False, "error": "Could not open video file."}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / max(fps, 1.0)

        if total_frames <= 0:
            cap.release()
            return {"success": False, "error": "Video contains 0 frames."}

        # Calculate sample indices evenly spaced
        sample_indices = np.linspace(
            start=int(total_frames * 0.1),
            stop=int(total_frames * 0.9),
            num=min(self.target_frames * 2, total_frames),
            dtype=int,
        )

        extracted_candidates = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            sharpness = self.calculate_sharpness(frame)
            timestamp = idx / max(fps, 1.0)
            extracted_candidates.append({
                "frame_index": int(idx),
                "timestamp_sec": round(float(timestamp), 2),
                "sharpness": round(sharpness, 1),
                "frame_bgr": frame,
            })

        cap.release()

        # Sort by sharpness descending and select the top sharpest frames
        sharp_candidates = [c for c in extracted_candidates if c["sharpness"] >= self.sharpness_threshold]
        if not sharp_candidates:
            # Fall back to best available candidates if all fall below strict threshold
            sharp_candidates = sorted(extracted_candidates, key=lambda x: x["sharpness"], reverse=True)[:self.target_frames]
        else:
            sharp_candidates = sorted(sharp_candidates, key=lambda x: x["sharpness"], reverse=True)[:self.target_frames]

        # Save selected frames to disk for vision agent consumption
        save_dir = Path(output_dir) if output_dir else vid_p.parent / "extracted_frames"
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_frames = []
        for i, cand in enumerate(sharp_candidates):
            frame_filename = save_dir / f"frame_{i+1}_t{cand['timestamp_sec']}s.jpg"
            cv2.imwrite(str(frame_filename), cand["frame_bgr"])
            saved_frames.append({
                "frame_num": i + 1,
                "timestamp_sec": cand["timestamp_sec"],
                "sharpness_score": cand["sharpness"],
                "file_path": str(frame_filename.resolve()),
            })

        return {
            "success": True,
            "video_metadata": {
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration_seconds": round(duration_sec, 2),
            },
            "frames_sampled": len(saved_frames),
            "key_frames": saved_frames,
        }

    def aggregate_multi_frame_predictions(self, frame_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combines predictions from multiple video frames into a consensus output."""
        if not frame_predictions:
            return {
                "consensus_available": False,
                "consensus_prediction": None,
                "frame_agreement_rate": 0.0,
                "frame_count": 0,
            }

        top_classes = [fp.get("top_prediction") for fp in frame_predictions if fp.get("top_prediction")]
        if not top_classes:
            return {
                "consensus_available": False,
                "consensus_prediction": None,
                "frame_agreement_rate": 0.0,
                "frame_count": len(frame_predictions),
            }

        from collections import Counter
        counts = Counter(top_classes)
        majority_class, count = counts.most_common(1)[0]
        agreement_rate = count / len(top_classes)

        return {
            "consensus_available": True,
            "consensus_prediction": majority_class,
            "frame_agreement_rate": round(agreement_rate * 100, 1),
            "frame_count": len(top_classes),
            "class_distribution": dict(counts),
            "stability_flag": "High Consistency" if agreement_rate >= 0.75 else "Variable Across Angles (Examine closely)",
        }

