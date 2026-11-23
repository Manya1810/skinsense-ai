"""SkinSense AI - Input Router Agent
Validates multimodal inputs (image, video, patient voice, symptom text),
extracts media metadata, and routes to downstream processing pipelines.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image


class InputAgent:
    """Agent responsible for inspecting, validating, and routing multimodal inputs."""

    def __init__(self):
        self.supported_image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.supported_video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        self.supported_audio_exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

    def inspect_and_route(
        self,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        text_symptoms: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validates inputs and extracts technical metadata."""
        inputs_present = {
            "has_image": bool(image_path and Path(image_path).exists()),
            "has_video": bool(video_path and Path(video_path).exists()),
            "has_audio": bool(audio_path and Path(audio_path).exists()),
            "has_text": bool(text_symptoms and text_symptoms.strip()),
        }

        if not (inputs_present["has_image"] or inputs_present["has_video"]):
            return {
                "valid": False,
                "error": "At least one visual input (skin image or video) is required for analysis.",
                "inputs_present": inputs_present,
                "metadata": {},
            }

        metadata = {}

        # Inspect Image
        if inputs_present["has_image"]:
            try:
                img_p = Path(image_path)
                with Image.open(img_p) as img:
                    w, h = img.size
                    mode = img.mode
                metadata["image"] = {
                    "path": str(img_p.resolve()),
                    "dimensions": f"{w}x{h}",
                    "format": img_p.suffix.lower(),
                    "valid": True,
                }
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Failed to open image: {str(e)}",
                    "inputs_present": inputs_present,
                    "metadata": {},
                }

        # Inspect Video
        if inputs_present["has_video"]:
            vid_p = Path(video_path)
            metadata["video"] = {
                "path": str(vid_p.resolve()),
                "format": vid_p.suffix.lower(),
                "file_size_bytes": vid_p.stat().st_size,
                "valid": True,
            }

        # Inspect Audio
        if inputs_present["has_audio"]:
            aud_p = Path(audio_path)
            metadata["audio"] = {
                "path": str(aud_p.resolve()),
                "format": aud_p.suffix.lower(),
                "valid": True,
            }

        # Inspect Text
        if inputs_present["has_text"]:
            metadata["text"] = {
                "char_length": len(text_symptoms.strip()),
                "word_count": len(text_symptoms.strip().split()),
            }

        return {
            "valid": True,
            "inputs_present": inputs_present,
            "metadata": metadata,
            "pipeline_route": {
                "run_video_sampling": inputs_present["has_video"],
                "run_image_vision": inputs_present["has_image"],
                "run_voice_transcription": inputs_present["has_audio"],
                "run_symptom_structuring": inputs_present["has_text"] or inputs_present["has_audio"],
                "run_knowledge_retrieval": True,
                "run_safety_guardrails": True,
                "run_report_coordinator": True,
            },
        }

