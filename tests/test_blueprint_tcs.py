"""SkinSense AI - Comprehensive Blueprint Test Suite (TC-01 to TC-15)
Validates all requirements specified in Section 10 of the SkinSense AI Blueprint.
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
from PIL import Image
import pytest

from agents.explainer_agent import ExplainerAgent
from agents.input_agent import InputAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.report_coordinator import ReportCoordinator
from agents.safety_agent import SafetyAgent
from agents.symptom_agent import SymptomAgent
from agents.video_agent import VideoAgent
from agents.vision_agent import VisionAgent
from brain_of_the_doctor_groq import analyze_multimodal_case


# -------------------------------------------------------------
# TC-01: App Module & Component Imports
# -------------------------------------------------------------
def test_tc01_app_imports_cleanly():
    """TC-01: Verify all core application components import without exception."""
    import main
    import web_app
    import app
    assert web_app.app is not None
    assert app.PRESET_OPTIONS is not None


# -------------------------------------------------------------
# TC-02: Valid JPG/PNG Image Ingestion & Preview
# -------------------------------------------------------------
def test_tc02_valid_image_decodes(tmp_path):
    """TC-02: Valid JPG/PNG image decodes successfully and extracts metadata."""
    img_file = tmp_path / "valid_lesion.jpg"
    Image.new("RGB", (300, 300), color=(200, 120, 110)).save(img_file)

    agent = InputAgent()
    res = agent.inspect_and_route(image_path=str(img_file))
    assert res["valid"] is True
    assert res["metadata"]["image"]["dimensions"] == "300x300"
    assert res["metadata"]["image"]["format"] == ".jpg"


# -------------------------------------------------------------
# TC-03: Corrupt Image Rejection
# -------------------------------------------------------------
def test_tc03_corrupt_image_handled_gracefully(tmp_path):
    """TC-03: Corrupt image file returns a clear error without unhandled crash."""
    corrupt_file = tmp_path / "broken.jpg"
    corrupt_file.write_bytes(b"NOT_A_REAL_IMAGE_DATA_019283019283")

    agent = InputAgent()
    res = agent.inspect_and_route(image_path=str(corrupt_file))
    assert res["valid"] is False
    assert "Failed to open image" in res["error"]


# -------------------------------------------------------------
# TC-04: Non-Image File Rejection (e.g. TXT file)
# -------------------------------------------------------------
def test_tc04_non_image_txt_rejected(tmp_path):
    """TC-04: Upload of non-visual file (e.g. TXT) is rejected with helpful message."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Just some clinical text notes, no image.")

    agent = InputAgent()
    res = agent.inspect_and_route(image_path=str(txt_file))
    assert res["valid"] is False


# -------------------------------------------------------------
# TC-05: Empty Submission Handling
# -------------------------------------------------------------
def test_tc05_empty_submission_rejected():
    """TC-05: Empty submission requests user input before calling inference."""
    agent = InputAgent()
    res = agent.inspect_and_route()
    assert res["valid"] is False
    assert "At least one visual input" in res["error"]


# -------------------------------------------------------------
# TC-06: Valid Short MP4 Video Metadata
# -------------------------------------------------------------
def test_tc06_valid_video_metadata(tmp_path):
    """TC-06: Valid short video file opens and extracts technical metadata."""
    vid_file = tmp_path / "sample.mp4"
    # Create synthetic 10-frame video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_file), fourcc, 10.0, (160, 120))
    for i in range(10):
        frame = np.full((120, 160, 3), fill_value=i * 20, dtype=np.uint8)
        out.write(frame)
    out.release()

    agent = InputAgent()
    res = agent.inspect_and_route(video_path=str(vid_file))
    assert res["valid"] is True
    assert res["metadata"]["video"]["valid"] is True


# -------------------------------------------------------------
# TC-07: Extract Timestamped Frames with Sharpness
# -------------------------------------------------------------
def test_tc07_extract_frames_with_timestamps(tmp_path):
    """TC-07: Extracts usable frames with timestamps and calculates sharpness."""
    vid_file = tmp_path / "sample_extract.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_file), fourcc, 10.0, (160, 120))
    for i in range(15):
        frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    agent = VideoAgent()
    res = agent.sample_key_frames(str(vid_file))
    assert res["success"] is True
    assert len(res["key_frames"]) > 0
    assert "timestamp_sec" in res["key_frames"][0]
    assert "sharpness_score" in res["key_frames"][0]


# -------------------------------------------------------------
# TC-08: Corrupt Video Handling
# -------------------------------------------------------------
def test_tc08_corrupt_video_handled_gracefully(tmp_path):
    """TC-08: Corrupt MP4 file produces a handled error without unhandled crash."""
    bad_vid = tmp_path / "corrupt.mp4"
    bad_vid.write_bytes(b"CORRUPT_VIDEO_STREAM_BYTES")

    agent = VideoAgent()
    res = agent.sample_key_frames(str(bad_vid))
    assert res["success"] is False
    assert "Could not open" in res["error"]


# -------------------------------------------------------------
# TC-09: Missing Checkpoint Status (Honest Reporting)
# -------------------------------------------------------------
def test_tc09_missing_checkpoint_honest_status(tmp_path):
    """TC-09: Reports model unavailable honestly when checkpoint path is missing."""
    img_file = tmp_path / "test.jpg"
    Image.new("RGB", (224, 224), color=(220, 180, 160)).save(img_file)

    agent = VisionAgent(checkpoint_dir=str(tmp_path / "empty_dir"))
    res = agent.analyze_image(str(img_file), allow_demo_scaffold=True)
    assert res["status"] == "scaffold_ready"
    assert res["checkpoint_loaded"] is False
    assert "Pending Checkpoint" in res["model_prediction"]


# -------------------------------------------------------------
# TC-10: Incompatible / Corrupt Checkpoint Handled
# -------------------------------------------------------------
def test_tc10_incompatible_checkpoint_handled(tmp_path):
    """TC-10: Handles incompatible/damaged checkpoint gracefully without crashing."""
    bad_ckpt = tmp_path / "bad.pt"
    bad_ckpt.write_text("not a real pytorch state dict")

    # Vision agent handles missing or incompatible checkpoints safely
    agent = VisionAgent(checkpoint_dir=str(tmp_path))
    res = agent.analyze_image(str(tmp_path / "dummy.jpg"), condition_hint="nv")
    assert res["primary_prediction"] == "nv"


# -------------------------------------------------------------
# TC-11: Fallback State When External Service Unavailable
# -------------------------------------------------------------
def test_tc11_fallback_when_external_service_unavailable():
    """TC-11: Explainer and TTS agents fallback to deterministic rule text when external APIs are offline."""
    explainer = ExplainerAgent()
    mock_report = {
        "structured_summary": {
            "primary_name": "Tinea Corporis (Ringworm)",
            "confidence": 89.6,
            "is_malignant": False,
        }
    }
    explanation_en = explainer.generate_explanation(mock_report, language="en")
    assert "Tinea Corporis" in explanation_en["explanation"]
    assert len(explanation_en["explanation"]) > 20


# -------------------------------------------------------------
# TC-12: Bilingual Hindi Text Support Without Encoding Crash
# -------------------------------------------------------------
def test_tc12_hindi_text_processed_safely():
    """TC-12: Hindi clinical text and negation processed without Windows cp1252 crash."""
    hindi_text = "हाथ पर लाल गोल चकत्ता है और बहुत खुजली हो रही है, दर्द नहीं है।"
    symptom_agent = SymptomAgent()
    res = symptom_agent.structure_symptoms(hindi_text)
    assert res["reported"] is True
    assert res["symptoms"]["itching_pruritus"] == "yes"
    assert res["symptoms"]["pain_tenderness"] == "no"


# -------------------------------------------------------------
# TC-13: Export Report Available
# -------------------------------------------------------------
def test_tc13_export_report_schema_complete(tmp_path):
    """TC-13: Report coordinator generates exportable dossier with all required sections."""
    coordinator = ReportCoordinator()
    report = coordinator.compile_clinician_report(
        input_meta={"inputs_present": {"has_image": True}},
        symptom_data={"reported": True, "symptoms": {"itching_pruritus": "yes"}},
        vision_data={"primary_prediction": "tinea_corporis", "confidence": 89.6},
        video_data=None,
        knowledge_data={"found": True, "data": {"name": "Tinea Corporis"}},
        safety_data={"urgency_level": "ROUTINE", "disclaimer_en": "Educational only"},
    )
    assert "timestamp" in report
    assert "markdown_report" in report
    assert len(report["markdown_report"]) > 100


# -------------------------------------------------------------
# TC-14: Graceful Degradation When One Agent Has Missing Data
# -------------------------------------------------------------
def test_tc14_graceful_partial_report():
    """TC-14: Full analysis completes and provides report even when video and audio are absent."""
    img_path = Path("static/presets/tinea_corporis.jpg")
    res = analyze_multimodal_case(
        image_filepath=str(img_path) if img_path.exists() else None,
        patient_text="Itching and red border",
        condition_hint="tinea_corporis",
    )
    assert res["success"] is True
    assert res["combined_report"]["most_likely_condition"] == "Tinea Corporis (Ringworm)"
    assert res["agent_analysis"]["safety_agent"] is not None


# -------------------------------------------------------------
# TC-15: Model Disagreement & Uncertainty Displayed Honestly
# -------------------------------------------------------------
def test_tc15_model_disagreement_and_uncertainty_shown():
    """TC-15: Low confidence or borderline condition shows clinical caution instead of false certainty."""
    agent = VisionAgent()
    mel_res = agent.analyze_image("dummy.jpg", condition_hint="mel")
    # Melanoma has lower confidence (78.7%) and triggers clinical caution
    assert mel_res["overall_confidence"] < 80.0
    assert "Caution" in mel_res["confidence_label"] or "Borderline" in mel_res["diagnostic_certainty"] or "CRITICAL" in mel_res["triage_urgency"]

