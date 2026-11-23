"""SkinSense AI - Automated Unit Tests for Multi-Agent System
Verifies Input Agent, Video Agent, Vision Agent, Symptom Agent,
Knowledge Agent, Safety Agent, Report Coordinator, and Explainer Agent.
"""

from pathlib import Path
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


def test_input_agent_validation(tmp_path):
    agent = InputAgent()

    # Create dummy image
    img_file = tmp_path / "test.jpg"
    Image.new("RGB", (224, 224), color=(200, 150, 120)).save(img_file)

    res = agent.inspect_and_route(image_path=str(img_file), text_symptoms="Slight itching")
    assert res["valid"] is True
    assert res["inputs_present"]["has_image"] is True
    assert res["inputs_present"]["has_text"] is True
    assert res["inputs_present"]["has_video"] is False


def test_symptom_agent_strict_grounding():
    agent = SymptomAgent()

    # English test
    text_en = "This mole has been growing over the last 2 weeks and has started bleeding."
    res_en = agent.structure_symptoms(text_en)
    assert res_en["reported"] is True
    assert res_en["symptoms"]["bleeding_oozing"] == "yes"
    assert res_en["symptoms"]["evolution_growth"] == "yes"
    assert res_en["symptoms"]["itching_pruritus"] == "unknown"  # Never hallucinated!
    assert res_en["symptoms"]["duration"] == "2 weeks"

    # Hindi test
    text_hi = "चेहरे पर यह तिल 1 महीने से है और इसमें काफी खुजली हो रही है लेकिन कोई दर्द नहीं है।"
    res_hi = agent.structure_symptoms(text_hi)
    assert res_hi["reported"] is True
    assert res_hi["symptoms"]["itching_pruritus"] == "yes"
    assert res_hi["symptoms"]["pain_tenderness"] == "no"
    assert res_hi["symptoms"]["bleeding_oozing"] == "unknown"


def test_knowledge_agent_retrieval():
    agent = KnowledgeAgent()
    res = agent.retrieve_condition_info("mel")
    assert res["found"] is True
    assert "Malignant Neoplasm" in res["data"]["category"]
    assert len(res["data"]["citations"]) > 0
    assert any("ISIC" in c["source"] for c in res["data"]["citations"])


def test_safety_agent_red_flags():
    agent = SafetyAgent()
    symptoms = {
        "symptoms": {
            "bleeding_oozing": "yes",
            "evolution_growth": "yes",
        }
    }
    vision_scaffold = {"status": "scaffold_ready"}

    triage = agent.evaluate_safety_and_triage(symptoms, vision_scaffold)
    assert triage["has_red_flags"] is True
    assert len(triage["red_flags"]) >= 2
    assert "Melanoma" not in triage["urgency_level"] or "MODERATE" in triage["urgency_level"]
    assert "DISCLAIMER" in triage["disclaimer_en"].upper()


def test_vision_agent_honest_scaffold(tmp_path):
    img_file = tmp_path / "test.jpg"
    Image.new("RGB", (224, 224), color=(220, 180, 160)).save(img_file)

    agent = VisionAgent(checkpoint_dir=str(tmp_path / "non_existent_checkpoints"))
    res = agent.analyze_image(str(img_file), allow_demo_scaffold=True)
    assert res["status"] == "scaffold_ready"
    assert res["checkpoint_loaded"] is False
    assert "Pending Checkpoint" in res["model_prediction"]


def test_video_agent_sharpness():
    agent = VideoAgent()
    # Blurry image (flat gray)
    flat_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    sharpness_flat = agent.calculate_sharpness(flat_frame)
    assert sharpness_flat == 0.0

    # High texture frame
    textured_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    sharpness_textured = agent.calculate_sharpness(textured_frame)
    assert sharpness_textured > sharpness_flat


def test_explainer_agent_bilingual():
    agent = ExplainerAgent()
    mock_report = {
        "structured_summary": {
            "primary_name": "Melanocytic Nevus (Common Mole)",
            "triage_urgency": "Routine",
            "red_flags": [],
        }
    }
    exp_en = agent.generate_explanation(mock_report, language="en")
    assert "Melanocytic Nevus" in exp_en["explanation"]
    assert len(exp_en["tts_speech_text"]) > 20

    exp_hi = agent.generate_explanation(mock_report, language="hi")
    assert len(exp_hi["explanation"]) > 20
    assert "नमस्ते" in exp_hi["explanation"] or "Melanocytic" in exp_hi["explanation"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])

