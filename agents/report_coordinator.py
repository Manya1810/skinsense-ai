"""SkinSense AI - Report Coordinator Agent
Synthesizes inputs from all agents (Input Router, Video Agent, Vision Agent,
Symptom Agent, Knowledge Agent, Safety Agent) into a unified Clinician-Review Report.
Preserves model disagreements, marks unavailable components, and enforces medical disclaimers.
"""

from datetime import datetime
from typing import Any, Dict, Optional


class ReportCoordinator:
    """Agent that coordinates multi-agent findings into a clinician-review dossier."""

    def compile_clinician_report(
        self,
        input_meta: Dict[str, Any],
        symptom_data: Dict[str, Any],
        vision_data: Dict[str, Any],
        video_data: Optional[Dict[str, Any]],
        knowledge_data: Dict[str, Any],
        safety_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combines all agent outputs into a standardized report structure."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Compile Markdown Text
        md_lines = []
        md_lines.append("# SkinSense AI — Multimodal Clinician Review Dossier")
        md_lines.append(f"**Generated:** {timestamp} | **Triage Status:** `{safety_data.get('urgency_level', 'Routine')}`\n")
        md_lines.append("> **IMPORTANT:** Educational prototype for clinician review only. Not a definitive medical diagnosis or prescription.\n")

        # Section 1: Inputs & Modalities
        md_lines.append("## 1. Input Modality Verification")
        inputs_p = input_meta.get("inputs_present", {})
        md_lines.append(f"- **Skin Image:** {'Provided' if inputs_p.get('has_image') else 'Not Provided'}")
        if input_meta.get("metadata", {}).get("image"):
            md_lines.append(f"  - Dimensions: `{input_meta['metadata']['image'].get('dimensions')}`")

        md_lines.append(f"- **Skin Video:** {'Provided' if inputs_p.get('has_video') else 'Not Provided'}")
        if video_data and video_data.get("frames_sampled"):
            md_lines.append(f"  - Key Frames Sampled: `{video_data.get('frames_sampled')}` frames evaluated for sharpness.")
            if "stability_flag" in video_data:
                md_lines.append(f"  - Multi-angle Frame Stability: `{video_data['stability_flag']}` ({video_data.get('frame_agreement_rate', 0)}% agreement)")

        md_lines.append(f"- **Patient Voice / Audio:** {'Provided' if inputs_p.get('has_audio') else 'Not Provided'}")
        md_lines.append(f"- **Reported Text:** {'Provided' if inputs_p.get('has_text') else 'Not Provided'}\n")

        # Section 2: Patient-Reported Symptoms (Explicit Unknowns)
        md_lines.append("## 2. Patient-Reported Symptom Matrix")
        if symptom_data.get("reported"):
            md_lines.append(f"**Patient Description:** *\"{symptom_data.get('raw_text', '')}\"*\n")
            sym = symptom_data.get("symptoms", {})
            md_lines.append("| Clinical Parameter | Patient Report Status |")
            md_lines.append("|---|---|")
            md_lines.append(f"| Lesion Duration | `{sym.get('duration')}` |")
            md_lines.append(f"| Evolution / Rapid Growth | `{sym.get('evolution_growth')}` |")
            md_lines.append(f"| Spontaneous Bleeding / Oozing | `{sym.get('bleeding_oozing')}` |")
            md_lines.append(f"| Pain / Tenderness | `{sym.get('pain_tenderness')}` |")
            md_lines.append(f"| Itching / Pruritus | `{sym.get('itching_pruritus')}` |")
            md_lines.append(f"| Color Variegation | `{sym.get('color_change')}` |")
            md_lines.append(f"| Sun / UV Exposure | `{sym.get('sun_exposure')}` |")
            md_lines.append(f"| Family History of Skin Cancer | `{sym.get('family_history')}` |")
            md_lines.append(f"| Reported Body Location | `{sym.get('anatomic_location')}` |")
        else:
            md_lines.append("*No patient symptoms verbally or textually provided.*")
        md_lines.append("")

        # Section 3: Vision Model Analysis (Honest Status & Probabilities)
        md_lines.append("## 3. Vision Model Analysis")
        if vision_data.get("checkpoint_loaded"):
            md_lines.append(f"- **Evaluated Architectures:** `{', '.join(vision_data.get('models_evaluated', []))}`")
            md_lines.append(f"- **Primary Prediction:** **{vision_data.get('primary_prediction_name')}** (`{vision_data.get('primary_prediction')}`)")
            md_lines.append(f"- **Raw Softmax Ensemble Score:** `{vision_data.get('raw_softmax_score')} (Note: statistical score, not clinical probability)`")

            md_lines.append("\n### Top Differential Probabilities:")
            for p in vision_data.get("top_3_ensemble", []):
                md_lines.append(f"- **{p['class_name']}**: `{p['percentage']}`")

            if vision_data.get("disagreement_warning"):
                md_lines.append(f"\n> [!WARNING] **Model Disagreement:** {vision_data.get('disagreement_warning')}")
        else:
            md_lines.append(f"- **Status:** `{vision_data.get('status', 'scaffold_ready')}`")
            md_lines.append(f"- **Prediction:** `{vision_data.get('model_prediction', 'Pending Checkpoint')}`")
            md_lines.append(f"- **Reason:** {vision_data.get('reason', 'Training in progress / checkpoint not yet loaded.')}")
            if "recommended_action" in vision_data:
                md_lines.append(f"- **Next Step:** {vision_data['recommended_action']}")
        md_lines.append("")

        # Section 4: Knowledge Base & Medical Citations
        md_lines.append("## 4. Curated Reference & Citations")
        kb_info = knowledge_data.get("data", {})
        md_lines.append(f"- **Condition:** **{kb_info.get('full_name')}** ({kb_info.get('category', 'Dermatology')})")
        md_lines.append(f"- **Clinical Definition:** {kb_info.get('description')}")
        if kb_info.get("differential_diagnosis"):
            md_lines.append(f"- **Differential Considerations:** {', '.join(kb_info.get('differential_diagnosis', []))}")
        if kb_info.get("guideline_recommendations"):
            md_lines.append(f"- **Clinical Practice Guidelines:** {kb_info.get('guideline_recommendations')}")

        citations = kb_info.get("citations", [])
        if citations:
            md_lines.append("\n**Authoritative Citations:**")
            for c in citations:
                md_lines.append(f"- [{c.get('source')} ({c.get('year')}) — {c.get('title')}]({c.get('url')})")
        md_lines.append("")

        # Section 5: Safety Guardrails & Triage
        md_lines.append("## 5. Clinical Safety & Triage Guardrails")
        md_lines.append(f"- **Triage Urgency:** `{safety_data.get('urgency_level')}`")
        if safety_data.get("red_flags"):
            md_lines.append("\n**Urgent Red Flags Detected:**")
            for rf in safety_data["red_flags"]:
                md_lines.append(f"- ⚠️ {rf}")

        if safety_data.get("abcde_flags"):
            md_lines.append("\n**ABCDE Criteria Triggers:**")
            for af in safety_data["abcde_flags"]:
                md_lines.append(f"- {af}")

        if safety_data.get("uncertainty_flags"):
            md_lines.append("\n**Model Uncertainty Notes:**")
            for uf in safety_data["uncertainty_flags"]:
                md_lines.append(f"- ℹ️ {uf}")

        md_lines.append(f"\n> **Medical Disclaimer:** {safety_data.get('disclaimer_en')}")

        markdown_report = "\n".join(md_lines)

        return {
            "timestamp": timestamp,
            "markdown_report": markdown_report,
            "structured_summary": {
                "triage_urgency": safety_data.get("urgency_level"),
                "primary_class": vision_data.get("primary_prediction"),
                "primary_name": vision_data.get("primary_prediction_name", kb_info.get("full_name")),
                "confidence_score": vision_data.get("raw_softmax_score"),
                "active_symptoms": symptom_data.get("active_symptoms", []),
                "red_flags": safety_data.get("red_flags", []),
                "models_agree": vision_data.get("models_agree", True),
            },
        }
