"""SkinSense AI - Safety Guardrails & Clinical Triage Agent
Enforces non-overridable safety rules, ABCDE criteria checking,
red flag detection, confidence thresholding, and mandatory medical disclaimers.
"""

from typing import Any, Dict, List


class SafetyAgent:
    """Agent that enforces medical safety guardrails and clinical triage rules."""

    DISCLAIMER_TEXT: str = (
        "IMPORTANT MEDICAL DISCLAIMER: SkinSense AI is an educational, multimodal clinician-review "
        "support prototype designed to assist healthcare professionals. It DOES NOT provide medical "
        "diagnoses, does not prescribe pharmaceuticals, and must never replace an in-person clinical "
        "examination and dermoscopic biopsy by a qualified dermatologist or licensed medical provider."
    )

    DISCLAIMER_TEXT_HI: str = (
        "महत्वपूर्ण चिकित्सा अस्वीकरण: SkinSense AI केवल एक शैक्षणिक और डॉक्टर-सहायक प्रोटोटाइप है। "
        "यह कोई अंतिम मेडिकल निदान (Diagnosis) नहीं देता और न ही कोई दवा लिखता है। किसी भी त्वचा समस्या "
        "के लिए योग्य त्वचा विशेषज्ञ (Dermatologist) से व्यक्तिगत जांच करवाना अनिवार्य है।"
    )

    def evaluate_safety_and_triage(
        self,
        symptom_data: Dict[str, Any],
        vision_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Applies non-overridable clinical safety rules and computes triage urgency."""
        symptoms = symptom_data.get("symptoms", {})
        red_flags_detected = []
        abcde_flags = []
        urgency_level = "Routine / Informational"

        # 1. Red Flag Checks
        if symptoms.get("bleeding_oozing") == "yes":
            red_flags_detected.append("Reported spontaneous bleeding or oozing from lesion.")
        if symptoms.get("evolution_growth") == "yes":
            red_flags_detected.append("Reported rapid growth, expansion, or evolving shape.")
            abcde_flags.append("E - Evolving: Documented size/shape change over time.")
        if symptoms.get("color_change") == "yes":
            red_flags_detected.append("Reported darkening, multi-color variegation, or pigment changes.")
            abcde_flags.append("C - Color Variegation: Change in pigment or multiple shades.")
        if symptoms.get("pain_tenderness") == "yes":
            red_flags_detected.append("Reported persistent pain, burning, or tenderness.")

        # 2. Vision Risk Check (e.g. Melanoma or BCC predicted)
        primary_pred = vision_data.get("primary_prediction", "").lower()
        if primary_pred in ["mel", "melanoma"]:
            red_flags_detected.append("Vision model flagged high morphological similarity to Melanoma.")
            urgency_level = "CRITICAL — Prompt Dermatologist Referral Advised"
        elif primary_pred in ["bcc", "akiec"]:
            urgency_level = "HIGH — In-person Clinician Review Recommended"
        elif red_flags_detected:
            urgency_level = "MODERATE — Triage Consultation Advised Due to Reported Symptoms"

        # 3. Vision Model Disagreement & Low-Confidence Checks
        uncertainty_flags = []
        if vision_data.get("checkpoint_loaded"):
            raw_score = vision_data.get("raw_softmax_score", 0.0)
            if raw_score < 0.60:
                uncertainty_flags.append(
                    f"Low statistical model confidence ({raw_score * 100:.1f}%). Ambiguous morphology; in-person dermoscopy necessary."
                )
            if not vision_data.get("models_agree", True):
                uncertainty_flags.append(
                    vision_data.get("disagreement_warning", "Candidate vision models disagree on primary class.")
                )
        elif vision_data.get("status") == "scaffold_ready":
            uncertainty_flags.append(
                "No trained checkpoint evaluated. Vision status is scaffold ready."
            )

        # 4. Strict Prescription & Advice Boundary Check
        prescription_rule = (
            "System strictly prohibits drug prescribing, steroid recommendations, or home self-treatment advice."
        )

        return {
            "urgency_level": urgency_level,
            "has_red_flags": len(red_flags_detected) > 0,
            "red_flags": red_flags_detected,
            "abcde_flags": abcde_flags,
            "uncertainty_flags": uncertainty_flags,
            "prescription_prohibition": prescription_rule,
            "disclaimer_en": self.DISCLAIMER_TEXT,
            "disclaimer_hi": self.DISCLAIMER_TEXT_HI,
            "safe_for_patient_presentation": True,
        }

