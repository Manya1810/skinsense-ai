"""SkinSense AI - Multilingual Explainer Agent (English & Hindi)
Translates validated clinician findings into clear, empathetic, patient-friendly guidance.
Strictly grounded in validated agent outputs: does NOT hallucinate diagnoses or prescribe medicines.
Generates clean speech-ready text for audio synthesis.
"""

import os
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()


class ExplainerAgent:
    """Agent that translates validated clinical findings into patient-friendly English & Hindi."""

    def __init__(self):
        self.groq_api_key = os.environ.get("GROQ_API_KEY")

    def generate_explanation(
        self,
        report_data: Dict[str, Any],
        language: str = "en",
    ) -> Dict[str, str]:
        """Generates patient-friendly explanations in English and Hindi."""
        summary = report_data.get("structured_summary", {})
        triage = summary.get("triage_urgency", "Routine")
        primary_name = summary.get("primary_name", "Skin Finding")
        red_flags = summary.get("red_flags", [])

        # Offline / deterministic grounded generator (guaranteed factual & safe)
        if language == "hi" or language == "hindi":
            explanation = self._generate_hindi_template(primary_name, triage, red_flags)
        else:
            explanation = self._generate_english_template(primary_name, triage, red_flags)

        # Optional LLM polish if API key is present
        if self.groq_api_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_api_key)
                lang_instruction = "in simple Hindi" if "hi" in language.lower() else "in clear English"
                prompt = (
                    f"You are an empathetic medical communicator. Rephrase this skin consultation summary {lang_instruction} "
                    f"for the patient in 2 or 3 calm, conversational sentences. "
                    f"Do NOT diagnose definitively, do NOT prescribe medicines or remedies. "
                    f"Do NOT use markdown, asterisks, bullet points, or special characters because this will be spoken aloud.\n\n"
                    f"Condition: {primary_name}\nTriage: {triage}\nKey notes: {', '.join(red_flags) if red_flags else 'No emergency signs'}"
                )
                response = client.chat.completions.create(
                    model=os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
                    max_completion_tokens=250,
                    messages=[
                        {"role": "system", "content": "You are a gentle medical communication assistant. Speak simply and clearly."},
                        {"role": "user", "content": prompt},
                    ],
                )
                refined = response.choices[0].message.content.strip()
                if refined:
                    explanation = refined
            except Exception as e:
                # Fallback to deterministic template
                pass

        # Clean any asterisks or brackets for TTS
        clean_tts = explanation.replace("*", "").replace("#", "").replace("`", "").strip()

        return {
            "explanation": explanation,
            "tts_speech_text": clean_tts,
            "language": language,
        }

    def _generate_english_template(self, condition_name: str, triage: str, red_flags: list) -> str:
        text = f"Hello. Based on your visual skin check and reported symptoms, the preliminary educational assessment corresponds to characteristics seen in {condition_name}. "
        if red_flags:
            text += f"Because there are notable symptoms such as changes or irritation, your triage urgency is rated as {triage}. "
            text += "We strongly recommend scheduling an in-person clinical review with a dermatologist for a professional dermoscopic check."
        else:
            text += "No acute emergency red flags were detected, but regular self-checks and a routine review by a licensed healthcare provider are always recommended."
        return text

    def _generate_hindi_template(self, condition_name: str, triage: str, red_flags: list) -> str:
        text = f"नमस्ते। आपकी त्वचा की छवि और बताए गए लक्षणों के आधार पर, यह शुरुआती जांच {condition_name} के लक्षणों से मेल खाती है। "
        if red_flags:
            text += f"क्योंकि कुछ विशेष बदलाव या असुविधा देखी गई है, इसलिए प्राथमिकता का स्तर {triage} है। "
            text += "कृपया जल्द से जल्द किसी योग्य त्वचा विशेषज्ञ यानी डर्मेटोलॉजिस्ट से व्यक्तिगत जांच करवाएं।"
        else:
            text += "फिलहाल कोई आपातकालीन लक्षण नहीं दिखे हैं, फिर भी किसी डॉक्टर से सामान्य जांच करवाना हमेशा सुरक्षित रहता है।"
        return text

