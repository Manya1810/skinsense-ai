"""SkinSense AI - Symptom Agent
Parses user descriptions (in English or Hindi) into a clinical symptom schema.
Strictly extracts ONLY symptoms actually reported, explicitly marking
unmentioned attributes as 'unknown' rather than fabricating data.
"""

import re
from typing import Any, Dict


class SymptomAgent:
    """Agent that structures reported patient symptoms into clinical parameters."""

    def __init__(self):
        # Keyword dictionaries for English and Hindi
        self.itching_keywords = [
            "itch", "itching", "itchy", "scratch", "pruritus",
            "खुजली", "खाज", "नोचना"
        ]
        self.bleeding_keywords = [
            "bleed", "bleeding", "blood", "ooze", "oozing", "crust",
            "खून", "रक्त", "खून बहना", "मवाद"
        ]
        self.pain_keywords = [
            "pain", "painful", "hurts", "sore", "tender", "burning", "stinging",
            "दर्द", "दुखना", "जलन", "पीड़ा"
        ]
        self.growth_keywords = [
            "grow", "growing", "grown", "larger", "bigger", "enlarged", "spreading", "expanding",
            "बढ़ रहा", "बड़ा हो रहा", "फैलाव", "वृद्धि"
        ]
        self.color_change_keywords = [
            "darker", "turned black", "color change", "changed color", "discolored", "multi-colored",
            "रंग बदल", "काला पड़", "रंग में बदलाव"
        ]
        self.sun_exposure_keywords = [
            "sun", "sunlight", "outdoor", "sunburn", "tanning",
            "धूप", "सूरज", "सनबर्न"
        ]
        self.family_history_keywords = [
            "family", "father", "mother", "brother", "sister", "genetic", "cancer in family",
            "परिवार", "माता", "पिता", "भाई", "बहन", "खानदान"
        ]
        self.locations = [
            "face", "cheek", "nose", "forehead", "neck", "back", "chest", "arm",
            "hand", "leg", "foot", "shoulder", "abdomen", "scalp", "ear",
            "चेहरा", "नाक", "पीठ", "छाती", "हाथ", "पैर", "कंधा", "सिर"
        ]

    def _check_presence(self, text: str, keywords: list) -> str:
        """Determines if a symptom is explicitly affirmed, denied, or unknown."""
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                # Check for negation preceding keyword (e.g. "no pain", "बिना दर्द")
                negation_before = rf"(?:no|not|neither|without|never|ना|नहीं|बिना)\s+[\w\s]{{0,15}}{re.escape(kw)}"
                # Check for negation following keyword (e.g. "दर्द नहीं", "pain-free")
                negation_after = rf"{re.escape(kw)}\s+[\w\s]{{0,15}}(?:नहीं|ना|free|absent|negative)"
                if re.search(negation_before, text_lower) or re.search(negation_after, text_lower):
                    return "no"
                return "yes"
        return "unknown"

    def _extract_duration(self, text: str) -> str:
        """Extracts timeline/duration if mentioned."""
        patterns = [
            r"(\d+\s*(?:days?|weeks?|months?|years?))",
            r"(few\s+(?:days?|weeks?|months?|years?))",
            r"(\d+\s*(?:दिन|हफ्ते|महीने|साल))",
            r"(काफी\s+दिनों\s+से|कुछ\s+महीनों\s+से)",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return "unknown"

    def _extract_location(self, text: str) -> str:
        """Extracts anatomic site if mentioned."""
        text_lower = text.lower()
        for loc in self.locations:
            if loc in text_lower:
                return loc
        return "unknown"

    def structure_symptoms(self, patient_text: str) -> Dict[str, Any]:
        """Structures patient complaints into a strictly grounded clinical dictionary."""
        if not patient_text or not patient_text.strip():
            return {
                "reported": False,
                "symptoms": {
                    "duration": "unknown",
                    "itching_pruritus": "unknown",
                    "bleeding_oozing": "unknown",
                    "pain_tenderness": "unknown",
                    "evolution_growth": "unknown",
                    "color_change": "unknown",
                    "sun_exposure": "unknown",
                    "family_history": "unknown",
                    "anatomic_location": "unknown",
                },
                "summary": "No patient symptoms verbally reported or transcribed.",
            }

        symptoms = {
            "duration": self._extract_duration(patient_text),
            "itching_pruritus": self._check_presence(patient_text, self.itching_keywords),
            "bleeding_oozing": self._check_presence(patient_text, self.bleeding_keywords),
            "pain_tenderness": self._check_presence(patient_text, self.pain_keywords),
            "evolution_growth": self._check_presence(patient_text, self.growth_keywords),
            "color_change": self._check_presence(patient_text, self.color_change_keywords),
            "sun_exposure": self._check_presence(patient_text, self.sun_exposure_keywords),
            "family_history": self._check_presence(patient_text, self.family_history_keywords),
            "anatomic_location": self._extract_location(patient_text),
        }

        # Count active flags
        active_flags = [k for k, v in symptoms.items() if v == "yes"]

        return {
            "reported": True,
            "raw_text": patient_text.strip(),
            "symptoms": symptoms,
            "active_symptom_count": len(active_flags),
            "active_symptoms": active_flags,
            "missing_fields": [k for k, v in symptoms.items() if v == "unknown"],
        }
