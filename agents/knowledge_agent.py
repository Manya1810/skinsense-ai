"""SkinSense AI - Knowledge Retrieval Agent
Retrieves verified dermatology information and authoritative medical citations
(ISIC, AAD, BAD, DermNet NZ, WHO) for conditions from the curated knowledge base.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class KnowledgeAgent:
    """Agent that provides source-grounded clinical references and citations."""

    def __init__(self, db_path: Optional[str] = None):
        default_db = Path(__file__).resolve().parent.parent / "knowledge_base" / "dermatology_db.json"
        self.db_path = Path(db_path) if db_path else default_db
        self.kb = self._load_kb()

    def _load_kb(self) -> Dict[str, Any]:
        if self.db_path.exists():
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"classes": {}, "triage_red_flags": []}

    def retrieve_condition_info(self, class_code: str) -> Dict[str, Any]:
        """Retrieves clinical details, ABCDE mapping, differential diagnoses, and citations."""
        classes = self.kb.get("classes", {})
        code_clean = class_code.lower().strip()

        if code_clean in classes:
            return {
                "found": True,
                "class_code": code_clean,
                "data": classes[code_clean],
            }

        return {
            "found": False,
            "class_code": code_clean,
            "data": {
                "full_name": class_code.upper(),
                "category": "Unknown / Unmapped Condition",
                "risk_level": "Undetermined",
                "description": "Condition definition not found in local curated dermatology knowledge base.",
                "citations": [],
            },
        }

    def get_red_flag_guidelines(self) -> List[str]:
        """Returns standard triage red flags from curated clinical guidelines."""
        return self.kb.get("triage_red_flags", [])

