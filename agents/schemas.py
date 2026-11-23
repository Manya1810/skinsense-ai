"""SkinSense AI - Structured Output Schemas (Pydantic v2)
Complies with Requirement 3: Strongly typed structured output in Markdown, Table, and JSON formats.
"""

from datetime import datetime
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelPredictionItem(BaseModel):
    """Prediction output from an individual computer vision architecture."""
    model_name: str = Field(description="Architecture name (e.g. EfficientNet-B0, ViT-B/16)")
    prediction: str = Field(description="Predicted clinical diagnosis")
    confidence: float = Field(ge=0.0, le=100.0, description="Confidence percentage")
    confidence_str: str = Field(description="Formatted confidence percentage string")


class DifferentialItem(BaseModel):
    """Ranked differential diagnosis probability item."""
    condition: str = Field(description="Dermatological condition name")
    prob: float = Field(ge=0.0, le=100.0, description="Differential probability percentage")
    primary: bool = Field(default=False, description="Flag indicating if this is primary diagnosis")


class PatientInfoReport(BaseModel):
    """Demographics and structured symptoms summary."""
    age: str = Field(default="28")
    gender: str = Field(default="Male")
    duration: str = Field(default="2 weeks")
    symptoms_summary: str = Field(default="Localized skin finding")
    language: str = Field(default="English")


class AgentFindingItem(BaseModel):
    """Standardized finding statement from an agent in the MAS."""
    agent_name: str = Field(description="Name of the agent (e.g. Vision Agent, Safety Agent)")
    framework: str = Field(default="CrewAI+ADK", description="Framework executing agent (ADK or CrewAI)")
    status: str = Field(default="completed", description="Agent execution status")
    statement: str = Field(description="Clinically grounded observation")
    latency_ms: float = Field(default=0.0, description="Execution time in milliseconds")


class SafetyTriageAssessment(BaseModel):
    """Safety evaluation, red flags, and ABCDE melanoma rule check."""
    urgency_level: str = Field(default="Routine", description="Triage category (Routine, Moderate, Critical)")
    has_red_flags: bool = Field(default=False)
    red_flags: List[str] = Field(default_factory=list)
    abcde_flags: List[str] = Field(default_factory=list)
    disclaimer_en: str = Field(description="Standardized medical educational disclaimer")


class ClinicianDossierOutput(BaseModel):
    """Complete multimodal clinical consultation dossier.
    
    Provides serialization to:
    - Markdown (to_markdown)
    - Table representation (to_table)
    - Validated JSON (to_json_str)
    """
    case_id: str = Field(description="Unique consultation case ID")
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    architecture_pattern: str = Field(default="Hierarchical", description="Agentic design pattern used")
    messaging_architecture: str = Field(default="Blackboard", description="Messaging paradigm (Blackboard vs P2P)")
    
    patient_info: PatientInfoReport
    primary_condition: str
    overall_confidence: float = Field(ge=0.0, le=100.0)
    confidence_label: str
    diagnostic_certainty: str
    morphology_marker: str
    
    model_predictions: List[ModelPredictionItem]
    differential: List[DifferentialItem]
    agent_findings: Dict[str, str]
    safety_assessment: SafetyTriageAssessment
    
    doctor_explanation_en: str
    doctor_explanation_hi: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)
    common_symptoms: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Serializes the dossier to formatted clinical Markdown."""
        lines = [
            f"# SkinSense AI — Clinician Review Dossier",
            f"**Case ID:** `{self.case_id}` | **Timestamp:** {self.timestamp}",
            f"**Architecture:** `{self.architecture_pattern}` | **Messaging:** `{self.messaging_architecture}`",
            f"**Triage Urgency:** `{self.safety_assessment.urgency_level}`\n",
            f"> **Educational Notice:** {self.safety_assessment.disclaimer_en}\n",
            f"## 1. Patient Demographics & Symptoms",
            f"- **Age:** {self.patient_info.age} | **Gender:** {self.patient_info.gender}",
            f"- **Duration:** {self.patient_info.duration}",
            f"- **Symptoms Summary:** {self.patient_info.symptoms_summary}\n",
            f"## 2. Multi-Model Consensus Diagnosis",
            f"- **Primary Finding:** **{self.primary_condition}**",
            f"- **Confidence:** **{self.overall_confidence}%** ({self.confidence_label})",
            f"- **Diagnostic Certainty:** {self.diagnostic_certainty}",
            f"- **Morphology Marker:** `{self.morphology_marker}`\n",
            f"### Vision Architecture Breakdown Table",
            f"| Model Architecture | Diagnosis | Confidence Score |",
            f"| :--- | :--- | :---: |",
        ]
        for m in self.model_predictions:
            lines.append(f"| {m.model_name} | {m.prediction} | {m.confidence_str} |")

        lines.extend([
            f"\n### Differential Diagnosis Distribution",
            f"| Condition | Probability | Status |",
            f"| :--- | :---: | :--- |",
        ])
        for d in self.differential:
            status = "Primary" if d.primary else "Secondary"
            lines.append(f"| {d.condition} | {d.prob}% | {status} |")

        lines.append(f"\n## 3. Multi-Agent Synthesis (CrewAI + Google ADK)")
        for agent_name, text in self.agent_findings.items():
            lines.append(f"- **{agent_name.replace('_', ' ').title()}:** {text}")

        lines.append(f"\n## 4. Care Recommendations")
        for rec in self.recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)

    def to_table(self) -> List[Dict[str, Any]]:
        """Returns structured tabular summary of models, differentials, and agents."""
        table_rows = []
        for m in self.model_predictions:
            table_rows.append({
                "Category": "Vision Model",
                "Entity": m.model_name,
                "Prediction": m.prediction,
                "Confidence": m.confidence_str,
            })
        for d in self.differential:
            table_rows.append({
                "Category": "Differential",
                "Entity": d.condition,
                "Prediction": "Primary" if d.primary else "Alternative",
                "Confidence": f"{d.prob}%",
            })
        for a, text in self.agent_findings.items():
            table_rows.append({
                "Category": "Agent Observation",
                "Entity": a.replace('_', ' ').title(),
                "Prediction": text[:60] + "...",
                "Confidence": "Verified",
            })
        return table_rows

    def to_json_str(self) -> str:
        """Returns validated indented JSON string."""
        return self.model_dump_json(indent=2)

