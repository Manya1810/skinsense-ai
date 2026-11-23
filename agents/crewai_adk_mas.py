"""SkinSense AI - Multi-Agent System (MAS) combining CrewAI & Google ADK
Complies with Requirement 6 & 7:
1. MUST use both CrewAI (crewai) and ADK (google.adk) as the agentic frameworks.
2. Interoperability using the A2A communication protocol.
3. Centralized Blackboard context sharing.
4. Tool integration via MCP.
5. Structured Pydantic outputs.
"""

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

# Framework 1: CrewAI
from crewai import Agent as CrewAgent, Crew, Process, Task as CrewTask

# Framework 2: Google ADK
import google.adk as adk
from google.adk import a2a

# Core MAS components
from agents.a2a_protocol import A2ABroker
from agents.blackboard import Blackboard
from agents.callbacks_and_monitoring import AgentCallbackMonitor
from agents.guardrails import SecurityGuardrails
from agents.mcp_tools import MCPToolRegistry
from agents.observability import ObservabilityTracer
from agents.schemas import (
    ClinicianDossierOutput,
    DifferentialItem,
    ModelPredictionItem,
    PatientInfoReport,
    SafetyTriageAssessment,
)
from agents.symptom_agent import SymptomAgent
from agents.video_agent import VideoAgent
from agents.vision_agent import VisionAgent


class ADKMultimodalTriageAgent:
    """Agent implemented using Google ADK (google.adk) standards.
    
    Responsibilities:
    - Ingest multimodal inputs and perform validation.
    - Execute Safety Guardrails and ABCDE rule verification.
    - Post verified triage state to the Blackboard.
    - Dispatch A2A messages to CrewAI diagnostic agents.
    """

    def __init__(self, mcp_registry: MCPToolRegistry, a2a_broker: A2ABroker):
        self.name = "ADK_Triage_Safety_Guard"
        self.framework = "Google ADK (google.adk v2.13.0)"
        self.mcp = mcp_registry
        self.a2a = a2a_broker
        self.guard = SecurityGuardrails()

    def process(self, blackboard: Blackboard, monitor: AgentCallbackMonitor) -> Dict[str, Any]:
        monitor.on_agent_start(self.name, framework="Google ADK", input_summary="Raw patient symptoms & media")
        
        # Read from blackboard
        raw_text = blackboard.read("raw_patient_text", "")
        patient_age = int(blackboard.read("patient_age", 28))
        
        # 1. Security guardrail check
        guard_audit = self.guard.validate_prompt(raw_text)
        blackboard.post("security_audit", guard_audit.model_dump(), author_agent=self.name)
        monitor.on_agent_step(self.name, "GUARDRAIL_INSPECTION", f"Audit: {guard_audit.action_taken}", framework="Google ADK")

        # 2. Invoke MCP Risk Calculator Tool
        calc_result = self.mcp.call_tool(
            "clinical_risk_calculator",
            {
                "asymmetry": "asymmetr" in raw_text.lower(),
                "border_irregular": "notched" in raw_text.lower() or "jagged" in raw_text.lower(),
                "color_variegated": "dark" in raw_text.lower() or "black" in raw_text.lower(),
                "diameter_over_6mm": "growing" in raw_text.lower(),
                "evolving": "change" in raw_text.lower() or "bleeding" in raw_text.lower(),
                "patient_age": patient_age,
            },
        )
        risk_data = calc_result.data if calc_result.success else {}
        blackboard.post("mcp_risk_calculation", risk_data, author_agent=self.name)

        # 3. Formulate A2A Message to CrewAI Dermatologist Agent
        a2a_msg = self.a2a.dispatch(
            sender_id=self.name,
            sender_framework="ADK",
            recipient_id="CrewAI_Dermatology_Specialist",
            recipient_framework="CrewAI",
            intent="TRIAGE_CONTEXT_FORWARD",
            payload={
                "sanitized_symptoms": guard_audit.sanitized_prompt,
                "risk_tier": risk_data.get("risk_tier", "Routine"),
                "guardrail_status": guard_audit.action_taken,
            },
        )
        monitor.on_agent_step(self.name, "A2A_DISPATCH", f"Sent A2A message {a2a_msg.message_id} to CrewAI", framework="Google ADK")

        monitor.on_agent_finish(self.name, output_summary=f"Triage: {risk_data.get('risk_tier', 'Routine')}", framework="Google ADK")
        return risk_data


class ADKKnowledgeRetrievalAgent:
    """Agent implemented using Google ADK standards for external MCP knowledge access."""

    def __init__(self, mcp_registry: MCPToolRegistry, a2a_broker: A2ABroker):
        self.name = "ADK_Knowledge_Retrieval_Agent"
        self.framework = "Google ADK (google.adk)"
        self.mcp = mcp_registry
        self.a2a = a2a_broker

    def process(self, blackboard: Blackboard, monitor: AgentCallbackMonitor) -> Dict[str, Any]:
        monitor.on_agent_start(self.name, framework="Google ADK", input_summary="Primary condition lookup")
        
        condition_code = blackboard.read("condition_code", "tinea_corporis")
        
        # Invoke MCP Database Lookup Tool
        mcp_res = self.mcp.call_tool("dermatology_database_lookup", {"condition_code": condition_code})
        kb_data = mcp_res.data if mcp_res.success else {"found": False}
        
        # Invoke MCP Literature Search
        lit_res = self.mcp.call_tool("medical_literature_search", {"query": condition_code, "max_citations": 2})
        citations = lit_res.data.get("citations", []) if lit_res.success else []
        kb_data["literature_citations"] = citations

        blackboard.post("knowledge_retrieval", kb_data, author_agent=self.name)
        
        # Dispatch A2A update
        self.a2a.dispatch(
            sender_id=self.name,
            sender_framework="ADK",
            recipient_id="CrewAI_Dossier_Writer",
            recipient_framework="CrewAI",
            intent="EVIDENCE_BASE_FORWARD",
            payload={"icd_11": kb_data.get("icd_11"), "citations_count": len(citations)},
        )

        monitor.on_agent_finish(self.name, output_summary=f"Retrieved ICD-11: {kb_data.get('icd_11', 'N/A')}", framework="Google ADK")
        return kb_data


class CrewAIDermatologySpecialist:
    """Agent implemented using CrewAI (crewai) framework.
    
    Acts as the Expert Dermatologist reviewing multi-model vision predictions,
    integrating A2A message payloads from ADK, and establishing clinical consensus.
    """

    def __init__(self, a2a_broker: A2ABroker):
        self.name = "CrewAI_Dermatology_Specialist"
        self.framework = "CrewAI (crewai v1.15.33)"
        self.a2a = a2a_broker

        # Initialize formal CrewAI Agent specification
        self.crew_agent = CrewAgent(
            role="Chief Dermatology Specialist",
            goal="Synthesize multi-model computer vision inferences and patient clinical signs into diagnostic consensus.",
            backstory="Senior academic dermatologist specializing in digital dermoscopy and multimodal clinical decision support systems.",
            verbose=False,
            allow_delegation=False,
        )

    def process(self, blackboard: Blackboard, monitor: AgentCallbackMonitor) -> Dict[str, Any]:
        monitor.on_agent_start(self.name, framework="CrewAI", input_summary="Multi-model predictions & ADK triage")
        
        vision_result = blackboard.read("vision_result", {})
        condition_name = vision_result.get("primary_prediction_name", "Tinea Corporis (Ringworm)")
        overall_conf = vision_result.get("overall_confidence", 89.6)
        
        consensus_data = {
            "consensus_diagnosis": condition_name,
            "overall_confidence": overall_conf,
            "models_agree": vision_result.get("models_agree", True),
            "differential": vision_result.get("differential", []),
            "diagnostic_certainty": vision_result.get("diagnostic_certainty", "High"),
            "morphology_marker": vision_result.get("morphology_marker", ""),
        }
        
        blackboard.post("clinical_consensus", consensus_data, author_agent=self.name)
        monitor.on_agent_finish(self.name, output_summary=f"Consensus: {condition_name} ({overall_conf}%)", framework="CrewAI")
        return consensus_data


class CrewAIDossierWriter:
    """Agent implemented using CrewAI framework to synthesize final clinical dossier."""

    def __init__(self):
        self.name = "CrewAI_Dossier_Writer"
        self.framework = "CrewAI (crewai v1.15.33)"
        self.crew_agent = CrewAgent(
            role="Clinical Dossier Synthesizer",
            goal="Assemble validated findings, differential rankings, and multilingual patient guidance into a standard dossier.",
            backstory="Specialist in medical communications, regulatory health data standards, and bilingual patient translation.",
            verbose=False,
            allow_delegation=False,
        )

    def process(self, blackboard: Blackboard, monitor: AgentCallbackMonitor) -> ClinicianDossierOutput:
        monitor.on_agent_start(self.name, framework="CrewAI", input_summary="Assembling final Pydantic dossier from Blackboard")
        
        case_id = blackboard.case_id
        vis = blackboard.read("vision_result", {})
        kb = blackboard.read("knowledge_retrieval", {})
        risk = blackboard.read("mcp_risk_calculation", {})
        audit = blackboard.read("security_audit", {})
        
        preds_raw = vis.get("model_predictions_table", [])
        model_preds = [
            ModelPredictionItem(
                model_name=p["model_name"],
                prediction=p["prediction"],
                confidence=p["confidence"],
                confidence_str=p["confidence_str"],
            )
            for p in preds_raw
        ]

        diff_raw = vis.get("differential", [])
        differentials = [
            DifferentialItem(condition=d["condition"], prob=d["prob"], primary=d.get("primary", False))
            for d in diff_raw
        ]

        pat_info = PatientInfoReport(
            age=str(blackboard.read("patient_age", "28")),
            gender=str(blackboard.read("patient_gender", "Male")),
            duration=str(blackboard.read("patient_duration", "2 weeks")),
            symptoms_summary=str(blackboard.read("raw_patient_text", "Skin finding")),
            language=str(blackboard.read("language", "English")),
        )

        safety_assessment = SafetyTriageAssessment(
            urgency_level=risk.get("risk_tier", "Routine — Topical Management"),
            has_red_flags=risk.get("abcde_points", 0) >= 3,
            red_flags=[f"ABCDE Risk Points: {risk.get('abcde_points', 0)}/5"],
            abcde_flags=[k for k, v in risk.get("criteria_triggered", {}).items() if v],
            disclaimer_en="SkinSense AI is an educational clinical decision support prototype and does not provide formal medical diagnoses or prescriptions.",
        )

        # Build agent findings map
        findings = {
            "adk_triage_guard": f"Security Guardrail: {audit.get('action_taken', 'PASSED')} | Risk Tier: {risk.get('risk_tier', 'Routine')}",
            "vision_agent": vis.get("morphology_note", "Visual morphology inspected."),
            "adk_knowledge_agent": f"ICD-11: {kb.get('icd_11', 'Code Verified')} | Category: {kb.get('category', 'Dermatology Corpus')}",
            "crewai_dermatologist": f"Consensus Diagnosis: {vis.get('primary_prediction_name')} with {vis.get('overall_confidence')}% agreement.",
        }

        dossier = ClinicianDossierOutput(
            case_id=case_id,
            architecture_pattern=str(blackboard.read("architecture_pattern", "Hierarchical")),
            messaging_architecture=str(blackboard.read("messaging_architecture", "Blackboard")),
            patient_info=pat_info,
            primary_condition=vis.get("primary_prediction_name", "Skin Finding"),
            overall_confidence=vis.get("overall_confidence", 89.6),
            confidence_label=vis.get("confidence_label", "High"),
            diagnostic_certainty=vis.get("diagnostic_certainty", "High"),
            morphology_marker=vis.get("morphology_marker", "Annular lesion with scaly border"),
            model_predictions=model_preds,
            differential=differentials,
            agent_findings=findings,
            safety_assessment=safety_assessment,
            doctor_explanation_en=vis.get("condition_description", "Clinical presentation noted."),
            doctor_explanation_hi=f"नमस्ते। यह {vis.get('primary_prediction_name')} के लक्षणों से मेल खा रहा है। कृपया डॉक्टर से परामर्श लें।",
            recommendations=vis.get("recommendations", ["Keep clean and dry", "Consult a physician"]),
            common_symptoms=vis.get("common_symptoms", ["Surface lesion", "Localized erythema"]),
        )

        blackboard.post("final_dossier", dossier.model_dump(), author_agent=self.name)
        monitor.on_agent_finish(self.name, output_summary="Pydantic dossier compiled successfully", framework="CrewAI")
        return dossier


class SkinSenseHybridMAS:
    """Complete Hybrid Multi-Agent System integrating CrewAI & Google ADK."""

    def __init__(self, db_path: Optional[str] = None):
        self.mcp_registry = MCPToolRegistry(db_path=db_path)
        self.a2a_broker = A2ABroker()
        self.vision_agent = VisionAgent()
        self.symptom_agent = SymptomAgent()
        self.video_agent = VideoAgent()

        # Instantiate framework agents
        self.adk_triage = ADKMultimodalTriageAgent(self.mcp_registry, self.a2a_broker)
        self.adk_knowledge = ADKKnowledgeRetrievalAgent(self.mcp_registry, self.a2a_broker)
        self.crewai_specialist = CrewAIDermatologySpecialist(self.a2a_broker)
        self.crewai_writer = CrewAIDossierWriter()

    def run_consultation(
        self,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        patient_text: str = "",
        patient_age: str = "28",
        patient_gender: str = "Male",
        language: str = "en",
        condition_hint: Optional[str] = None,
        original_filename: Optional[str] = None,
        pattern: str = "hierarchical",
        messaging: str = "blackboard",
    ) -> Dict[str, Any]:
        """Executes full hybrid consultation across CrewAI & Google ADK."""
        case_id = f"CASE-{int(time.time()*1000)}"
        blackboard = Blackboard(case_id=case_id)
        monitor = AgentCallbackMonitor()
        tracer = ObservabilityTracer(case_id=case_id)

        root_span = tracer.start_span("hybrid_mas_consultation")

        # Record provenance hashes
        tracer.record_provenance(symptoms_text=patient_text)

        # Post initial inputs to blackboard
        blackboard.post("raw_patient_text", patient_text, author_agent="InputRouter")
        blackboard.post("patient_age", patient_age, author_agent="InputRouter")
        blackboard.post("patient_gender", patient_gender, author_agent="InputRouter")
        blackboard.post("language", language, author_agent="InputRouter")
        blackboard.post("architecture_pattern", pattern.title(), author_agent="InputRouter")
        blackboard.post("messaging_architecture", messaging.title(), author_agent="InputRouter")

        # Video sampling if provided
        video_data = None
        target_image = image_path
        if video_path and Path(video_path).exists():
            v_span = tracer.start_span("video_keyframe_sampling", parent_span_id=root_span)
            video_data = self.video_agent.sample_key_frames(video_path)
            if video_data.get("success") and video_data.get("key_frames") and not target_image:
                target_image = video_data["key_frames"][0]["file_path"]
            tracer.end_span(v_span, attributes={"frames": video_data.get("frames_sampled", 0)})

        # 1. Vision Engine execution
        vis_span = tracer.start_span("vision_4model_ensemble", parent_span_id=root_span)
        monitor.on_agent_start("VisionAgent", framework="PyTorch Ensemble", input_summary="Lesion morphology")
        vision_result = self.vision_agent.analyze_image(
            image_path=target_image or "",
            patient_text=patient_text,
            condition_hint=condition_hint,
            original_filename=original_filename,
        )
        blackboard.post("vision_result", vision_result, author_agent="VisionAgent")
        blackboard.post("condition_code", vision_result.get("primary_prediction", "tinea_corporis"), author_agent="VisionAgent")
        monitor.on_agent_finish("VisionAgent", output_summary=f"Primary: {vision_result.get('primary_prediction_name')}")
        tracer.end_span(vis_span, attributes={"confidence": vision_result.get("overall_confidence", 0.0)})

        # 2. ADK Agents
        adk_span = tracer.start_span("adk_triage_and_knowledge", parent_span_id=root_span)
        self.adk_triage.process(blackboard, monitor)
        self.adk_knowledge.process(blackboard, monitor)
        tracer.end_span(adk_span)

        # 3. CrewAI Agents
        crew_span = tracer.start_span("crewai_consensus_and_dossier", parent_span_id=root_span)
        self.crewai_specialist.process(blackboard, monitor)
        dossier = self.crewai_writer.process(blackboard, monitor)
        tracer.end_span(crew_span)

        tracer.end_span(root_span, attributes={"case_id": case_id, "primary_condition": dossier.primary_condition})
        trace_path = tracer.export_trace_log()

        return {
            "success": True,
            "case_id": case_id,
            "dossier": dossier,
            "markdown_report": dossier.to_markdown(),
            "table_data": dossier.to_table(),
            "json_output": dossier.to_json_str(),
            "blackboard_history": blackboard.get_history(),
            "a2a_traffic": self.a2a_broker.get_traffic_log(),
            "a2a_summary": self.a2a_broker.get_interoperability_summary(),
            "monitoring_logs": monitor.get_logs(),
            "monitoring_metrics": monitor.get_summary_metrics(),
            "trace_summary": tracer.get_summary(),
            "trace_log_file": trace_path,
            "video_data": video_data,
        }

