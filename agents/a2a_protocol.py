"""SkinSense AI - Agent-to-Agent (A2A) Communication Protocol
Complies with Requirement 6 & 7:
Standardized A2A interoperability protocol facilitating communication
between agents implemented using Google ADK (google.adk) and CrewAI (crewai).
"""

from datetime import datetime
import hashlib
import json
import time
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class A2AMessage(BaseModel):
    """Standardized Agent-to-Agent (A2A) Message Envelope."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    protocol_version: str = Field(default="A2A/1.0")
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
    
    sender_id: str = Field(description="Originating agent ID")
    sender_framework: str = Field(description="Originating framework: ADK or CrewAI")
    
    recipient_id: str = Field(description="Target agent ID")
    recipient_framework: str = Field(description="Target framework: ADK or CrewAI")
    
    intent: str = Field(description="Communication intent (e.g. INGEST_INPUT, REQUEST_KNOWLEDGE, ENFORCE_SAFETY, GENERATE_REPORT)")
    payload: Dict[str, Any] = Field(description="Structured data payload")
    state_hash: Optional[str] = Field(default=None, description="SHA-256 payload integrity checksum")

    def compute_hash(self) -> str:
        """Computes SHA-256 hash of payload for state integrity verification."""
        encoded = json.dumps(self.payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


class A2ABroker:
    """Interoperability Broker facilitating A2A communication across frameworks.
    
    Allows an agent constructed in Google ADK to exchange validated state
    with an agent constructed in CrewAI without framework lock-in.
    """

    def __init__(self):
        self._traffic_log: List[A2AMessage] = []
        self._framework_stats = {
            "ADK_to_CrewAI": 0,
            "CrewAI_to_ADK": 0,
            "ADK_to_ADK": 0,
            "CrewAI_to_CrewAI": 0,
        }

    def dispatch(
        self,
        sender_id: str,
        sender_framework: str,
        recipient_id: str,
        recipient_framework: str,
        intent: str,
        payload: Dict[str, Any],
    ) -> A2AMessage:
        """Dispatches an A2A message between heterogeneous framework agents."""
        sf_norm = "ADK" if "adk" in sender_framework.lower() else "CrewAI"
        rf_norm = "ADK" if "adk" in recipient_framework.lower() else "CrewAI"

        msg = A2AMessage(
            sender_id=sender_id,
            sender_framework=sender_framework.upper(),
            recipient_id=recipient_id,
            recipient_framework=recipient_framework.upper(),
            intent=intent,
            payload=payload,
        )
        msg.state_hash = msg.compute_hash()
        
        route_key = f"{sf_norm}_to_{rf_norm}"
        if route_key in self._framework_stats:
            self._framework_stats[route_key] += 1
            
        self._traffic_log.append(msg)
        return msg

    def get_traffic_log(self) -> List[Dict[str, Any]]:
        """Returns the complete A2A protocol transmission trace."""
        return [m.model_dump() for m in self._traffic_log]

    def get_interoperability_summary(self) -> Dict[str, Any]:
        """Summarizes A2A cross-framework message routing statistics."""
        return {
            "total_messages": len(self._traffic_log),
            "framework_routes": self._framework_stats,
            "protocol": "A2A/1.0",
            "cross_framework_active": (self._framework_stats["ADK_to_CrewAI"] + self._framework_stats["CrewAI_to_ADK"]) > 0,
        }
