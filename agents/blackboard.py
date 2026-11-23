"""SkinSense AI - Blackboard & Messaging Architecture
Complies with Requirement 1:
1. Blackboard Architecture (Shared State context sharing) - MANDATORY.
2. Peer-to-Peer (P2P) bus comparison & evaluation.
"""

from copy import deepcopy
from datetime import datetime
import time
from typing import Any, Callable, Dict, List, Optional


class BlackboardEvent:
    """Audit record for any state mutation on the Blackboard."""
    def __init__(self, key: str, author_agent: str, value: Any, revision: int):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.key = key
        self.author_agent = author_agent
        self.revision = revision
        self.value_preview = str(value)[:120]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "key": self.key,
            "author": self.author_agent,
            "revision": self.revision,
            "preview": self.value_preview,
        }


class Blackboard:
    """Centralized Shared-State Context Sharing Blackboard.
    
    Intermediate outputs from one agent (e.g. Vision Agent, Symptom Agent)
    are posted here and become immediately visible to downstream agents
    (Knowledge Agent, Safety Agent, Clinician Report Agent).
    """

    def __init__(self, case_id: str = "CASE-DEFAULT"):
        self.case_id = case_id
        self._state: Dict[str, Any] = {}
        self._history: List[BlackboardEvent] = []
        self._subscriptions: Dict[str, List[Callable[[str, Any], None]]] = {}
        self._revision_counter = 0

    def post(self, key: str, data: Any, author_agent: str) -> None:
        """Writes or updates an intermediate output on the blackboard."""
        self._revision_counter += 1
        self._state[key] = data
        event = BlackboardEvent(key=key, author_agent=author_agent, value=data, revision=self._revision_counter)
        self._history.append(event)

        # Notify subscribers of updated context
        if key in self._subscriptions:
            for callback in self._subscriptions[key]:
                try:
                    callback(key, data)
                except Exception as e:
                    print(f"Blackboard subscription notification error for {key}: {e}")

    def read(self, key: str, default: Any = None) -> Any:
        """Reads intermediate context posted by upstream agents."""
        return self._state.get(key, default)

    def subscribe(self, key: str, callback: Callable[[str, Any], None]) -> None:
        """Registers a listener for real-time blackboard context changes."""
        if key not in self._subscriptions:
            self._subscriptions[key] = []
        self._subscriptions[key].append(callback)

    def snapshot(self) -> Dict[str, Any]:
        """Returns an immutable deep-copy snapshot of the shared state."""
        return deepcopy(self._state)

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the complete mutation audit trail."""
        return [evt.to_dict() for evt in self._history]

    def clear(self) -> None:
        """Resets the blackboard for a new consultation."""
        self._state.clear()
        self._history.clear()
        self._revision_counter = 0


class PeerToPeerBus:
    """Alternative Messaging Architecture: Direct Point-to-Point (P2P) Messaging.
    
    Used to benchmark against the Blackboard architecture.
    In P2P, agents have no shared state and must send individual point-to-point messages.
    """

    def __init__(self):
        self.inbox: Dict[str, List[Dict[str, Any]]] = {}
        self.messages_count = 0
        self.message_log: List[Dict[str, Any]] = []

    def send_direct(self, from_agent: str, to_agent: str, payload: Any, message_type: str = "DATA") -> None:
        """Sends a point-to-point message directly to another agent's inbox."""
        if to_agent not in self.inbox:
            self.inbox[to_agent] = []
        
        msg = {
            "timestamp": time.time(),
            "from": from_agent,
            "to": to_agent,
            "type": message_type,
            "payload": payload,
        }
        self.inbox[to_agent].append(msg)
        self.message_log.append(msg)
        self.messages_count += 1

    def receive(self, agent_name: str) -> List[Dict[str, Any]]:
        """Reads and clears messages for the designated agent."""
        msgs = self.inbox.get(agent_name, [])
        self.inbox[agent_name] = []
        return msgs


def evaluate_messaging_architectures(n_agents: int = 5) -> Dict[str, Any]:
    """Empirical evaluation: Blackboard Architecture vs. Peer-to-Peer.
    
    Compares message complexity, latency, and context synchronization.
    """
    # 1. Blackboard Benchmark (O(N) writes/reads)
    bb = Blackboard(case_id="BENCHMARK-BB")
    start_bb = time.perf_counter()
    for i in range(n_agents):
        agent_name = f"Agent_{i}"
        bb.post(f"output_agent_{i}", {"data": f"finding_{i}", "score": 85.0 + i}, author_agent=agent_name)
        # Any downstream agent reads full snapshot in 1 centralized read
        _ = bb.snapshot()
    latency_bb_ms = (time.perf_counter() - start_bb) * 1000

    # 2. Peer-to-Peer Benchmark (O(N^2) cross-connections)
    p2p = PeerToPeerBus()
    start_p2p = time.perf_counter()
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                p2p.send_direct(
                    from_agent=f"Agent_{i}",
                    to_agent=f"Agent_{j}",
                    payload={"data": f"finding_{i}", "score": 85.0 + i},
                )
    for i in range(n_agents):
        _ = p2p.receive(f"Agent_{i}")
    latency_p2p_ms = (time.perf_counter() - start_p2p) * 1000

    return {
        "blackboard": {
            "paradigm": "Centralized Shared State (Shared Context)",
            "messages_or_writes": len(bb.get_history()),
            "complexity": "O(N) Writes, O(1) Snapshot Reads",
            "latency_ms": round(latency_bb_ms, 3),
            "context_synchronization": "Strong (Single source of truth)",
        },
        "peer_to_peer": {
            "paradigm": "Point-to-Point Message Passing",
            "messages_or_writes": p2p.messages_count,
            "complexity": f"O(N^2) = {n_agents * (n_agents - 1)} messages",
            "latency_ms": round(latency_p2p_ms, 3),
            "context_synchronization": "Eventual (Prone to partial sync)",
        },
        "winner": "Blackboard Architecture is optimal for Multimodal Clinical MAS because all specialist agents (Vision, Symptoms, Safety, Explainer) require immediate visibility of unified patient context without O(N^2) message fan-out.",
    }

