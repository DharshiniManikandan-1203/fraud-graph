from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class EntityType(str, Enum):
    USER = "USER"
    ACCOUNT = "ACCOUNT"
    TRANSACTION = "TRANSACTION"
    DEVICE = "DEVICE"
    IP = "IP"
    BENEFICIARY = "BENEFICIARY"


class EdgeType(str, Enum):
    OWNS = "OWNS"
    TRANSFERRED = "TRANSFERRED"
    ACCESSED_FROM = "ACCESSED_FROM"
    CONNECTED_VIA = "CONNECTED_VIA"
    HAS_BENEFICIARY = "HAS_BENEFICIARY"
    LINKED_CREDENTIAL = "LINKED_CREDENTIAL"


class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"  # >= 0.85
    HIGH = "HIGH"          # >= 0.65
    MEDIUM = "MEDIUM"      # >= 0.40
    LOW = "LOW"            # < 0.40


class NodeData(BaseModel):
    id: str
    label: str
    type: EntityType
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    flagged: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)
    community_id: Optional[int] = None
    created_at: Optional[str] = None


class EdgeData(BaseModel):
    id: str
    source: str
    target: str
    type: EdgeType
    amount: Optional[float] = None
    timestamp: Optional[str] = None
    risk_weight: float = 0.0
    is_suspicious: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class GraphData(BaseModel):
    nodes: List[NodeData]
    edges: List[EdgeData]
    total_nodes: int = 0
    total_edges: int = 0
    flagged_nodes: int = 0
    total_volume_at_risk: float = 0.0
    avg_network_risk: float = 0.0


class SuspiciousConnection(BaseModel):
    chain: str  # e.g., "Account A -> Device X -> Account B"
    source_id: str
    intermediary_id: str
    target_id: str
    connection_type: str  # "SHARED_DEVICE", "SHARED_IP", "CIRCULAR_TRANSFER", "MULE_FANOUT"
    description: str
    severity: RiskLevel


class RiskFactor(BaseModel):
    name: str
    score: float  # contribution
    weight: float
    description: str
    category: str  # "GRAPH_CENTRALITY", "DEVICE_SHARING", "IP_COLLISION", "VELOCITY", "ML_ANOMALY"


class NodeDetailResponse(BaseModel):
    node: NodeData
    risk_score: float
    risk_level: RiskLevel
    suspicious_connections: List[SuspiciousConnection] = Field(default_factory=list)
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    ego_subgraph: GraphData
    shared_devices: List[Dict[str, Any]] = Field(default_factory=list)
    shared_ips: List[Dict[str, Any]] = Field(default_factory=list)
    transaction_history: List[Dict[str, Any]] = Field(default_factory=list)
    graph_metrics: Dict[str, Any] = Field(default_factory=dict)
    ml_probabilities: Dict[str, float] = Field(default_factory=dict)
    gnn_embedding: List[float] = Field(default_factory=list)
    sar_summary: Optional[str] = None


class FraudRing(BaseModel):
    id: str
    name: str
    typology: str  # "DEVICE_FARM", "CIRCULAR_WASH", "MULE_NETWORK", "IP_CLUSTER", "ATO_SYBIL"
    severity: RiskLevel
    risk_score: float
    member_count: int
    member_ids: List[str]
    total_volume: float
    description: str
    detected_rules: List[str]
    nodes: List[NodeData]
    edges: List[EdgeData]


class TransactionScoreRequest(BaseModel):
    source_account: str
    target_account_or_beneficiary: str
    amount: float
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    currency: str = "USD"
    channel: str = "ONLINE"
    timestamp: Optional[str] = None


class TransactionScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    risk_level: RiskLevel
    decision: str  # "APPROVE", "REVIEW", "REJECT", "FREEZE_ACCOUNT"
    triggered_rules: List[str]
    suspicious_connections: List[SuspiciousConnection]
    risk_factors: List[RiskFactor]
    ml_anomaly_score: float
    xgb_fraud_prob: float
    gnn_risk_score: float
    impact_summary: str
