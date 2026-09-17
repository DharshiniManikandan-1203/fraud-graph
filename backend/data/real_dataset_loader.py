"""
Real-World Dataset Loader & Custom CSV/JSON Ingestion Engine.
Ingests unified real-world benchmarks (Elliptic, PaySim, IBM AML)
and custom external CSV transaction logs into FraudGraphEngine.
"""

import json
import io
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from ..core.graph_engine import FraudGraphEngine
from ..models.schema import EntityType, EdgeType


DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
UNIFIED_DATASET_PATH = DATASETS_DIR / "unified_real_fraud_graph.json"


class RealDatasetLoader:
    """
    Handles ingestion of real-world benchmarks and arbitrary CSV/JSON streams.
    """

    def __init__(self, graph_engine: FraudGraphEngine):
        self.ge = graph_engine

    def load_unified_real_dataset(self) -> Dict[str, Any]:
        """
        Loads the compiled, harmonized Elliptic + PaySim + IBM AML dataset into the active graph.
        """
        if not UNIFIED_DATASET_PATH.exists():
            from .real_dataset_preparer import build_unified_real_dataset
            build_unified_real_dataset()

        with open(UNIFIED_DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.ge.clear()

        # Ingest Nodes
        for n in data.get("nodes", []):
            node_type_str = n.get("type", "ACCOUNT").upper()
            try:
                node_type = EntityType(node_type_str)
            except ValueError:
                node_type = EntityType.ACCOUNT

            self.ge.add_node(
                node_id=n["id"],
                node_type=node_type,
                label=n.get("label", n["id"]),
                risk_score=n.get("risk_score", 0.50),
                flagged=n.get("flagged", False),
                balance=n.get("balance", 0.0),
                account_type=n.get("account_type", "CHECKING"),
                currency=n.get("currency", "USD"),
                kyc_status=n.get("kyc_status"),
                is_emulator=n.get("is_emulator", False),
                is_rooted=n.get("is_rooted", False),
                is_vpn=n.get("is_vpn", False),
                is_datacenter=n.get("is_datacenter", False),
                is_tor=n.get("is_tor", False),
                source_dataset=n.get("source_dataset", "REAL_COMBINED"),
                ground_truth_label=n.get("ground_truth_label", 0),
                ground_truth_typology=n.get("ground_truth_typology", "UNSPECIFIED"),
            )

        # Ingest Edges
        for e in data.get("edges", []):
            edge_type_str = e.get("type", "TRANSFERRED").upper()
            try:
                edge_type = EdgeType(edge_type_str)
            except ValueError:
                edge_type = EdgeType.TRANSFERRED

            self.ge.add_edge(
                source=e["source"],
                target=e["target"],
                edge_type=edge_type,
                amount=float(e.get("amount", 0.0)),
                timestamp=e.get("timestamp"),
                risk_weight=0.90 if e.get("is_fraud") else 0.10,
                is_suspicious=bool(e.get("is_fraud", False)),
                memo=e.get("memo"),
                channel=e.get("channel"),
                source_dataset=e.get("source_dataset", "REAL_COMBINED"),
            )

        return {
            "dataset_name": data.get("metadata", {}).get("name", "Unified Real-World Fraud Graph"),
            "total_nodes": len(self.ge.graph.nodes),
            "total_edges": len(self.ge.graph.edges),
            "sources": data.get("metadata", {}).get("sources", []),
            "metadata": data.get("metadata", {}),
        }

    def ingest_custom_csv(self, csv_content: str) -> Dict[str, Any]:
        """
        Parses raw CSV content with smart column inference and builds graph entities and edges.
        """
        df = pd.read_csv(io.StringIO(csv_content))
        self.ge.clear()

        # Identify columns
        cols = {c.lower().strip(): c for c in df.columns}
        
        src_col = cols.get("source_account") or cols.get("source") or cols.get("nameorig") or cols.get("from") or cols.get("sender")
        dst_col = cols.get("target_account") or cols.get("target") or cols.get("namedest") or cols.get("to") or cols.get("receiver") or cols.get("beneficiary")
        amt_col = cols.get("amount") or cols.get("amt") or cols.get("value") or cols.get("tx_amount")
        time_col = cols.get("timestamp") or cols.get("time") or cols.get("date") or cols.get("step")
        fraud_col = cols.get("is_fraud") or cols.get("isfraud") or cols.get("fraud") or cols.get("label") or cols.get("isflaggedfraud")
        dev_col = cols.get("device_id") or cols.get("device") or cols.get("hardware_id")
        ip_col = cols.get("ip_address") or cols.get("ip") or cols.get("ip_addr")
        type_col = cols.get("channel") or cols.get("type") or cols.get("tx_type")

        if not src_col or not dst_col:
            raise ValueError(f"CSV must contain source and destination columns (found: {list(df.columns)})")

        for _, row in df.iterrows():
            src_id = str(row[src_col]).strip()
            dst_id = str(row[dst_col]).strip()
            amt = float(row[amt_col]) if amt_col and pd.notna(row[amt_col]) else 100.0
            t_stamp = str(row[time_col]) if time_col and pd.notna(row[time_col]) else None
            is_fraud = bool(row[fraud_col] == 1 or row[fraud_col] is True or str(row[fraud_col]).lower() in ["true", "1", "fraud"]) if fraud_col and pd.notna(row[fraud_col]) else False
            channel = str(row[type_col]) if type_col and pd.notna(row[type_col]) else "TRANSFER"

            # Add Source Account
            if not self.ge.graph.has_node(src_id):
                self.ge.add_node(
                    node_id=src_id,
                    node_type=EntityType.ACCOUNT,
                    label=f"Acc {src_id}",
                    risk_score=0.85 if is_fraud else 0.10,
                    flagged=is_fraud,
                    source_dataset="CUSTOM_UPLOAD",
                )

            # Add Target Node (Account or Beneficiary)
            if not self.ge.graph.has_node(dst_id):
                is_bene = "BENE" in dst_id.upper() or "MERCHANT" in dst_id.upper() or "M" == dst_id[:1]
                self.ge.add_node(
                    node_id=dst_id,
                    node_type=EntityType.BENEFICIARY if is_bene else EntityType.ACCOUNT,
                    label=f"Target {dst_id}",
                    risk_score=0.80 if is_fraud else 0.08,
                    flagged=is_fraud,
                    source_dataset="CUSTOM_UPLOAD",
                )

            # Optional Device Link
            if dev_col and pd.notna(row[dev_col]):
                dev_id = str(row[dev_col]).strip()
                if not self.ge.graph.has_node(dev_id):
                    self.ge.add_node(
                        node_id=dev_id,
                        node_type=EntityType.DEVICE,
                        label=f"Device {dev_id}",
                        risk_score=0.75 if is_fraud else 0.05,
                        is_emulator=is_fraud,
                        source_dataset="CUSTOM_UPLOAD",
                    )
                self.ge.add_edge(src_id, dev_id, EdgeType.ACCESSED_FROM)

            # Optional IP Link
            if ip_col and pd.notna(row[ip_col]):
                ip_id = str(row[ip_col]).strip()
                if not self.ge.graph.has_node(ip_id):
                    self.ge.add_node(
                        node_id=ip_id,
                        node_type=EntityType.IP,
                        label=f"IP {ip_id}",
                        risk_score=0.70 if is_fraud else 0.04,
                        is_vpn=is_fraud,
                        source_dataset="CUSTOM_UPLOAD",
                    )
                self.ge.add_edge(src_id, ip_id, EdgeType.CONNECTED_VIA)

            # Add Transaction Edge
            self.ge.add_edge(
                source=src_id,
                target=dst_id,
                edge_type=EdgeType.TRANSFERRED,
                amount=amt,
                timestamp=t_stamp,
                risk_weight=0.90 if is_fraud else 0.10,
                is_suspicious=is_fraud,
                channel=channel,
                source_dataset="CUSTOM_UPLOAD",
            )

        return {
            "message": "Custom CSV successfully ingested into FraudGraphEngine",
            "total_nodes": len(self.ge.graph.nodes),
            "total_edges": len(self.ge.graph.edges),
            "rows_processed": len(df),
        }

    @staticmethod
    def get_available_datasets() -> List[Dict[str, Any]]:
        """
        Returns list of available public benchmark and synthetic datasets.
        """
        return [
            {
                "id": "REAL_COMBINED",
                "name": "⚡ Combined Real Dataset (Elliptic + PaySim + IBM)",
                "category": "REAL_WORLD_BENCHMARK",
                "description": "Unified multi-relational dataset combining real Bitcoin transaction graphs (Elliptic), mobile money financial logs (PaySim), and multi-hop money laundering SAR networks (IBM Research).",
                "sources": ["MIT-IBM / Elliptic", "NTNU / Kaggle", "IBM Research / FinCEN SAR"],
                "typologies": ["Darknet Markets", "Ransomware Extortion", "Peeling Chains", "CoinJoin Mixers", "Account Takeover (ATO)", "Rapid Cash-Out Drains", "4-Hop Circular Layering", "Sub-CTR Structuring"],
                "is_real": True,
                "node_count": 45,
                "edge_count": 46,
            },
            {
                "id": "ELLIPTIC_BITCOIN",
                "name": "⛓️ Elliptic Bitcoin AML Graph",
                "category": "REAL_WORLD_CRYPTO",
                "description": "Authentic cryptocurrency transaction graph with real illicit ransomware/darknet nodes, peeling chains, and mixer dispersal contracts.",
                "sources": ["MIT-IBM / Elliptic (Kaggle)"],
                "typologies": ["Ransomware Payouts", "Darknet Market Cashouts", "Wasabi CoinJoin Pools", "Blender.io Sanctioned Mixers"],
                "is_real": True,
                "node_count": 12,
                "edge_count": 8,
            },
            {
                "id": "PAYSIM_FINANCIAL",
                "name": "📱 PaySim Mobile Banking Fraud",
                "category": "REAL_WORLD_BANKING",
                "description": "Real mobile money transaction logs with unauthorized transfers, account draining, and rapid ATM cash-out liquidation.",
                "sources": ["NTNU / Kaggle PaySim"],
                "typologies": ["Phished Account Takeover", "Mule Account Funneling", "Immediate Cash-Out Liquidations"],
                "is_real": True,
                "node_count": 11,
                "edge_count": 10,
            },
            {
                "id": "ENTERPRISE",
                "name": "🌐 Synthetic Enterprise Multi-Topology",
                "category": "SYNTHETIC_SIMULATION",
                "description": "Comprehensive simulation covering clean baseline banking, device farm syndicates, circular wash rings, and mule aggregation.",
                "sources": ["FraudGraph Synthetic Generator"],
                "typologies": ["Sybil Device Farms", "Circular 4-Hop Wash Rings", "Mule Aggregation Hubs", "ATO Incidents"],
                "is_real": False,
                "node_count": 35,
                "edge_count": 42,
            },
        ]
