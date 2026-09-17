from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from ..core.graph_engine import FraudGraphEngine
from ..core.rules_engine import RulesEngine
from ..models.schema import (
    RiskLevel,
    RiskFactor,
    SuspiciousConnection,
    NodeDetailResponse,
    NodeData,
    GraphData,
)
from .feature_extractor import GraphFeatureExtractor
from .models import MLFraudPipeline
from .gnn import VectorizedGraphSAGE


class EnsembleFraudEngine:
    """
    Ensemble Fraud Risk Decisioning Engine.
    Fuses Graph Topology Rules, XGBoost ML probabilities,
    Isolation Forest anomaly outliers, and GNN message passing embeddings.
    """

    def __init__(
        self,
        graph_engine: FraudGraphEngine,
        rules_engine: RulesEngine,
        ml_pipeline: MLFraudPipeline,
        gnn: VectorizedGraphSAGE,
    ):
        self.ge = graph_engine
        self.rules = rules_engine
        self.ml = ml_pipeline
        self.gnn = gnn
        self.fe = GraphFeatureExtractor(graph_engine)
        self.cached_gnn_results: Dict[str, Any] = {}

    def run_full_scoring_pass(self):
        """
        Runs feature extraction, trains/updates ML models, executes GNN forward pass,
        and assigns risk scores to all graph entities.
        """
        # 1. Extract feature matrix for all accounts
        df_features, account_ids = self.fe.extract_features_matrix()
        if len(account_ids) == 0:
            return

        # 2. Determine labels based on ground-truth metadata or rule flags for bootstrapping
        labels = []
        for acc in account_ids:
            gt_label = self.ge.graph.nodes[acc].get("ground_truth_label")
            node = self.ge.get_node(acc)
            flagged = node.flagged if node else False
            if gt_label is not None:
                is_fraud = int(gt_label)
            else:
                rule_score, _, _, triggered = self.rules.evaluate_account_rules(acc)
                is_fraud = 1 if (flagged or rule_score >= 0.60 or len(triggered) >= 2) else 0
            labels.append(is_fraud)

        labels = np.array(labels)

        # 3. Train ML pipeline
        self.ml.train(df_features, labels)

        # 4. Construct adjacency matrix for GNN
        N = len(account_ids)
        acc_to_idx = {acc: i for i, acc in enumerate(account_ids)}
        adj = np.eye(N)  # self loops

        # Add transaction and shared-entity adjacency
        for i, u in enumerate(account_ids):
            # Check neighbors in graph
            for v in self.ge.get_undirected_view().neighbors(u):
                if v in acc_to_idx:
                    adj[i, acc_to_idx[v]] = 1.0
                elif self.ge.graph.nodes[v].get("type") in ["DEVICE", "IP", "USER"]:
                    # 2-hop connection through device or IP
                    for w in self.ge.get_undirected_view().neighbors(v):
                        if w in acc_to_idx and w != u:
                            adj[i, acc_to_idx[w]] = 1.0

        X_feats = df_features[GraphFeatureExtractor.FEATURE_NAMES].fillna(0.0).values
        self.cached_gnn_results = self.gnn.fit_and_embed(
            node_ids=account_ids,
            X_features=X_feats,
            adj_matrix=adj,
            labels=labels,
            epochs=120,
        )

        # 5. Compute ensemble scores for each account
        for acc in account_ids:
            score, risk_lvl, _, _, _ = self.score_account(acc)
            self.ge.update_node_risk(acc, score)

        # 6. Propagate scores to User, Device, and IP nodes
        self._propagate_entity_risks()

    def _propagate_entity_risks(self):
        """Propagates risk from accounts to associated Users, Devices, IPs, and Beneficiaries."""
        for node_id, data in self.ge.graph.nodes(data=True):
            ntype = data.get("type")
            if ntype == "ACCOUNT":
                continue

            nbrs = list(self.ge.get_undirected_view().neighbors(node_id))
            acc_risks = [
                float(self.ge.graph.nodes[n].get("risk_score", 0.0))
                for n in nbrs
                if self.ge.graph.nodes[n].get("type") == "ACCOUNT"
            ]

            if acc_risks:
                max_risk = max(acc_risks)
                avg_risk = sum(acc_risks) / len(acc_risks)
                # If device/IP is shared by multiple accounts, amplify score
                shared_penalty = 0.15 if (len(acc_risks) >= 2 and ntype in ["DEVICE", "IP"]) else 0.0
                propagated_score = min(0.99, max_risk * 0.85 + avg_risk * 0.15 + shared_penalty)
                self.ge.update_node_risk(node_id, round(propagated_score, 2))

    def score_account(
        self, account_id: str
    ) -> Tuple[float, RiskLevel, List[SuspiciousConnection], List[RiskFactor], Dict[str, float]]:
        """
        Calculates calibrated ensemble risk score for an account entity.
        Returns (risk_score, risk_level, suspicious_connections, risk_factors, ml_probs)
        """
        # 1. Rule Engine evaluation
        rule_score, connections, rule_factors, triggered_rules = self.rules.evaluate_account_rules(account_id)

        # 2. ML Pipeline predictions
        feat_dict = self.fe.extract_features_for_account(account_id)
        iso_score, xgb_prob, feat_contribs = self.ml.predict_account(feat_dict)

        # 3. GNN score & embedding
        gnn_score = self.cached_gnn_results.get("gnn_fraud_scores", {}).get(account_id, xgb_prob * 0.9)

        # 4. Ensemble Blending
        # Weights: Graph Rules (0.35), XGBoost (0.30), Isolation Forest Anomaly (0.15), GNN (0.20)
        raw_ensemble = (
            rule_score * 0.35
            + xgb_prob * 0.30
            + iso_score * 0.15
            + gnn_score * 0.20
        )

        # Hard boosts for critical triggers (e.g., active circular ring, multi-account emulator collision)
        if any("RULE_CIRCULAR_TRANSFER" in r for r in triggered_rules):
            raw_ensemble = max(raw_ensemble, 0.92)
        if any("RULE_DEVICE_COLLISION" in r for r in triggered_rules) and feat_dict.get("has_emulator_device", 0) > 0:
            raw_ensemble = max(raw_ensemble, 0.93)
        if any("RULE_MULE_AGGREGATION" in r for r in triggered_rules):
            raw_ensemble = max(raw_ensemble, 0.90)

        # Respect confirmed illicit ground truth or pre-flagged entities
        node_raw = self.ge.graph.nodes.get(account_id, {})
        if node_raw.get("ground_truth_label") == 1 or node_raw.get("flagged"):
            raw_ensemble = max(raw_ensemble, 0.78)

        final_score = round(float(np.clip(raw_ensemble, 0.02, 0.99)), 2)

        # Determine level
        if final_score >= 0.80:
            risk_level = RiskLevel.CRITICAL
        elif final_score >= 0.60:
            risk_level = RiskLevel.HIGH
        elif final_score >= 0.35:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Combine risk factors
        all_factors = list(rule_factors)

        if xgb_prob >= 0.50:
            all_factors.append(
                RiskFactor(
                    name=f"XGBoost Supervised Fraud Probability ({xgb_prob:.0%})",
                    score=xgb_prob,
                    weight=0.30,
                    description=f"ML tree ensemble scored high fraud likelihood based on multi-relational graph attributes.",
                    category="ML_ANOMALY",
                )
            )

        if iso_score >= 0.65:
            all_factors.append(
                RiskFactor(
                    name=f"Isolation Forest Multi-Dimensional Outlier ({iso_score:.0%})",
                    score=iso_score,
                    weight=0.15,
                    description="Unsupervised anomaly detector identified unusual topological & velocity feature vectors.",
                    category="ML_ANOMALY",
                )
            )

        if gnn_score >= 0.60:
            all_factors.append(
                RiskFactor(
                    name=f"GNN GraphSAGE Relational Embedding Risk ({gnn_score:.0%})",
                    score=gnn_score,
                    weight=0.20,
                    description="Deep message passing in 2-hop neighborhood indicates high topological proximity to fraud clusters.",
                    category="GRAPH_CENTRALITY",
                )
            )

        ml_probs = {
            "xgb_fraud_probability": xgb_prob,
            "isolation_forest_anomaly": iso_score,
            "gnn_fraud_score": gnn_score,
            "base_rule_score": round(rule_score, 4),
        }

        return final_score, risk_level, connections, all_factors, ml_probs

    def get_full_node_profile(self, node_id: str) -> Optional[NodeDetailResponse]:
        """Builds comprehensive entity dossier for UI inspector, SAR reports, and case investigation."""
        node_obj = self.ge.get_node(node_id)
        if not node_obj:
            return None

        # Ego subgraph
        ego_sub = self.ge.get_ego_graph(node_id, radius=2)

        # If it's an account, run detailed account scoring
        if node_obj.type.value == "ACCOUNT":
            risk_score, risk_lvl, connections, factors, ml_probs = self.score_account(node_id)
            shared_devs, shared_ips = self.ge.find_shared_entities(node_id)
        else:
            risk_score = node_obj.risk_score
            risk_lvl = node_obj.risk_level
            connections = []
            factors = []
            ml_probs = {}
            shared_devs = []
            shared_ips = []

        # Graph centrality metrics
        g_metrics = {
            "pagerank": round(self.ge.get_pagerank(node_id), 6),
            "betweenness": round(self.ge.get_betweenness(node_id), 6),
            "in_degree": self.ge.graph.in_degree(node_id),
            "out_degree": self.ge.graph.out_degree(node_id),
            "community_id": node_obj.community_id,
        }

        # Transaction history
        tx_history = []
        for u, v, data in self.ge.graph.edges(data=True):
            if u == node_id or v == node_id:
                if data.get("type") == "TRANSFERRED":
                    direction = "OUTBOUND" if u == node_id else "INBOUND"
                    counterparty = v if u == node_id else u
                    c_node = self.ge.get_node(counterparty)
                    tx_history.append(
                        {
                            "id": data.get("id", f"tx_{u}_{v}"),
                            "direction": direction,
                            "counterparty_id": counterparty,
                            "counterparty_label": c_node.label if c_node else counterparty,
                            "amount": data.get("amount", 0.0),
                            "timestamp": data.get("timestamp"),
                            "is_suspicious": data.get("is_suspicious", False),
                        }
                    )

        # GNN embedding vector & 2D PCA coords
        gnn_emb = self.cached_gnn_results.get("embeddings", {}).get(node_id, [])

        # Generate SAR (Suspicious Activity Report) compliance narrative
        sar_text = self._generate_sar_summary(node_obj, risk_score, risk_lvl, connections, factors, tx_history)

        return NodeDetailResponse(
            node=node_obj,
            risk_score=risk_score,
            risk_level=risk_lvl,
            suspicious_connections=connections,
            risk_factors=factors,
            ego_subgraph=ego_sub,
            shared_devices=shared_devs,
            shared_ips=shared_ips,
            transaction_history=tx_history[:15],
            graph_metrics=g_metrics,
            ml_probabilities=ml_probs,
            gnn_embedding=gnn_emb,
            sar_summary=sar_text,
        )

    def _generate_sar_summary(
        self,
        node: NodeData,
        risk_score: float,
        risk_level: RiskLevel,
        connections: List[SuspiciousConnection],
        factors: List[RiskFactor],
        tx_history: List[Dict[str, Any]],
    ) -> str:
        """Constructs regulatory-compliant Suspicious Activity Report (SAR) summary narrative."""
        lines = [
            f"=== SUSPICIOUS ACTIVITY REPORT (SAR) INVESTIGATION DOSSIER ===",
            f"TARGET ENTITY: {node.label} (ID: {node.id} | Type: {node.type.value})",
            f"COMPOSITE RISK RATING: {risk_score:.2f} [{risk_level.value}]",
            f"STATUS: {'FLAGGED / RESTRICTED' if node.flagged or risk_score >= 0.75 else 'MONITORED'}",
            "",
            "1. EXECUTIVE SUMMARY & KEY TYPOLOGY FINDINGS:",
        ]

        if not factors:
            lines.append("   - Normal transactional behavior with no anomalous topological clusters detected.")
        else:
            for idx, factor in enumerate(factors[:4], 1):
                lines.append(f"   {idx}. {factor.name}: {factor.description}")

        lines.append("")
        lines.append("2. SUSPICIOUS GRAPH CONNECTION PATHWAYS:")
        if not connections:
            lines.append("   - No multi-accounting device/IP collisions or circular layering paths identified.")
        else:
            for idx, conn in enumerate(connections[:4], 1):
                lines.append(f"   [{idx}] {conn.chain}")
                lines.append(f"       Details: {conn.description}")

        lines.append("")
        lines.append(f"3. RECENT ACTIVITY SNAPSHOT ({len(tx_history)} recorded transactions):")
        for tx in tx_history[:3]:
            lines.append(f"   - {tx['direction']} ${tx['amount']:,.2f} with {tx['counterparty_label']} [{'SUSPICIOUS' if tx['is_suspicious'] else 'STANDARD'}]")

        lines.append("")
        lines.append("4. RECOMMENDED COMPLIANCE ACTION:")
        if risk_level == RiskLevel.CRITICAL:
            lines.append("   >>> IMMEDIATE ACTION REQUIRED: Freeze account credentials, block linked hardware fingerprints, submit FinCEN SAR filing.")
        elif risk_level == RiskLevel.HIGH:
            lines.append("   >>> ENHANCED DUE DILIGENCE (EDD): Require multi-factor biometric re-authentication and restrict transfer velocity limits.")
        elif risk_level == RiskLevel.MEDIUM:
            lines.append("   >>> CONTINUOUS MONITORING: Add to high-risk watchlist; alert on transfers exceeding $5,000 threshold.")
        else:
            lines.append("   >>> STANDARD CLEARANCE: Entity within standard behavioral baseline.")

        return "\n".join(lines)
