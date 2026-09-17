import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from ..core.graph_engine import FraudGraphEngine
from ..models.schema import EntityType, EdgeType


class GraphFeatureExtractor:
    """
    Extracts tabular and multi-relational graph topological features
    for machine learning anomaly detection and supervised risk classification.
    """

    FEATURE_NAMES = [
        "pagerank",
        "betweenness",
        "in_degree",
        "out_degree",
        "total_degree",
        "degree_ratio",
        "clustering_coefficient",
        "community_size",
        "shared_device_count",
        "max_accounts_on_shared_device",
        "has_emulator_device",
        "shared_ip_count",
        "max_accounts_on_shared_ip",
        "has_vpn_or_tor_ip",
        "nbr_1hop_flagged_ratio",
        "nbr_avg_risk_score",
        "in_volume_total",
        "out_volume_total",
        "volume_velocity_ratio",
        "avg_tx_amount",
        "max_tx_amount",
        "in_cycle_flag",
        "min_cycle_length",
    ]

    def __init__(self, graph_engine: FraudGraphEngine):
        self.ge = graph_engine

    def extract_features_for_account(self, account_id: str) -> Dict[str, float]:
        """Extracts a vector of features for a single account node."""
        if account_id not in self.ge.graph:
            return {f: 0.0 for f in self.FEATURE_NAMES}

        g = self.ge.graph
        undirected = self.ge.get_undirected_view()
        node_data = g.nodes[account_id]

        # 1. Centralities
        pr = self.ge.get_pagerank(account_id)
        bw = self.ge.get_betweenness(account_id)

        # In & Out degree for transactions/connections
        in_deg = g.in_degree(account_id)
        out_deg = g.out_degree(account_id)
        tot_deg = in_deg + out_deg
        deg_ratio = float(in_deg) / max(1.0, float(out_deg))

        # Clustering coefficient on undirected projection
        try:
            clust_coeff = nx.clustering(undirected, account_id)
        except Exception:
            clust_coeff = 0.0

        # Community metrics
        communities = self.ge.compute_communities()
        comm_id = communities.get(account_id, 0)
        comm_size = sum(1 for _, c in communities.items() if c == comm_id)

        # 2. Shared Entities
        shared_devices, shared_ips = self.ge.find_shared_entities(account_id)

        shared_dev_count = len(shared_devices)
        max_acc_dev = max([d["account_count"] for d in shared_devices], default=1)
        has_emulator = 1.0 if any(d.get("device_details", {}).get("is_emulator", False) or d.get("device_details", {}).get("is_rooted", False) for d in shared_devices) else 0.0

        shared_ip_count = len(shared_ips)
        max_acc_ip = max([i["account_count"] for i in shared_ips], default=1)
        has_vpn_tor = 1.0 if any(i.get("ip_details", {}).get("is_vpn", False) or i.get("ip_details", {}).get("is_tor", False) or i.get("ip_details", {}).get("is_datacenter", False) for i in shared_ips) else 0.0

        # 3. Neighborhood Risk
        nbrs = list(undirected.neighbors(account_id))
        flagged_nbrs = 0
        nbr_risks = []
        for n in nbrs:
            nd = g.nodes[n]
            r = float(nd.get("risk_score", 0.0))
            nbr_risks.append(r)
            if nd.get("flagged", False) or r >= 0.70:
                flagged_nbrs += 1

        nbr_1hop_ratio = float(flagged_nbrs) / max(1, len(nbrs))
        nbr_avg_risk = float(np.mean(nbr_risks)) if nbr_risks else 0.0

        # 4. Volumes & Amounts
        in_vols = [d.get("amount", 0.0) or 0.0 for _, _, d in g.in_edges(account_id, data=True) if d.get("type") == EdgeType.TRANSFERRED.value]
        out_vols = [d.get("amount", 0.0) or 0.0 for _, _, d in g.out_edges(account_id, data=True) if d.get("type") == EdgeType.TRANSFERRED.value]
        all_vols = in_vols + out_vols

        in_vol_tot = float(sum(in_vols))
        out_vol_tot = float(sum(out_vols))
        vol_velocity_ratio = out_vol_tot / max(1.0, in_vol_tot) if in_vol_tot > 0 else (1.0 if out_vol_tot > 0 else 0.0)
        avg_amt = float(np.mean(all_vols)) if all_vols else 0.0
        max_amt = float(np.max(all_vols)) if all_vols else 0.0

        # 5. Cycles
        cycles = self.ge.detect_directed_cycles(max_cycle_length=6)
        account_cycles = [c for c in cycles if account_id in c]
        in_cycle_flag = 1.0 if len(account_cycles) > 0 else 0.0
        min_cycle_len = float(min([len(c) for c in account_cycles], default=0))

        feat_dict = {
            "pagerank": round(pr, 6),
            "betweenness": round(bw, 6),
            "in_degree": float(in_deg),
            "out_degree": float(out_deg),
            "total_degree": float(tot_deg),
            "degree_ratio": round(deg_ratio, 4),
            "clustering_coefficient": round(clust_coeff, 4),
            "community_size": float(comm_size),
            "shared_device_count": float(shared_dev_count),
            "max_accounts_on_shared_device": float(max_acc_dev),
            "has_emulator_device": has_emulator,
            "shared_ip_count": float(shared_ip_count),
            "max_accounts_on_shared_ip": float(max_acc_ip),
            "has_vpn_or_tor_ip": has_vpn_tor,
            "nbr_1hop_flagged_ratio": round(nbr_1hop_ratio, 4),
            "nbr_avg_risk_score": round(nbr_avg_risk, 4),
            "in_volume_total": round(in_vol_tot, 2),
            "out_volume_total": round(out_vol_tot, 2),
            "volume_velocity_ratio": round(vol_velocity_ratio, 4),
            "avg_tx_amount": round(avg_amt, 2),
            "max_tx_amount": round(max_amt, 2),
            "in_cycle_flag": in_cycle_flag,
            "min_cycle_length": min_cycle_len,
        }

        return feat_dict

    def extract_features_matrix(self, account_ids: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Extracts feature DataFrame for all accounts or a subset."""
        if account_ids is None:
            account_ids = [
                n for n, d in self.ge.graph.nodes(data=True) if d.get("type") == EntityType.ACCOUNT.value
            ]

        records = []
        for acc in account_ids:
            feats = self.extract_features_for_account(acc)
            feats["account_id"] = acc
            records.append(feats)

        if not records:
            df = pd.DataFrame(columns=self.FEATURE_NAMES + ["account_id"])
            return df, []

        df = pd.DataFrame(records)
        return df, account_ids
