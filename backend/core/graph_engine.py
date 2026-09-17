import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from datetime import datetime
import json
from ..models.schema import (
    EntityType,
    EdgeType,
    RiskLevel,
    NodeData,
    EdgeData,
    GraphData,
    SuspiciousConnection,
)


class FraudGraphEngine:
    """
    In-memory multi-relational heterogeneous graph engine for financial fraud detection.
    Models entities (User, Account, Device, IP, Beneficiary) and relationships.
    """

    def __init__(self):
        # MultiDiGraph allows multiple directed edges between the same two nodes (e.g., multiple transactions)
        self.graph = nx.MultiDiGraph()
        # Undirected view / projected cache for fast neighborhood and community queries
        self._undirected_cache: Optional[nx.Graph] = None
        self._pagerank_cache: Dict[str, float] = {}
        self._betweenness_cache: Dict[str, float] = {}
        self._communities_cache: Dict[str, int] = {}
        self._dirty: bool = True

    def clear(self):
        self.graph.clear()
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._undirected_cache = None
        self._pagerank_cache = {}
        self._betweenness_cache = {}
        self._communities_cache = {}
        self._dirty = True

    def add_node(
        self,
        node_id: str,
        node_type: Union[EntityType, str],
        label: Optional[str] = None,
        risk_score: float = 0.0,
        flagged: bool = False,
        **details
    ) -> NodeData:
        self._invalidate_cache()
        if isinstance(node_type, str):
            node_type = EntityType(node_type)

        if not label:
            label = f"{node_type.value}:{node_id}"

        risk_level = self._compute_risk_level(risk_score)

        node_data = {
            "id": node_id,
            "label": label,
            "type": node_type.value,
            "risk_score": float(risk_score),
            "risk_level": risk_level.value,
            "flagged": flagged,
            "details": details,
            "created_at": details.get("created_at", datetime.now().isoformat()),
        }

        self.graph.add_node(node_id, **node_data)
        return NodeData(**node_data)

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: Union[EdgeType, str],
        edge_id: Optional[str] = None,
        amount: Optional[float] = None,
        timestamp: Optional[str] = None,
        risk_weight: float = 0.0,
        is_suspicious: bool = False,
        **details
    ) -> EdgeData:
        self._invalidate_cache()
        if isinstance(edge_type, str):
            edge_type = EdgeType(edge_type)

        if not edge_id:
            edge_id = f"{source}_{edge_type.value}_{target}_{len(self.graph.edges(source, target))}"

        if not timestamp:
            timestamp = datetime.now().isoformat()

        edge_data = {
            "id": edge_id,
            "source": source,
            "target": target,
            "type": edge_type.value,
            "amount": float(amount) if amount is not None else None,
            "timestamp": timestamp,
            "risk_weight": float(risk_weight),
            "is_suspicious": is_suspicious,
            "details": details,
        }

        self.graph.add_edge(source, target, key=edge_id, **edge_data)
        return EdgeData(**edge_data)

    def get_node(self, node_id: str) -> Optional[NodeData]:
        if node_id not in self.graph:
            return None
        attrs = dict(self.graph.nodes[node_id])
        attrs["type"] = EntityType(attrs["type"])
        attrs["risk_level"] = RiskLevel(attrs["risk_level"])
        return NodeData(**attrs)

    def update_node_risk(self, node_id: str, risk_score: float, flagged: Optional[bool] = None):
        if node_id in self.graph:
            self.graph.nodes[node_id]["risk_score"] = float(risk_score)
            self.graph.nodes[node_id]["risk_level"] = self._compute_risk_level(risk_score).value
            if flagged is not None:
                self.graph.nodes[node_id]["flagged"] = flagged

    def _compute_risk_level(self, score: float) -> RiskLevel:
        if score >= 0.80:
            return RiskLevel.CRITICAL
        elif score >= 0.60:
            return RiskLevel.HIGH
        elif score >= 0.35:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def get_undirected_view(self) -> nx.Graph:
        if self._undirected_cache is None or self._dirty:
            self._undirected_cache = self.graph.to_undirected(as_view=False)
        return self._undirected_cache

    def compute_centrality_metrics(self):
        """Computes PageRank, Degree, and Betweenness centralities across the graph."""
        if not self._pagerank_cache:
            try:
                # Directed PageRank
                self._pagerank_cache = nx.pagerank(self.graph, alpha=0.85, max_iter=200)
            except Exception:
                # Fallback for disconnected or singular graph
                self._pagerank_cache = {n: 1.0 / max(1, len(self.graph)) for n in self.graph.nodes()}

        if not self._betweenness_cache:
            try:
                undirected = self.get_undirected_view()
                # Sample betweenness if graph is large
                k = min(len(undirected), 200) if len(undirected) > 200 else None
                self._betweenness_cache = nx.betweenness_centrality(undirected, k=k)
            except Exception:
                self._betweenness_cache = {n: 0.0 for n in self.graph.nodes()}

    def get_pagerank(self, node_id: str) -> float:
        if not self._pagerank_cache:
            self.compute_centrality_metrics()
        return self._pagerank_cache.get(node_id, 0.0)

    def get_betweenness(self, node_id: str) -> float:
        if not self._betweenness_cache:
            self.compute_centrality_metrics()
        return self._betweenness_cache.get(node_id, 0.0)

    def compute_communities(self) -> Dict[str, int]:
        """Calculates community partitions using Louvain or greedy modularity."""
        if self._communities_cache:
            return self._communities_cache

        undirected = self.get_undirected_view()
        if len(undirected) == 0:
            return {}

        try:
            # Louvain community detection from networkx.community
            communities = nx.community.louvain_communities(undirected, seed=42)
            mapping = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    mapping[node] = idx
            self._communities_cache = mapping
        except Exception:
            # Fallback to connected components
            mapping = {}
            for idx, comp in enumerate(nx.connected_components(undirected)):
                for node in comp:
                    mapping[node] = idx
            self._communities_cache = mapping

        return self._communities_cache

    def get_ego_graph(self, node_id: str, radius: int = 2) -> GraphData:
        """Extracts the k-hop ego subgraph centered around node_id."""
        if node_id not in self.graph:
            return GraphData(nodes=[], edges=[])

        undirected = self.get_undirected_view()
        ego_nodes = nx.single_source_shortest_path_length(undirected, node_id, cutoff=radius)
        sub_nodes_set = set(ego_nodes.keys())

        nodes_list: List[NodeData] = []
        for n in sub_nodes_set:
            node_obj = self.get_node(n)
            if node_obj:
                nodes_list.append(node_obj)

        edges_list: List[EdgeData] = []
        for u, v, k, data in self.graph.edges(sub_nodes_set, keys=True, data=True):
            if v in sub_nodes_set:
                edges_list.append(
                    EdgeData(
                        id=k,
                        source=u,
                        target=v,
                        type=EdgeType(data.get("type", "TRANSFERRED")),
                        amount=data.get("amount"),
                        timestamp=data.get("timestamp"),
                        risk_weight=data.get("risk_weight", 0.0),
                        is_suspicious=data.get("is_suspicious", False),
                        details=data.get("details", {}),
                    )
                )

        return GraphData(
            nodes=nodes_list,
            edges=edges_list,
            total_nodes=len(nodes_list),
            total_edges=len(edges_list),
        )

    def find_shared_entities(self, account_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Finds all accounts connected to the given account via shared Device or shared IP.
        E.g. Account A -> Device X <- Account B
             Account A -> IP 1 <- Account C
        """
        shared_devices = []
        shared_ips = []

        if account_id not in self.graph:
            return shared_devices, shared_ips

        # 1. Find all devices connected to account_id (direct or through User)
        connected_devices: Set[str] = set()
        connected_ips: Set[str] = set()

        # Check neighbors
        for neighbor in self.graph.neighbors(account_id):
            ntype = self.graph.nodes[neighbor].get("type")
            if ntype == EntityType.DEVICE.value:
                connected_devices.add(neighbor)
            elif ntype == EntityType.IP.value:
                connected_ips.add(neighbor)
            elif ntype == EntityType.USER.value:
                # Check user's devices & IPs
                for user_nbr in self.graph.neighbors(neighbor):
                    untype = self.graph.nodes[user_nbr].get("type")
                    if untype == EntityType.DEVICE.value:
                        connected_devices.add(user_nbr)
                    elif untype == EntityType.IP.value:
                        connected_ips.add(user_nbr)

        # Inbound edges (if device or IP points to account/user)
        for predecessor in self.graph.predecessors(account_id):
            ptype = self.graph.nodes[predecessor].get("type")
            if ptype == EntityType.DEVICE.value:
                connected_devices.add(predecessor)
            elif ptype == EntityType.IP.value:
                connected_ips.add(predecessor)

        # 2. For each device, find all other accounts using it
        for dev in connected_devices:
            dev_node = self.graph.nodes[dev]
            other_accounts = set()
            # Look at all nodes connected to this device
            for dev_nbr in self.get_undirected_view().neighbors(dev):
                if dev_nbr == account_id:
                    continue
                nbr_type = self.graph.nodes[dev_nbr].get("type")
                if nbr_type == EntityType.ACCOUNT.value:
                    other_accounts.add(dev_nbr)
                elif nbr_type == EntityType.USER.value:
                    # Look at accounts owned by user
                    for user_nbr in self.get_undirected_view().neighbors(dev_nbr):
                        if (
                            user_nbr != account_id
                            and self.graph.nodes[user_nbr].get("type") == EntityType.ACCOUNT.value
                        ):
                            other_accounts.add(user_nbr)

            if other_accounts:
                shared_devices.append(
                    {
                        "device_id": dev,
                        "device_label": dev_node.get("label", dev),
                        "device_details": dev_node.get("details", {}),
                        "shared_with_accounts": list(other_accounts),
                        "account_count": len(other_accounts) + 1,
                    }
                )

        # 3. For each IP, find all other accounts using it
        for ip in connected_ips:
            ip_node = self.graph.nodes[ip]
            other_accounts = set()
            for ip_nbr in self.get_undirected_view().neighbors(ip):
                if ip_nbr == account_id:
                    continue
                nbr_type = self.graph.nodes[ip_nbr].get("type")
                if nbr_type == EntityType.ACCOUNT.value:
                    other_accounts.add(ip_nbr)
                elif nbr_type == EntityType.USER.value:
                    for user_nbr in self.get_undirected_view().neighbors(ip_nbr):
                        if (
                            user_nbr != account_id
                            and self.graph.nodes[user_nbr].get("type") == EntityType.ACCOUNT.value
                        ):
                            other_accounts.add(user_nbr)

            if other_accounts:
                shared_ips.append(
                    {
                        "ip_id": ip,
                        "ip_label": ip_node.get("label", ip),
                        "ip_details": ip_node.get("details", {}),
                        "shared_with_accounts": list(other_accounts),
                        "account_count": len(other_accounts) + 1,
                    }
                )

        return shared_devices, shared_ips

    def detect_directed_cycles(self, max_cycle_length: int = 5) -> List[List[str]]:
        """Detects directed cycles in the transaction subgraph for circular layering detection."""
        # Create a simple directed transaction graph between accounts
        tx_graph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") == EdgeType.TRANSFERRED.value:
                tx_graph.add_edge(u, v)

        cycles = []
        try:
            # nx.simple_cycles yields elementary cycles
            for cycle in nx.simple_cycles(tx_graph):
                if 3 <= len(cycle) <= max_cycle_length:
                    cycles.append(cycle)
                if len(cycles) >= 50:
                    break
        except Exception:
            pass
        return cycles

    def find_paths_between(self, source_id: str, target_id: str, cutoff: int = 4) -> List[List[str]]:
        """Finds all connection paths between two entities up to cutoff hops."""
        if source_id not in self.graph or target_id not in self.graph:
            return []
        undirected = self.get_undirected_view()
        try:
            paths = list(nx.all_simple_paths(undirected, source=source_id, target=target_id, cutoff=cutoff))
            return paths[:20]
        except Exception:
            return []

    def export_graph_data(self) -> GraphData:
        """Exports full graph data formatted for visualization."""
        communities = self.compute_communities()
        nodes_list: List[NodeData] = []
        edges_list: List[EdgeData] = []
        flagged_count = 0
        total_volume = 0.0
        risk_scores = []

        for node_id, data in self.graph.nodes(data=True):
            risk = float(data.get("risk_score", 0.0))
            flagged = bool(data.get("flagged", False))
            if flagged or risk >= 0.75:
                flagged_count += 1
            risk_scores.append(risk)

            node_obj = NodeData(
                id=node_id,
                label=data.get("label", node_id),
                type=EntityType(data.get("type", "ACCOUNT")),
                risk_score=risk,
                risk_level=RiskLevel(data.get("risk_level", self._compute_risk_level(risk).value)),
                flagged=flagged,
                community_id=communities.get(node_id, 0),
                details=data.get("details", {}),
                created_at=data.get("created_at"),
            )
            nodes_list.append(node_obj)

        for u, v, k, data in self.graph.edges(keys=True, data=True):
            amt = data.get("amount")
            if amt:
                total_volume += float(amt)

            edges_list.append(
                EdgeData(
                    id=str(k),
                    source=u,
                    target=v,
                    type=EdgeType(data.get("type", "TRANSFERRED")),
                    amount=amt,
                    timestamp=data.get("timestamp"),
                    risk_weight=float(data.get("risk_weight", 0.0)),
                    is_suspicious=bool(data.get("is_suspicious", False)),
                    details=data.get("details", {}),
                )
            )

        avg_risk = sum(risk_scores) / max(1, len(risk_scores))

        return GraphData(
            nodes=nodes_list,
            edges=edges_list,
            total_nodes=len(nodes_list),
            total_edges=len(edges_list),
            flagged_nodes=flagged_count,
            total_volume_at_risk=round(total_volume, 2),
            avg_network_risk=round(avg_risk, 3),
        )
