from typing import List, Dict, Any, Tuple, Optional
from ..models.schema import (
    EntityType,
    EdgeType,
    RiskLevel,
    SuspiciousConnection,
    RiskFactor,
    FraudRing,
    NodeData,
    EdgeData,
)
from .graph_engine import FraudGraphEngine
import networkx as nx


class RulesEngine:
    """
    Deterministic graph topology and financial heuristic pattern matchers.
    Identifies suspicious connected communities, device/IP collision chains,
    circular wash rings, and mule networks.
    """

    def __init__(self, graph_engine: FraudGraphEngine):
        self.ge = graph_engine

    def evaluate_account_rules(self, account_id: str) -> Tuple[float, List[SuspiciousConnection], List[RiskFactor], List[str]]:
        """
        Evaluates deterministic graph pattern rules for a specific account.
        Returns:
            - base_rule_risk_score (0.0 to 1.0)
            - suspicious_connections: List of explicit chains (e.g. Account A -> Device X -> Account B)
            - risk_factors: Breakdown of score contributions
            - triggered_rules: List of rule names triggered
        """
        suspicious_connections: List[SuspiciousConnection] = []
        risk_factors: List[RiskFactor] = []
        triggered_rules: List[str] = []
        cumulative_score = 0.0

        if account_id not in self.ge.graph:
            return 0.0, [], [], []

        node_data = self.ge.graph.nodes[account_id]
        node_label = node_data.get("label", account_id)

        # 1. Device Sharing Collision Rule: Account A -> Device X -> Account B
        shared_devices, shared_ips = self.ge.find_shared_entities(account_id)

        for dev_info in shared_devices:
            dev_id = dev_info["device_id"]
            dev_label = dev_info.get("device_label", dev_id)
            shared_accounts = dev_info["shared_with_accounts"]
            account_count = dev_info["account_count"]
            dev_details = dev_info.get("device_details", {})
            is_emulator = dev_details.get("is_emulator", False)
            is_rooted = dev_details.get("is_rooted", False)

            # Build chain descriptions
            for other_acc in shared_accounts:
                other_node = self.ge.get_node(other_acc)
                other_label = other_node.label if other_node else other_acc

                severity = RiskLevel.CRITICAL if account_count >= 3 or is_emulator else RiskLevel.HIGH
                chain_str = f"{node_label} ──[ACCESSED_FROM]──► {dev_label} ◄──[ACCESSED_FROM]── {other_label}"
                
                suspicious_connections.append(
                    SuspiciousConnection(
                        chain=chain_str,
                        source_id=account_id,
                        intermediary_id=dev_id,
                        target_id=other_acc,
                        connection_type="SHARED_DEVICE",
                        description=f"Device '{dev_label}' is shared across {account_count} distinct financial accounts"
                        + (" (Emulator/Rooted detected)" if (is_emulator or is_rooted) else ""),
                        severity=severity,
                    )
                )

            # Score contribution
            dev_risk_weight = 0.35 if account_count == 2 else 0.55
            if is_emulator or is_rooted:
                dev_risk_weight += 0.20
            dev_risk_weight = min(0.85, dev_risk_weight)

            rule_name = f"RULE_DEVICE_COLLISION_{account_count}_ACCOUNTS"
            triggered_rules.append(rule_name)
            risk_factors.append(
                RiskFactor(
                    name=f"Shared Device Collision ({account_count} accounts on {dev_label})",
                    score=dev_risk_weight,
                    weight=0.35,
                    description=f"Hardware fingerprint is bound to {account_count} accounts. High probability of multi-accounting or bot farm.",
                    category="DEVICE_SHARING",
                )
            )
            cumulative_score += dev_risk_weight

        # 2. IP Address Collision Rule: Account A -> IP 1 -> Account C
        for ip_info in shared_ips:
            ip_id = ip_info["ip_id"]
            ip_label = ip_info.get("ip_label", ip_id)
            shared_accounts = ip_info["shared_with_accounts"]
            account_count = ip_info["account_count"]
            ip_details = ip_info.get("ip_details", {})
            is_vpn = ip_details.get("is_vpn", False)
            is_datacenter = ip_details.get("is_datacenter", False)
            is_tor = ip_details.get("is_tor", False)

            for other_acc in shared_accounts:
                other_node = self.ge.get_node(other_acc)
                other_label = other_node.label if other_node else other_acc

                severity = RiskLevel.CRITICAL if (is_tor or is_vpn and account_count >= 3) else RiskLevel.MEDIUM
                chain_str = f"{node_label} ──[CONNECTED_VIA]──► {ip_label} ◄──[CONNECTED_VIA]── {other_label}"

                suspicious_connections.append(
                    SuspiciousConnection(
                        chain=chain_str,
                        source_id=account_id,
                        intermediary_id=ip_id,
                        target_id=other_acc,
                        connection_type="SHARED_IP",
                        description=f"IP address '{ip_label}' shared across {account_count} accounts"
                        + (" [VPN/Datacenter Proxy]" if (is_vpn or is_datacenter) else "")
                        + (" [TOR Exit Node]" if is_tor else ""),
                        severity=severity,
                    )
                )

            ip_risk_weight = 0.20 if account_count == 2 else 0.40
            if is_vpn or is_datacenter:
                ip_risk_weight += 0.15
            if is_tor:
                ip_risk_weight += 0.35
            ip_risk_weight = min(0.80, ip_risk_weight)

            rule_name = f"RULE_IP_COLLISION_{account_count}_ACCOUNTS"
            triggered_rules.append(rule_name)
            risk_factors.append(
                RiskFactor(
                    name=f"Shared IP Network Collision ({ip_label})",
                    score=ip_risk_weight,
                    weight=0.25,
                    description=f"IP is shared across {account_count} accounts with proxy/VPN signals.",
                    category="IP_COLLISION",
                )
            )
            cumulative_score += ip_risk_weight

        # 3. Circular Money Laundering Ring Detection
        cycles = self.ge.detect_directed_cycles(max_cycle_length=6)
        participating_cycles = [c for c in cycles if account_id in c]
        if participating_cycles:
            shortest_cycle = min(participating_cycles, key=len)
            cycle_chain_labels = [self.ge.graph.nodes[n].get("label", n) for n in shortest_cycle]
            cycle_chain_str = " ──► ".join(cycle_chain_labels) + f" ──► {cycle_chain_labels[0]}"

            rule_name = f"RULE_CIRCULAR_TRANSFER_CYCLE_LEN_{len(shortest_cycle)}"
            triggered_rules.append(rule_name)
            
            suspicious_connections.append(
                SuspiciousConnection(
                    chain=cycle_chain_str,
                    source_id=shortest_cycle[0],
                    intermediary_id=shortest_cycle[1] if len(shortest_cycle) > 2 else shortest_cycle[0],
                    target_id=shortest_cycle[-1],
                    connection_type="CIRCULAR_TRANSFER",
                    description=f"Closed-loop fund rotation cycle of length {len(shortest_cycle)} (Layering / Wash Trading)",
                    severity=RiskLevel.CRITICAL,
                )
            )

            risk_factors.append(
                RiskFactor(
                    name=f"Circular Laundering Loop ({len(shortest_cycle)}-Hop Cycle)",
                    score=0.88,
                    weight=0.40,
                    description=f"Funds circulate in a closed directed cycle through {len(shortest_cycle)} accounts to obfuscate source of funds.",
                    category="GRAPH_CENTRALITY",
                )
            )
            cumulative_score += 0.85

        # 4. Money Mule Fan-In / Fan-Out Pattern
        in_tx_accounts = set()
        out_tx_accounts = set()
        in_volume = 0.0
        out_volume = 0.0

        for u, v, data in self.ge.graph.in_edges(account_id, data=True):
            if data.get("type") == EdgeType.TRANSFERRED.value:
                in_tx_accounts.add(u)
                in_volume += data.get("amount", 0.0) or 0.0

        for u, v, data in self.ge.graph.out_edges(account_id, data=True):
            if data.get("type") == EdgeType.TRANSFERRED.value:
                out_tx_accounts.add(v)
                out_volume += data.get("amount", 0.0) or 0.0

        if len(in_tx_accounts) >= 3 and len(out_tx_accounts) >= 1 and (out_volume >= 0.70 * in_volume):
            rule_name = "RULE_MULE_AGGREGATION_SMURFING"
            triggered_rules.append(rule_name)
            
            risk_factors.append(
                RiskFactor(
                    name="Money Mule Funnel (Structuring / Smurfing)",
                    score=0.82,
                    weight=0.35,
                    description=f"Fan-in from {len(in_tx_accounts)} distinct accounts (${in_volume:,.2f}) with swift drain (${out_volume:,.2f}) to downstream entities.",
                    category="VELOCITY",
                )
            )
            cumulative_score += 0.80

        # 5. Neighborhood Contagion: Check if connected to already flagged accounts
        flagged_neighbors = 0
        total_neighbors = 0
        for nbr in self.ge.get_undirected_view().neighbors(account_id):
            total_neighbors += 1
            nbr_node = self.ge.graph.nodes[nbr]
            if nbr_node.get("flagged") or float(nbr_node.get("risk_score", 0.0)) >= 0.75:
                flagged_neighbors += 1

        if flagged_neighbors > 0:
            contagion_ratio = flagged_neighbors / max(1, total_neighbors)
            score_contrib = min(0.75, contagion_ratio * 0.70 + 0.15)
            triggered_rules.append("RULE_HIGH_RISK_NEIGHBORHOOD_CONTAGION")
            risk_factors.append(
                RiskFactor(
                    name="High-Risk Neighborhood Contagion",
                    score=score_contrib,
                    weight=0.30,
                    description=f"{flagged_neighbors} of {total_neighbors} directly connected entities are flagged high risk ({contagion_ratio:.0%}).",
                    category="GRAPH_CENTRALITY",
                )
            )
            cumulative_score += score_contrib

        # Calibrate base rule score between 0.0 and 1.0 (saturating sigmoid-like combination)
        base_rule_risk_score = 1.0 - (1.0 / (1.0 + cumulative_score * 0.85)) if cumulative_score > 0 else 0.05
        return base_rule_risk_score, suspicious_connections, risk_factors, triggered_rules

    def detect_all_fraud_rings(self) -> List[FraudRing]:
        """
        Scans the entire graph to discover and cluster suspicious connected communities
        (Device Farms, IP Clusters, Circular Wash Rings, Mule Networks).
        """
        detected_rings: List[FraudRing] = []
        ring_id_counter = 1

        # 1. Device Farm Rings (Devices shared by >= 2 accounts)
        device_nodes = [n for n, d in self.ge.graph.nodes(data=True) if d.get("type") == EntityType.DEVICE.value]
        for dev in device_nodes:
            dev_data = self.ge.graph.nodes[dev]
            # Find accounts using this device
            connected_accs = set()
            for nbr in self.ge.get_undirected_view().neighbors(dev):
                if self.ge.graph.nodes[nbr].get("type") == EntityType.ACCOUNT.value:
                    connected_accs.add(nbr)
                elif self.ge.graph.nodes[nbr].get("type") == EntityType.USER.value:
                    for u_nbr in self.ge.get_undirected_view().neighbors(nbr):
                        if self.ge.graph.nodes[u_nbr].get("type") == EntityType.ACCOUNT.value:
                            connected_accs.add(u_nbr)

            if len(connected_accs) >= 2:
                all_member_ids = list(connected_accs) + [dev]
                # Gather subgraph nodes and edges
                sub_nodes = [self.ge.get_node(m) for m in all_member_ids if self.ge.get_node(m) is not None]
                sub_edges = []
                total_vol = 0.0
                for u in all_member_ids:
                    for v in all_member_ids:
                        if self.ge.graph.has_edge(u, v):
                            for k, ed in self.ge.graph[u][v].items():
                                amt = ed.get("amount")
                                if amt:
                                    total_vol += amt
                                sub_edges.append(
                                    EdgeData(
                                        id=str(k),
                                        source=u,
                                        target=v,
                                        type=EdgeType(ed.get("type", "ACCESSED_FROM")),
                                        amount=amt,
                                        timestamp=ed.get("timestamp"),
                                        risk_weight=0.85,
                                        is_suspicious=True,
                                        details=ed.get("details", {}),
                                    )
                                )

                avg_risk = sum(n.risk_score for n in sub_nodes) / max(1, len(sub_nodes))
                risk_score = max(0.85, avg_risk)

                detected_rings.append(
                    FraudRing(
                        id=f"RING-DEV-{ring_id_counter:03d}",
                        name=f"Device Farm Cluster #{ring_id_counter} ({dev_data.get('label', dev)})",
                        typology="DEVICE_FARM",
                        severity=RiskLevel.CRITICAL if len(connected_accs) >= 3 else RiskLevel.HIGH,
                        risk_score=round(risk_score, 2),
                        member_count=len(sub_nodes),
                        member_ids=all_member_ids,
                        total_volume=round(total_vol, 2),
                        description=f"Device fingerprint '{dev_data.get('label', dev)}' shared across {len(connected_accs)} distinct accounts. High Sybil/multi-accounting indicator.",
                        detected_rules=["RULE_DEVICE_COLLISION_FARM", "RULE_SHARED_HARDWARE_FINGERPRINT"],
                        nodes=sub_nodes,
                        edges=sub_edges,
                    )
                )
                ring_id_counter += 1

        # 2. Circular Wash Trading Rings
        cycles = self.ge.detect_directed_cycles(max_cycle_length=6)
        seen_cycles_sets = []
        for cycle in cycles:
            cycle_set = set(cycle)
            if cycle_set in seen_cycles_sets:
                continue
            seen_cycles_sets.append(cycle_set)

            sub_nodes = [self.ge.get_node(m) for m in cycle if self.ge.get_node(m) is not None]
            sub_edges = []
            total_vol = 0.0
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if self.ge.graph.has_edge(u, v):
                    for k, ed in self.ge.graph[u][v].items():
                        amt = ed.get("amount")
                        if amt:
                            total_vol += amt
                        sub_edges.append(
                            EdgeData(
                                id=str(k),
                                source=u,
                                target=v,
                                type=EdgeType(ed.get("type", "TRANSFERRED")),
                                amount=amt,
                                timestamp=ed.get("timestamp"),
                                risk_weight=0.92,
                                is_suspicious=True,
                                details=ed.get("details", {}),
                            )
                        )

            detected_rings.append(
                FraudRing(
                    id=f"RING-CYCLE-{ring_id_counter:03d}",
                    name=f"Circular Laundering Ring #{ring_id_counter} ({len(cycle)} Hops)",
                    typology="CIRCULAR_WASH",
                    severity=RiskLevel.CRITICAL,
                    risk_score=0.94,
                    member_count=len(sub_nodes),
                    member_ids=cycle,
                    total_volume=round(total_vol, 2),
                    description=f"Detected closed directed fund transfer loop among {len(cycle)} accounts (${total_vol:,.2f} volume). Layering typology.",
                    detected_rules=["RULE_DIRECTED_CYCLE_LAYERING", "RULE_CLOSED_LOOP_TRANSFER"],
                    nodes=sub_nodes,
                    edges=sub_edges,
                )
            )
            ring_id_counter += 1

        # 3. Money Mule Aggregation Hubs
        for node_id, data in self.ge.graph.nodes(data=True):
            if data.get("type") != EntityType.ACCOUNT.value:
                continue
            in_edges = [e for e in self.ge.graph.in_edges(node_id, data=True) if e[2].get("type") == EdgeType.TRANSFERRED.value]
            out_edges = [e for e in self.ge.graph.out_edges(node_id, data=True) if e[2].get("type") == EdgeType.TRANSFERRED.value]
            
            if len(in_edges) >= 3 and len(out_edges) >= 1:
                in_sources = [e[0] for e in in_edges]
                out_targets = [e[1] for e in out_edges]
                all_mule_nodes = list(set(in_sources + [node_id] + out_targets))
                sub_nodes = [self.ge.get_node(m) for m in all_mule_nodes if self.ge.get_node(m) is not None]
                
                total_vol = sum(e[2].get("amount", 0.0) or 0.0 for e in in_edges)
                sub_edges = []
                for e in in_edges + out_edges:
                    sub_edges.append(
                        EdgeData(
                            id=f"mule_{e[0]}_{e[1]}",
                            source=e[0],
                            target=e[1],
                            type=EdgeType.TRANSFERRED,
                            amount=e[2].get("amount"),
                            timestamp=e[2].get("timestamp"),
                            risk_weight=0.88,
                            is_suspicious=True,
                        )
                    )

                detected_rings.append(
                    FraudRing(
                        id=f"RING-MULE-{ring_id_counter:03d}",
                        name=f"Mule Aggregator Network #{ring_id_counter} ({data.get('label', node_id)})",
                        typology="MULE_NETWORK",
                        severity=RiskLevel.CRITICAL,
                        risk_score=0.91,
                        member_count=len(sub_nodes),
                        member_ids=all_mule_nodes,
                        total_volume=round(total_vol, 2),
                        description=f"Central mule account receives rapid deposits from {len(in_sources)} distinct victim accounts and funnels out to {len(out_targets)} destination endpoints.",
                        detected_rules=["RULE_MULE_FAN_IN_AGGREGATION", "RULE_RAPID_DISPERSION"],
                        nodes=sub_nodes,
                        edges=sub_edges,
                    )
                )
                ring_id_counter += 1

        return detected_rings
