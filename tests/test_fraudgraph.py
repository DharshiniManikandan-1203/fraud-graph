import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from backend.core.graph_engine import FraudGraphEngine
from backend.core.rules_engine import RulesEngine
from backend.ml.feature_extractor import GraphFeatureExtractor
from backend.ml.models import MLFraudPipeline
from backend.ml.gnn import VectorizedGraphSAGE
from backend.ml.ensemble_engine import EnsembleFraudEngine
from backend.data.generator import SyntheticGraphGenerator
from backend.models.schema import EntityType, EdgeType, RiskLevel


def test_graph_engine_basic():
    ge = FraudGraphEngine()
    ge.add_node("ACC_1", EntityType.ACCOUNT, label="Account 1", risk_score=0.10)
    ge.add_node("DEV_1", EntityType.DEVICE, label="Device 1", risk_score=0.05)
    ge.add_edge("ACC_1", "DEV_1", EdgeType.ACCESSED_FROM)

    assert len(ge.graph.nodes) == 2
    assert len(ge.graph.edges) == 1
    assert ge.get_node("ACC_1") is not None
    assert ge.get_node("ACC_1").label == "Account 1"


def test_shared_entity_detection():
    ge = FraudGraphEngine()
    # Account A -> Device X <- Account B
    ge.add_node("ACC_A", EntityType.ACCOUNT, label="Account A", risk_score=0.80)
    ge.add_node("ACC_B", EntityType.ACCOUNT, label="Account B", risk_score=0.85)
    ge.add_node("DEV_X", EntityType.DEVICE, label="Device X", risk_score=0.90, is_emulator=True)

    ge.add_edge("ACC_A", "DEV_X", EdgeType.ACCESSED_FROM)
    ge.add_edge("ACC_B", "DEV_X", EdgeType.ACCESSED_FROM)

    shared_devs, shared_ips = ge.find_shared_entities("ACC_A")
    assert len(shared_devs) == 1
    assert shared_devs[0]["device_id"] == "DEV_X"
    assert "ACC_B" in shared_devs[0]["shared_with_accounts"]
    assert shared_devs[0]["account_count"] == 2


def test_circular_cycle_detection():
    ge = FraudGraphEngine()
    # Directed cycle: 1 -> 2 -> 3 -> 1
    ge.add_node("A1", EntityType.ACCOUNT)
    ge.add_node("A2", EntityType.ACCOUNT)
    ge.add_node("A3", EntityType.ACCOUNT)

    ge.add_edge("A1", "A2", EdgeType.TRANSFERRED, amount=1000.0)
    ge.add_edge("A2", "A3", EdgeType.TRANSFERRED, amount=950.0)
    ge.add_edge("A3", "A1", EdgeType.TRANSFERRED, amount=900.0)

    cycles = ge.detect_directed_cycles(max_cycle_length=4)
    assert len(cycles) >= 1
    assert len(cycles[0]) == 3


def test_rules_engine_evaluation():
    ge = FraudGraphEngine()
    rules = RulesEngine(ge)

    # Setup device collision
    ge.add_node("ACC_A", EntityType.ACCOUNT, label="Account A")
    ge.add_node("ACC_B", EntityType.ACCOUNT, label="Account B")
    ge.add_node("DEV_X", EntityType.DEVICE, label="Device X", is_emulator=True)

    ge.add_edge("ACC_A", "DEV_X", EdgeType.ACCESSED_FROM)
    ge.add_edge("ACC_B", "DEV_X", EdgeType.ACCESSED_FROM)

    score, conns, factors, triggered = rules.evaluate_account_rules("ACC_A")
    assert score > 0.30
    assert len(conns) >= 1
    assert any("DEVICE_COLLISION" in r for r in triggered)


def test_full_scenario_generation_and_ensemble():
    ge = FraudGraphEngine()
    rules = RulesEngine(ge)
    ml = MLFraudPipeline()
    gnn = VectorizedGraphSAGE()
    ensemble = EnsembleFraudEngine(ge, rules, ml, gnn)
    gen = SyntheticGraphGenerator(ge)

    gen.generate_full_enterprise_scenario()
    assert len(ge.graph.nodes) > 25
    assert len(ge.graph.edges) > 30

    ensemble.run_full_scoring_pass()

    # Check Account A in device farm ring
    acc_a = ge.get_node("ACC_FARM_A")
    assert acc_a is not None
    assert acc_a.risk_score >= 0.75
    assert acc_a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    # Check detected fraud rings
    rings = rules.detect_all_fraud_rings()
    assert len(rings) >= 2
    ring_types = [r.typology for r in rings]
    assert "DEVICE_FARM" in ring_types or "CIRCULAR_WASH" in ring_types

    # Check node profile
    profile = ensemble.get_full_node_profile("ACC_FARM_A")
    assert profile is not None
    assert len(profile.suspicious_connections) >= 1
    assert profile.sar_summary is not None
    assert "SAR" in profile.sar_summary


if __name__ == "__main__":
    print("Running unit tests...")
    test_graph_engine_basic()
    test_shared_entity_detection()
    test_circular_cycle_detection()
    test_rules_engine_evaluation()
    test_full_scenario_generation_and_ensemble()
    print("ALL TESTS PASSED SUCCESSFULLY!")
