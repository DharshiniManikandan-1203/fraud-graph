import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.core.graph_engine import FraudGraphEngine
from backend.core.rules_engine import RulesEngine
from backend.data.real_dataset_preparer import build_unified_real_dataset
from backend.data.real_dataset_loader import RealDatasetLoader
from backend.ml.feature_extractor import GraphFeatureExtractor
from backend.ml.models import MLFraudPipeline
from backend.ml.gnn import VectorizedGraphSAGE
from backend.ml.ensemble_engine import EnsembleFraudEngine
from backend.models.schema import EntityType, EdgeType


def test_build_and_load_unified_real_dataset():
    print("Testing unified real dataset preparation and loading...")
    data = build_unified_real_dataset()
    assert data["metadata"]["total_nodes"] >= 40
    assert data["metadata"]["total_edges"] >= 40
    assert data["metadata"]["fraud_nodes_count"] > 0
    assert data["metadata"]["licit_nodes_count"] > 0

    ge = FraudGraphEngine()
    loader = RealDatasetLoader(ge)
    load_res = loader.load_unified_real_dataset()

    assert len(ge.graph.nodes) >= 40
    assert len(ge.graph.edges) >= 40
    print(f"[PASS] Loaded {len(ge.graph.nodes)} nodes and {len(ge.graph.edges)} edges from real dataset.")


def test_custom_csv_ingestion():
    print("Testing custom CSV ingestion engine...")
    ge = FraudGraphEngine()
    loader = RealDatasetLoader(ge)

    sample_csv = (
        "source_account,target_account,amount,timestamp,channel,device_id,ip_address,is_fraud\n"
        "ACC_USER_1,ACC_MERCHANT_1,250.00,2024-03-01T12:00:00Z,TRANSFER,DEV_IPHONE_14,IP_73_182_10_4,0\n"
        "ACC_MULE_X,BENE_CRYPTO_MIXER,22500.00,2024-03-01T14:30:00Z,CRYPTO_BRIDGE,DEV_EMU_99,IP_185_220_1,1\n"
        "ACC_MULE_Y,BENE_CRYPTO_MIXER,19800.00,2024-03-01T14:45:00Z,CRYPTO_BRIDGE,DEV_EMU_99,IP_185_220_1,1\n"
    )

    res = loader.ingest_custom_csv(sample_csv)
    assert res["total_nodes"] >= 6
    assert res["total_edges"] >= 3
    assert ge.graph.has_node("ACC_USER_1")
    assert ge.graph.has_node("BENE_CRYPTO_MIXER")
    assert ge.graph.has_node("DEV_EMU_99")

    # Shared device check: DEV_EMU_99 shared by ACC_MULE_X and ACC_MULE_Y
    shared_devs, _ = ge.find_shared_entities("ACC_MULE_X")
    assert len(shared_devs) == 1
    assert shared_devs[0]["device_id"] == "DEV_EMU_99"
    assert "ACC_MULE_Y" in shared_devs[0]["shared_with_accounts"]
    print("[PASS] Custom CSV parsed correctly and multi-account device collisions detected.")


def test_real_dataset_ml_and_scoring_pass():
    print("Testing end-to-end feature extraction and scoring pass on real dataset...")
    ge = FraudGraphEngine()
    loader = RealDatasetLoader(ge)
    loader.load_unified_real_dataset()

    fe = GraphFeatureExtractor(ge)
    df_feat, acc_ids = fe.extract_features_matrix()
    assert len(acc_ids) > 15
    assert len(fe.FEATURE_NAMES) == 23
    assert all(f in df_feat.columns for f in fe.FEATURE_NAMES)

    rules = RulesEngine(ge)
    ml = MLFraudPipeline()
    gnn = VectorizedGraphSAGE(in_dim=23, hidden_dim=32, embed_dim=16)
    ensemble = EnsembleFraudEngine(ge, rules, ml, gnn)

    ensemble.run_full_scoring_pass()

    # Check that high-risk illicit Bitcoin/PaySim/IBM entities are scored appropriately
    btc_illicit = ge.get_node("BTC_TX_ILL_99402")
    if btc_illicit:
        assert btc_illicit.risk_score >= 0.70

    ps_mule = ge.get_node("ACC_PS_C553264065")
    if ps_mule:
        assert ps_mule.risk_score >= 0.70

    print("[PASS] Feature extraction and ensemble scoring pass completed successfully.")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING REAL DATASET UNIT TESTS")
    print("=" * 60)
    test_build_and_load_unified_real_dataset()
    test_custom_csv_ingestion()
    test_real_dataset_ml_and_scoring_pass()
    print("=" * 60)
    print("ALL REAL DATASET UNIT TESTS PASSED 100%!")
    print("=" * 60)
