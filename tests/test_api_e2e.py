import urllib.request
import json
import sys

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read().decode("utf-8")

def post(path, payload):
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data_bytes,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read().decode("utf-8")

def run_e2e_tests():
    print("--- 1. Testing Root HTML & Static Assets ---")
    status, html = get("/")
    assert status == 200
    assert "<title>FraudGraph" in html
    assert 'id="cy"' in html
    assert 'id="inspector-panel"' in html
    print("[PASS] Root HTML served correctly")

    status, css = get("/static/style.css")
    assert status == 200 and "--bg-main" in css
    print("[PASS] Static CSS served")

    status, js = get("/static/app.js")
    assert status == 200 and "FraudGraphApp" in js
    print("[PASS] Static JS served")

    print("\n--- 2. Testing /api/health ---")
    status, body = get("/api/health")
    assert status == 200
    health = json.loads(body)
    assert health["status"] == "healthy"
    print(f"[PASS] Health check: {health['nodes_count']} nodes, {health['edges_count']} edges")

    print("\n--- 3. Testing /api/graph ---")
    status, body = get("/api/graph")
    assert status == 200
    graph = json.loads(body)
    assert graph["total_nodes"] > 20
    assert graph["total_edges"] > 25
    print(f"[PASS] Full graph: {graph['total_nodes']} nodes, {graph['total_edges']} edges, Flagged: {graph['flagged_nodes']}, Avg Risk: {graph['avg_network_risk']}")

    print("\n--- 4. Testing /api/nodes/ACC_FARM_A (Target Archetype) ---")
    status, body = get("/api/nodes/ACC_FARM_A")
    assert status == 200
    node_dossier = json.loads(body)
    assert node_dossier["risk_score"] >= 0.85
    assert node_dossier["risk_level"] in ["HIGH", "CRITICAL"]
    print(f"[PASS] Account A Risk Score: {node_dossier['risk_score']} [{node_dossier['risk_level']}]")
    print("Suspicious Connection Chains:")
    for conn in node_dossier["suspicious_connections"]:
        print(f"  • {conn['chain']}")
        print(f"    └─ {conn['description']}")
    assert len(node_dossier["suspicious_connections"]) >= 2
    assert any("Device X" in c["chain"] for c in node_dossier["suspicious_connections"])
    assert any("IP" in c["chain"] for c in node_dossier["suspicious_connections"])

    print("\n--- 5. Testing /api/fraud-rings ---")
    status, body = get("/api/fraud-rings")
    assert status == 200
    rings = json.loads(body)
    print(f"[PASS] Detected {len(rings)} Fraud Rings:")
    for r in rings:
        print(f"  • [{r['id']}] {r['name']} ({r['typology']} | Risk: {r['risk_score']} | {r['member_count']} members)")
    assert len(rings) >= 2

    print("\n--- 6. Testing /api/analyze-transaction (Dry-Run Scoring) ---")
    tx_req = {
        "source_account": "ACC_FARM_A",
        "target_account_or_beneficiary": "BENE_CRYPTO_SWAP",
        "amount": 12500.0,
        "device_id": "DEV_FARM_X99",
        "ip_address": "IP_185_220_101_5",
        "channel": "CRYPTO_BRIDGE"
    }
    status, body = post("/api/analyze-transaction", tx_req)
    assert status == 200
    tx_res = json.loads(body)
    print(f"[PASS] Simulated Tx Risk: {tx_res['risk_score']} [{tx_res['risk_level']}] => Decision: {tx_res['decision']}")
    print(f"       Rules Triggered: {tx_res['triggered_rules']}")
    assert tx_res["risk_score"] >= 0.80
    assert tx_res["decision"] in ["REJECT", "FREEZE_ACCOUNT"]

    print("\n--- 7. Testing /api/ml/metrics ---")
    status, body = get("/api/ml/metrics")
    assert status == 200
    ml_data = json.loads(body)
    metrics = ml_data["metrics"]
    print(f"[PASS] ML Metrics: ROC-AUC={metrics.get('roc_auc')}, Precision={metrics.get('precision')}, Recall={metrics.get('recall')}, F1={metrics.get('f1_score')}")
    print("Top Feature Importances:")
    for fi in ml_data["feature_importances"][:5]:
        print(f"  • {fi['feature']}: {fi['importance']}%")
    assert len(ml_data["gnn_2d_embeddings"]) > 0
    print(f"[PASS] GNN 2D Embeddings computed for {len(ml_data['gnn_2d_embeddings'])} accounts")

    print("\n--- 8. Testing /api/path-finder ---")
    status, body = get("/api/path-finder?source_id=ACC_FARM_A&target_id=ACC_FARM_C")
    assert status == 200
    path_data = json.loads(body)
    print(f"[PASS] Discovered {path_data['total_paths_found']} connection pathways between ACC_FARM_A and ACC_FARM_C:")
    for p in path_data["paths"][:3]:
        print(f"  • {p['path_labels']} ({p['hop_count']} hops)")
    assert path_data["total_paths_found"] >= 1

    print("\n--- 9. Testing /api/datasets/available ---")
    status, body = get("/api/datasets/available")
    assert status == 200
    datasets_list = json.loads(body)
    assert len(datasets_list) >= 3
    print(f"[PASS] Available Datasets: {[d['id'] for d in datasets_list]}")

    print("\n--- 10. Testing /api/datasets/real-metrics ---")
    status, body = get("/api/datasets/real-metrics")
    assert status == 200
    real_metrics = json.loads(body)
    assert "test_metrics" in real_metrics
    tm = real_metrics["test_metrics"]
    print(f"[PASS] Real Dataset Test Metrics: ROC-AUC={tm['roc_auc']}, Accuracy={tm['accuracy']}, Precision={tm['precision']}, Recall={tm['recall']}, F1={tm['f1_score']}")

    print("\n--- 11. Testing /api/datasets/load-real (Unified Real Dataset Ingestion) ---")
    status, body = post("/api/datasets/load-real?dataset_name=REAL_COMBINED", {})
    assert status == 200
    real_load = json.loads(body)
    assert real_load["total_nodes"] >= 40
    print(f"[PASS] Loaded Real Dataset: {real_load['total_nodes']} nodes, {real_load['total_edges']} edges, Flagged: {real_load['flagged_nodes']}")

    print("\n--- 12. Testing /api/datasets/upload-csv (Custom Transaction Ingestion) ---")
    custom_csv = "source_account,target_account,amount,timestamp,channel,device_id,ip_address,is_fraud\nACC_T1,ACC_T2,500.00,2024-03-01T10:00:00Z,TRANSFER,DEV_1,IP_1,0\nACC_M1,BENE_OFFSHORE,45000.00,2024-03-01T11:00:00Z,WIRE,DEV_1,IP_1,1\n"
    status, body = post("/api/datasets/upload-csv", {"csv_content": custom_csv})
    assert status == 200
    upload_res = json.loads(body)
    assert upload_res["total_nodes"] >= 4
    print(f"[PASS] Custom CSV Ingested: {upload_res['total_nodes']} nodes, {upload_res['total_edges']} edges")

    print("\n=======================================================")
    print("ALL END-TO-END E2E API AND SYSTEM TESTS PASSED 100%!")
    print("=======================================================")

if __name__ == "__main__":
    run_e2e_tests()

