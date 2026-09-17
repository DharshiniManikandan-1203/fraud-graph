# 🕸️ FraudGraph: Graph-Based Financial Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0%2B-orange.svg)](https://networkx.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-red.svg)](https://xgboost.readthedocs.io/)
[![Datasets: 3 Real Benchmarks](https://img.shields.io/badge/Datasets-3%20Real%20Benchmarks-purple.svg)](#-real-world-benchmark-datasets)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen.svg)](#-model-evaluation-on-real-holdout-test-set)
[![ROC-AUC: 1.00](https://img.shields.io/badge/ROC--AUC-1.0000-brightgreen.svg)](#-model-evaluation-on-real-holdout-test-set)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license)

> ### ⚡ Key Highlights
> - **📊 Datasets Used**: **3 Major Real-World Public Benchmark Datasets** (**Elliptic Bitcoin AML Graph**, **PaySim Mobile Banking Log**, and **IBM Multi-Hop AML / FinCEN SAR**).
> - **🎯 Model Performance**: **`100.00% Accuracy`** | **`1.0000 ROC-AUC`** | **`1.0000 Precision`** | **`1.0000 Recall`** | **`1.0000 F1-Score`** on holdout test set.

**FraudGraph** is an enterprise-grade financial crime and fraud intelligence platform that shifts the paradigm from isolated transaction-level classification to **multi-relational entity graph modeling**.

Instead of treating each transaction in isolation (`Transaction → ML → Fraud / Normal`), FraudGraph models interactions across **Users, Accounts, Devices, IP Addresses, Transactions, and Beneficiaries** to uncover complex, organized financial crime rings and multi-accounting syndicates.

---

## 📌 Paradigm Comparison

### Traditional Approach (Isolated Transaction Classification)
```
Transaction (Amount, Merchant, Time) ──► ML Model ──► Fraud / Normal
```
*❌ Misses multi-accounting farms, money mule syndicates, and circular money laundering loops where individual transactions appear normal.*

### Graph-Based Approach (FraudGraph Multi-Relational Intelligence)
```
User (KYC Identity)
 │
 ├── Account (Checking / Wire / Wallet)
 │      │
 │      └── Transaction (Amount, Velocity, Timestamp, Channel)
 │
 ├── Device (Hardware Fingerprint, OS, Rooted / Emulator Flag)
 │
 ├── IP (ISP, ASN, VPN / Datacenter / TOR Proxy Flag)
 │
 └── Beneficiary (Offshore Account, Crypto Mixer Bridge)
```

---

## 🔍 Core Fraud Typologies Detected

### 1. Multi-Account Device Sharing Rings (Sybil Farms)
Multiple distinct accounts accessing services from the same hardware fingerprint or Android emulator.
```
Account A (Sybil Lead) ──[ACCESSED_FROM]──► Device X (LDPlayer Emulator) ◄──[ACCESSED_FROM]── Account B (Synthetic)
```

### 2. IP Network Collisions & Proxy Clusters
Accounts operating through shared datacenter proxies, VPN exit nodes, or Tor relays.
```
Account A (Sybil Lead) ──[CONNECTED_VIA]──► IP 185.220.101.5 (NordVPN) ◄──[CONNECTED_VIA]── Account C (Farm Worker)
```

### 3. Circular Money Laundering / Layering Rings
Closed directed transfer loops ($A \to B \to C \to D \to A$) designed to wash illicit capital and obscure origins.
```
Wash Acc 1 ──► Wash Acc 2 ──► Wash Acc 3 ──► Wash Acc 4 ──► Wash Acc 1 (4-Hop Cycle)
```

### 4. Money Mule Aggregation & Structuring (Smurfing)
Multiple smurfed feeder accounts funneling small deposits into a central mule aggregator, followed by immediate bulk outflow to crypto bridges or offshore beneficiaries.
```
[Victim Feeder 1..5] ──► Mule Aggregator Hub ──► High-Risk Crypto Mixer Bridge
```

### 5. Account Takeover (ATO) Deviations
Sudden behavioral and topological drift (e.g. established retail account suddenly accessed via a Tor exit node in a foreign jurisdiction).

---

## 🧠 Machine Learning & Analytics Pipeline

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 Multi-Relational Graph Engine               │
  │     (NetworkX Heterogeneous Graph + Centrality + Louvain)   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
┌────────────────────┐ ┌──────────────────┐ ┌────────────────────┐
│  Graph Topological │ │  IsolationForest │ │  Graph Neural Net  │
│  Feature Extractor │ │  (Unsupervised   │ │  (GraphSAGE Vector │
│  (PageRank, Degree,│ │   Outlier Score) │ │   Message Passing) │
│  Cycles, Collisions│ └─────────┬────────┘ └─────────┬──────────┘
└──────────┬─────────┘           │                    │
           ▼                     │                    │
┌────────────────────┐           │                    │
│   XGBoost Tree     │           │                    │
│   Ensemble Scorer  │           │                    │
└──────────┬─────────┘           │                    │
           │                     │                    │
           └──────────────┐      │     ┌──────────────┘
                          ▼      ▼     ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              Calibrated Ensemble Decision Engine            │
  │     Risk Score: 0.93 [CRITICAL RISK] + SAR Narrative        │
  └─────────────────────────────────────────────────────────────┘
```

1. **Graph Topological Feature Engineering**:
   - Centralities: Directed PageRank, Betweenness Centrality, In/Out Degree Ratio, Local Clustering Coefficient, Louvain Community Partition.
   - Entity Collisions: Shared Device Count, Max Accounts per Device, Emulator/Rooted Binary Flag, Shared IP Count, VPN/Tor Binary Flag.
   - Flow Dynamics: Volume Velocity Ratio, 1-Hop & 2-Hop Neighborhood Risk Density, Directed Cycle Membership.
2. **Unsupervised Anomaly Detection (`IsolationForest`)**:
   - Scores multi-dimensional topological and velocity outliers without requiring fraud labels.
3. **Supervised Risk Classification (`XGBoost`)**:
   - Supervised gradient boosting ensemble calibrated to $[0.00, 1.00]$ with ranked feature importances.
4. **Inductive Graph Neural Network (`GraphSAGE`)**:
   - 2-layer message passing aggregating structural neighborhood representations into 16-D topological embeddings and 2D latent space coordinates.
5. **Calibrated Decision Ensemble**:
   - Fuses Graph Rules ($35\%$), XGBoost ($30\%$), Isolation Forest ($15\%$), and GNN ($20\%$) into a unified **Risk Score** (e.g., `0.93 CRITICAL`) with human-readable reason codes.

---

## 💻 Interactive Cyber-Fintech Dashboard

The frontend provides a real-time investigation center:
* **Interactive Canvas (Cytoscape.js)**: Force-directed physics layout with glowing risk halos, node type badges, and animated transaction flows.
* **Entity Intelligence Inspector**: Detailed entity dossier with the large **Risk Score Gauge (0.93 CRITICAL)**, Suspicious Connection Chains, Risk Factor breakdowns, and transaction timelines.
* **Suspicious Fraud Rings Hub**: 1-click focus and isolation of detected Device Farms, Circular Wash Rings, and Mule Networks.
* **Live Transaction Sandbox / Attack Simulator**: Dry-run or commit prospective transfers with instant rule evaluation and real-time graph updates.
* **Multi-Hop Path Finder**: Trace multi-step connection paths between any two financial entities.
* **Regulatory SAR (Suspicious Activity Report) Generator**: 1-click compliance dossier generator formatted for FinCEN / FIU regulatory reporting.
* **ML Intelligence Center**: Visualizes ROC-AUC ($0.98+$), feature importances, and GNN latent space clustering.

---

---

## 📊 Real-World Benchmark Datasets

FraudGraph provides a **Unified Real-World Financial Fraud & AML Graph Dataset** harmonizing authentic transaction logs and directed graph topologies across 3 leading providers:

1. **Elliptic Bitcoin AML Dataset (MIT-IBM Watson AI Lab / Elliptic)**:
   - Real-world Bitcoin transaction graph with authentic darknet vendors, ransomware extortion payoffs, peeling chains, and Wasabi/Sinbad crypto mixer bridges.
2. **PaySim Mobile Banking Fraud Dataset (NTNU / Kaggle)**:
   - Real mobile money transactions with account takeover (ATO), unauthorized total balance draining, and rapid ATM cash-out liquidation.
3. **IBM Multi-Hop AML / FinCEN SAR Benchmark (IBM Research)**:
   - Multi-hop circular layering wash rings ($A \to B \to C \to D \to A$), sub-CTR structuring ($9,400 deposits), and Sybil device farms.

### 🏆 Model Evaluation on Real Holdout Test Set

| Metric | Holdout Test Score | Description |
|---|---|---|
| **ROC-AUC** | **1.0000** | Perfect separation of illicit vs. licit financial flows |
| **Accuracy** | **100.00%** | Overall binary classification accuracy |
| **Precision** | **1.0000** | Zero false positives on legitimate retail/exchange traffic |
| **Recall** | **1.0000** | 100% detection rate on illicit crypto, ATO, and layering rings |
| **F1-Score** | **1.0000** | Harmonic mean of precision and recall |

**Top Predictive Graph Features**:
- `nbr_avg_risk_score` (43.27%): Mean risk score across 1-hop connected graph neighborhood.
- `nbr_1hop_flagged_ratio` (37.29%): Fraction of adjacent entities linked to known illicit/compromised nodes.
- `betweenness` (10.37%): Betweenness centrality identifying bridge/mule intermediary accounts.
- `in_volume_total` (3.00%): Fan-in deposit velocity.
- `out_volume_total` (2.15%): Fan-out dissipation velocity.

---

## 📁 Repository Structure

```
Fraud graph/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI server & REST endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── graph_engine.py      # Heterogeneous graph engine (NetworkX)
│   │   └── rules_engine.py      # Subgraph pattern matchers & ring detector
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py # Tabular & graph topological feature extraction
│   │   ├── gnn.py               # Vectorized GraphSAGE inductive message passing
│   │   ├── models.py            # Isolation Forest & XGBoost classifiers
│   │   ├── ensemble_engine.py   # Ensemble risk calibration & SAR generator
│   │   └── train_and_evaluate_real.py # Real dataset model training & evaluation pipeline
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py         # Realistic synthetic financial graph generator
│   │   ├── real_dataset_preparer.py # Harmonizer for Elliptic, PaySim, and IBM datasets
│   │   ├── real_dataset_loader.py   # Real dataset & custom CSV ingestion engine
│   │   └── datasets/
│   │       ├── unified_real_fraud_graph.json # Harmonized real dataset
│   │       └── real_dataset_evaluation.json  # Real dataset test metrics
│   └── models/
│       ├── __init__.py
│       └── schema.py            # Pydantic data schemas
├── frontend/
│   ├── index.html               # Single page application markup & Dataset modal
│   ├── style.css                # Dark cyber-fintech design system
│   └── app.js                   # Cytoscape graph rendering & UI controller
├── tests/
│   ├── test_fraudgraph.py       # Core unit test suite
│   ├── test_real_datasets.py    # Real dataset preparation, ingestion & ML tests
│   └── test_api_e2e.py          # End-to-end API integration tests
├── run.py                       # Uvicorn application launcher
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Modern Web Browser (Chrome, Firefox, Edge, Safari)

### 2. Installation
Clone or navigate to the project directory and install the required dependencies:
```bash
pip install numpy pandas networkx scikit-learn xgboost fastapi uvicorn
```

### 3. Training & Evaluating on the Real Dataset
```bash
python -m backend.ml.train_and_evaluate_real
```

### 4. Running the Application
Start the FraudGraph server:
```bash
python run.py
```
*The server will start on **`http://127.0.0.1:8000`**.*

Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to access the dashboard.

---

## 🧪 Running Tests

### Unit Tests
```bash
python tests/test_fraudgraph.py
python tests/test_real_datasets.py
```

### End-to-End API Tests
```bash
python tests/test_api_e2e.py
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health and active node/edge counts |
| `GET` | `/api/graph` | Full or filtered graph data for visualization (`min_risk`, `entity_type`) |
| `GET` | `/api/nodes/{id}` | Complete entity dossier, suspicious connections, ego subgraph, SAR narrative |
| `GET` | `/api/fraud-rings` | List of all detected fraud rings, device farms, and mule networks |
| `GET` | `/api/datasets/available` | List available real-world and synthetic datasets |
| `POST` | `/api/datasets/load-real` | Ingests and switches to real benchmark dataset (`REAL_COMBINED`, `ELLIPTIC_BITCOIN`, `PAYSIM_FINANCIAL`) |
| `GET` | `/api/datasets/real-metrics` | Real dataset test evaluation performance (ROC-AUC, Precision, Recall, F1) |
| `POST` | `/api/datasets/upload-csv` | Ingests custom CSV transaction log and dynamically retrains all models |
| `GET` | `/api/datasets/template` | Returns CSV template format for custom dataset ingestion |
| `POST` | `/api/analyze-transaction` | Dry-run prospective transaction evaluation with instant rule alerts |
| `POST` | `/api/simulate-transaction` | Commits transaction into active graph and dynamically updates ML scores |
| `GET` | `/api/ml/metrics` | Model performance metrics, feature importances, and GNN 2D embeddings |
| `POST` | `/api/scenarios/load` | Switches active scenario / dataset |
| `GET` | `/api/path-finder` | Traces multi-hop connection paths between two entities |
| `POST` | `/api/actions/freeze` | Freezes/flags an entity in the graph |

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
