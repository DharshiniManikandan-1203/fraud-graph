# 🕸️ FraudGraph: Graph-Based Financial Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0%2B-orange.svg)](https://networkx.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-red.svg)](https://xgboost.readthedocs.io/)
[![Cytoscape.js](https://img.shields.io/badge/Cytoscape.js-Interactive%20Graph-brightgreen.svg)](https://js.cytoscape.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](#license)

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
│   │   └── ensemble_engine.py   # Ensemble risk calibration & SAR generator
│   ├── data/
│   │   ├── __init__.py
│   │   └── generator.py         # Realistic synthetic financial graph generator
│   └── models/
│       ├── __init__.py
│       └── schema.py            # Pydantic data schemas
├── frontend/
│   ├── index.html               # Single page application markup
│   ├── style.css                # Dark cyber-fintech design system
│   └── app.js                   # Cytoscape graph rendering & UI controller
├── tests/
│   ├── test_fraudgraph.py       # Core unit test suite
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

### 3. Running the Application
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
```
*Verifies graph operations, cycle detection, feature extraction, ML training, and rules engine.*

### End-to-End API Tests
```bash
python tests/test_api_e2e.py
```
*Verifies all REST API endpoints, real-time transaction simulation, and target archetype validation.*

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health and active node/edge counts |
| `GET` | `/api/graph` | Full or filtered graph data for visualization (`min_risk`, `entity_type`) |
| `GET` | `/api/nodes/{id}` | Complete entity dossier, suspicious connections, ego subgraph, SAR narrative |
| `GET` | `/api/fraud-rings` | List of all detected fraud rings, device farms, and mule networks |
| `POST` | `/api/analyze-transaction` | Dry-run prospective transaction evaluation with instant rule alerts |
| `POST` | `/api/simulate-transaction` | Commits transaction into active graph and dynamically updates ML scores |
| `GET` | `/api/ml/metrics` | Model performance metrics, feature importances, and GNN 2D embeddings |
| `POST` | `/api/scenarios/load` | Switches active scenario (`ENTERPRISE`, `DEVICE_FARM`, `CIRCULAR_WASH`, etc.) |
| `GET` | `/api/path-finder` | Traces multi-hop connection paths between two entities |
| `POST` | `/api/actions/freeze` | Freezes/flags an entity in the graph |

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
