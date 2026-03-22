# Hybrid AMM+ML Vulnerability Risk Assessment System

A **six-layer architecture** that couples **Absorbing Markov Models (AMM)** with **Machine Learning** through a bidirectional feedback loop to perform continuous vulnerability risk assessment — designed for **air-gapped** and **internet-connected** environments alike.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Data Import](#data-import)
  - [Daily Batch Pipeline](#daily-batch-pipeline)
  - [On-Demand Assessment](#on-demand-assessment)
  - [Web Dashboard](#web-dashboard)
  - [Backtesting](#backtesting)
  - [Data Freshness](#data-freshness)
- [Air-Gapped Deployment](#air-gapped-deployment)
  - [Staging Machine (Internet-Connected)](#staging-machine-internet-connected)
  - [Transfer to Air-Gapped Host](#transfer-to-air-gapped-host)
- [Layered Architecture Details](#layered-architecture-details)
  - [Layer 0 — Knowledge Graph](#layer-0--knowledge-graph)
  - [Layer 1 — Feature Engineering](#layer-1--feature-engineering)
  - [Layer 2 — ML Engine](#layer-2--ml-engine)
  - [Layer 3 — Markov Engine](#layer-3--markov-engine)
  - [Layer 4 — Risk Aggregation](#layer-4--risk-aggregation)
  - [Layer 5 — Presentation](#layer-5--presentation)
- [Feedback Loop](#feedback-loop)
- [Risk States](#risk-states)
- [Prioritization Algorithms](#prioritization-algorithms)
- [API Reference](#api-reference)
  - [REST API](#rest-api)
  - [GraphQL API](#graphql-api)
- [Testing](#testing)
- [License](#license)

---

## Overview

This system provides a mathematically principled, multi-stage pipeline for scoring and prioritizing vulnerability–asset pairs across an organization. Its core novelty is the **bidirectional coupling** between:

- **ML Models** — predict exploit likelihood, impact severity, and asset criticality from graph-derived features.
- **Absorbing Markov Chains** — model vulnerability lifecycle transitions (Unknown → Disclosed → ExploitAvailable → ActivelyExploited → Mitigated → Remediated) as a stochastic process with absorbing states.

Each pipeline cycle feeds ML predictions into the Markov transition probability matrix (TPM), then feeds the resulting Markov state distributions back into the feature space for the next ML inference pass, achieving iterative convergence.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Layer 5 — Presentation                       │
│          REST API · GraphQL · Web Dashboard · What-If Sim           │
├──────────────────────────────────────────────────────────────────────┤
│                     Layer 4 — Risk Aggregation                      │
│   Composite Risk · Attack-Path Propagation · BU Rollup · Forecast   │
│                   Multi-Algorithm Prioritization                    │
├────────────────────────────┬─────────────────────────────────────────┤
│   Layer 2 — ML Engine      │     Layer 3 — Markov Engine            │
│   ELP (Exploit Likelihood) │     Chapman-Kolmogorov Solver          │
│   ISA (Impact Severity)    │     Coupled Markov Networks            │
│   ACC (Asset Criticality)  │◄───►Absorption Time Analysis           │
│   Confidence Degradation   │     Risk Decay Calculator              │
│   Model Version Manager    │     Warm-Start Estimator               │
├────────────────────────────┴─────────────────────────────────────────┤
│                    Layer 1 — Feature Engineering                     │
│   Temporal · Threat Intel · Topological · Textual Embeddings        │
│   Historical Statistics · Markov Feedback Features · Feature Store  │
├──────────────────────────────────────────────────────────────────────┤
│                    Layer 0 — Knowledge Graph                        │
│   Graph Store (NetworkX/GraphML) · Entity Resolution · Ontologies   │
│   Data Freshness Monitor · Import Manager                           │
│   Connectors: NVD · CISA KEV · ExploitDB · OTX · CMDB · SIEM ·    │
│               Network Scans · Vendor Advisories                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
project/
├── run.py                          # CLI entry point
├── pyproject.toml                  # Build config & dependencies
├── pytest.ini                      # Test configuration
├── config/
│   ├── settings.yaml               # System-wide settings
│   ├── risk_states.yaml            # Markov risk state definitions
│   ├── cost_model.yaml             # Remediation cost parameters
│   ├── prioritization.yaml         # Prioritization algorithm configs
│   └── business_units.yaml         # Organizational hierarchy
├── data/
│   └── db/                         # DuckDB + graph storage (auto-created)
├── src/
│   ├── config.py                   # Pydantic configuration loader
│   ├── models.py                   # Domain models (Pydantic v2)
│   ├── layer0_knowledge_graph/     # Data ingestion & graph storage
│   │   ├── graph_store.py          # NetworkX-based graph with GraphML persistence
│   │   ├── entity_resolution.py    # Cross-source entity deduplication
│   │   ├── ontologies.py           # NodeType & RelationType enums
│   │   ├── import_manager.py       # Auto-routing file importer
│   │   ├── data_freshness_monitor.py
│   │   └── connectors/             # Per-source data loaders
│   │       ├── nvd_file_loader.py
│   │       ├── cisa_kev_loader.py
│   │       ├── exploitdb_file_loader.py
│   │       ├── otx_file_loader.py
│   │       ├── cmdb_file_loader.py
│   │       ├── network_scan_loader.py
│   │       ├── siem_alert_loader.py
│   │       └── vendor_advisory_loader.py
│   ├── layer1_feature_engineering/  # Feature computation
│   │   ├── feature_assembler.py    # Orchestrates all feature families
│   │   ├── feature_store.py        # DuckDB-backed feature persistence
│   │   ├── temporal_features.py    # Time-based features
│   │   ├── threat_intel_features.py # KEV, exploit, OTX indicators
│   │   ├── topological_features.py # Graph-derived structural features
│   │   ├── textual_embeddings.py   # TF-IDF/SVD or Transformer embeddings
│   │   ├── historical_statistics.py # CWE history, vendor patch rates
│   │   └── markov_feedback_features.py # Markov state distribution features
│   ├── layer2_ml_engine/           # ML model training & inference
│   │   ├── elp.py                  # Exploit Likelihood Predictor
│   │   ├── isa.py                  # Impact Severity Adjuster
│   │   ├── acc.py                  # Asset Criticality Classifier
│   │   ├── inference_engine.py     # Orchestrates all three models
│   │   ├── confidence_degradation.py # Time-based confidence decay
│   │   └── model_version_manager.py  # Model lifecycle tracking
│   ├── layer3_markov_engine/       # Absorbing Markov Model
│   │   ├── markov_engine.py        # Top-level Markov orchestrator
│   │   ├── tpm_computer.py         # Transition Probability Matrix computation
│   │   ├── chapman_kolmogorov.py   # State evolution solver
│   │   ├── absorption_time_analyzer.py # Expected time to remediation
│   │   ├── coupled_markov_networks.py  # Inter-asset coupling
│   │   ├── risk_decay_calculator.py    # Half-life based risk decay
│   │   ├── state_manager.py        # SQLite-backed state persistence
│   │   └── warm_start_estimator.py # Initial distribution bootstrap
│   ├── layer4_risk_aggregation/    # Final risk scoring & prioritization
│   │   ├── risk_aggregation_engine.py
│   │   ├── attack_path_propagation.py
│   │   ├── business_unit_rollup.py
│   │   ├── prioritization.py       # 5 prioritization algorithms + ensemble
│   │   └── temporal_risk_forecasting.py
│   ├── layer5_presentation/        # REST, GraphQL, Dashboard
│   │   ├── app.py                  # FastAPI application factory
│   │   ├── rest_api.py             # REST endpoints
│   │   ├── graphql_api.py          # Strawberry GraphQL schema
│   │   ├── service_layer.py        # API ↔ core bridge
│   │   └── static/index.html       # Web dashboard
│   ├── pipeline/                   # Pipeline orchestration
│   │   ├── daily_batch.py          # Full daily cycle
│   │   ├── on_demand.py            # Single-pair real-time assessment
│   │   └── feedback_loop.py        # ML ↔ Markov convergence loop
│   ├── staging/                    # Air-gapped deployment support
│   │   ├── download_feeds.py       # Download NVD/KEV feeds on staging machine
│   │   ├── train_models.py         # Train models on staging machine
│   │   └── package_for_transfer.py # Package data+models for USB transfer
│   └── backtesting/
│       └── backtesting_engine.py   # Validation against ground truth
└── tests/
    └── test_core.py                # Unit tests (24 test cases)
```

---

## Requirements

- **Python** ≥ 3.10
- **No internet required** on the production (air-gapped) host
- **No GPU required** — all ML models use sklearn/XGBoost (CPU-only)

Core dependencies (automatically installed):

- `numpy`, `scipy`, `scikit-learn`, `xgboost` — ML and numerical computing
- `networkx` — Knowledge graph backend
- `duckdb` — Feature store and analytics
- `pydantic` (v2) — Configuration and data models
- `fastapi`, `uvicorn` — Web server
- `strawberry-graphql` — GraphQL endpoint
- `joblib` — Model serialization
- `pyyaml` — Configuration files
- `pandas` — Data manipulation

Optional:

- `sentence-transformers`, `torch` — For transformer-based text embeddings (heavier, requires GPU for speed)

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd project

# Install in development mode
pip install -e ".[dev]"

# Or install with transformer embedding support
pip install -e ".[dev,embeddings]"
```

---

## Configuration

All configuration lives in the `config/` directory:

| File                  | Purpose                                                          |
| --------------------- | ---------------------------------------------------------------- |
| `settings.yaml`       | System-wide settings: paths, layer parameters, server config     |
| `risk_states.yaml`    | Markov risk state definitions (6 states)                         |
| `cost_model.yaml`     | Remediation costs, downtime costs, breach costs                  |
| `prioritization.yaml` | Algorithm weights and parameters for 5 prioritization strategies |
| `business_units.yaml` | Organizational hierarchy and criticality tiers                   |

Key settings in `settings.yaml`:

```yaml
layer0:
  graph_backend: "networkx" # or "neo4j"

layer1:
  embedding_backend: "tfidf" # or "transformer"
  tfidf_svd_components: 128
  topological_max_hops: 3

layer2:
  confidence_decay_halflife_days: 90 # model staleness decay

layer3:
  default_half_life_days: 30 # risk decay half-life
  coupling_max_hops: 2 # inter-asset Markov coupling depth

layer5:
  host: "127.0.0.1"
  port: 8080
```

---

## Usage

All commands are executed through the unified `run.py` CLI:

### Data Import

Import vulnerability data, CMDB exports, scan results, or threat intel feeds into the knowledge graph:

```bash
# Import a single file
python run.py import data/imports/nvd_cves_2024.json

# Import all files in a directory
python run.py import data/imports/
```

The `ImportManager` auto-detects the file format and routes it to the appropriate connector (NVD, CISA KEV, ExploitDB, OTX, CMDB, network scans, SIEM alerts, vendor advisories).

### Daily Batch Pipeline

Run the full multi-layer assessment cycle:

```bash
python run.py batch
python run.py batch --cycle-id 2025-03-20
```

The daily batch pipeline executes:

1. **Data freshness check** — validates all data sources are within staleness thresholds
2. **Pass 1 (Forward):** Feature assembly → ML inference → Markov evolution
3. **Pass 2 (Feedback):** Markov feedback features → Re-inference → Markov update
4. **Post-processing:** Risk aggregation, attack-path propagation, BU rollup, temporal forecasting, multi-algorithm prioritization

### On-Demand Assessment

Assess a single vulnerability–asset pair in real time:

```bash
python run.py assess CVE-2024-1234 asset-web-01
```

Returns exploit likelihood, impact severity, Markov state distribution, composite risk, and confidence score.

### Web Dashboard

Start the web server with REST API, GraphQL, and interactive dashboard:

```bash
python run.py serve
```

The dashboard is accessible at `http://127.0.0.1:8080/dashboard`. API documentation is available at `/docs` (Swagger) and `/redoc`.

### Backtesting

Validate predictions against historical ground truth:

```bash
python run.py backtest
python run.py backtest --period 2024-Q4
```

Evaluates: ROC-AUC for exploit prediction, MAE/RMSE for impact, Markov state accuracy, and Precision/Recall@10 for prioritization.

### Data Freshness

Check how up-to-date each data source is:

```bash
python run.py freshness
```

---

## Air-Gapped Deployment

The system is designed for split-environment deployment:

### Staging Machine (Internet-Connected)

1. **Download feeds** from public sources (NVD, CISA KEV):

```bash
python -m staging.download_feeds --output-dir ./transfer
```

2. **Train ML models** on downloaded data:

```bash
python -m staging.train_models --data-dir ./transfer/data --output-dir ./transfer/models
```

3. **Package for transfer** (generates checksums for integrity verification):

```bash
python -m staging.package_for_transfer --source-dir ./transfer --output ./transfer_bundle.tar.gz
```

### Transfer to Air-Gapped Host

1. Copy the transfer bundle via USB/approved media
2. Verify SHA-256 checksums
3. Place data files in `data/imports/`
4. Place trained models (`.joblib` files) in `models/elp/`, `models/isa/`, `models/acc/`
5. Run `python run.py import data/imports/` to ingest the data
6. Run `python run.py batch` to execute the full pipeline

---

## Layered Architecture Details

### Layer 0 — Knowledge Graph

A heterogeneous property graph storing all vulnerability, asset, threat intel, and relationship data.

**Node Types:**

- `VULNERABILITY` — CVE records with CVSS scores, descriptions, CWE associations
- `ASSET` — IT/OT assets with criticality, business unit, network zone
- `CWE` — Common Weakness Enumeration entries
- `CPE` — Common Platform Enumeration identifiers
- `INDICATOR` — Exploit code, threat intel indicators (OTX pulses)
- `SERVICE` — Network services discovered by scans
- `ALERT` — SIEM alert events

**Connectors:** NVD (CVE data), CISA KEV (known exploited vulns), ExploitDB (public exploits), OTX (threat indicators), CMDB (asset inventory), network scans (discovered services), SIEM alerts, vendor advisories.

**Entity Resolution:** Cross-source deduplication using CVE-ID matching, hostname/IP normalization, and fuzzy CPE matching.

**Storage:** NetworkX in-memory graph persisted as GraphML. Optional Neo4j backend.

### Layer 1 — Feature Engineering

Computes six feature families for each vulnerability–asset pair:

| Family                    | Features                                                                                 | Source                    |
| ------------------------- | ---------------------------------------------------------------------------------------- | ------------------------- |
| **Temporal**              | `days_since_disclosure`, `exploit_velocity`, `patch_lag`, `days_since_last_scan`         | Graph timestamps          |
| **Threat Intel**          | `is_in_kev`, `has_public_exploit`, `otx_pulse_count`, `threat_score`                     | KEV, ExploitDB, OTX       |
| **Topological**           | `asset_degree`, `vuln_degree`, `shared_cpe_count`, `network_exposure`                    | Graph structure           |
| **Textual Embeddings**    | `embed_0..embed_D` (D-dimensional)                                                       | TF-IDF/SVD or Transformer |
| **Historical Statistics** | `cvss_base_score`, `epss_score_estimate`, `historical_exploit_rate`, `vendor_patch_rate` | CWE history               |
| **Markov Feedback**       | `markov_state_*`, `markov_entropy`, `markov_absorption_time`, `markov_trend_*`           | Layer 3 states            |

Features are stored in a **DuckDB-backed Feature Store** with versioned cycles for reproducibility.

### Layer 2 — ML Engine

Three gradient-boosted/ensemble models:

| Model                                  | Task                  | Output                                  |
| -------------------------------------- | --------------------- | --------------------------------------- |
| **ELP** (Exploit Likelihood Predictor) | Binary classification | P(exploit) ∈ [0, 1]                     |
| **ISA** (Impact Severity Adjuster)     | Regression            | Adjusted impact score                   |
| **ACC** (Asset Criticality Classifier) | Multi-class (4 tiers) | P(critical), P(high), P(medium), P(low) |

**Confidence Degradation:** Model predictions are discounted based on model age using exponential half-life decay (configurable, default 90 days).

**Model Version Manager:** Tracks model versions, training dates, and supports A/B testing with configurable holdout fractions.

### Layer 3 — Markov Engine

Models vulnerability lifecycle as a **6-state absorbing Markov chain**:

```
Unknown(0) → Disclosed(1) → ExploitAvailable(2) → ActivelyExploited(3)
                    ↓                  ↓                      ↓
               Mitigated(4) ← ← ← ← ← ← ← ← ← ← ← ← ← ← ┘
                    ↓
              Remediated(5)  ← absorbing state
```

Key capabilities:

- **TPM Computation:** ML predictions modulate base transition probabilities
- **Chapman-Kolmogorov Solver:** Propagates distributions forward in time via matrix exponentiation
- **Absorption Time Analysis:** Expected time to reach the Remediated (absorbing) state
- **Coupled Markov Networks:** Inter-asset coupling where one asset's exploitation increases neighboring assets' transition rates
- **Warm-Start Estimator:** Bootstraps initial distributions for new vulnerabilities using k-nearest-neighbor interpolation of similar CVEs
- **Risk Decay:** Half-life-based decay of risk for mitigated/remediated states

### Layer 4 — Risk Aggregation

Combines all signals into actionable, prioritized outputs:

- **Composite Risk Score:** Weighted combination of ELP (30%), ISA (25%), Markov dominant-state risk (25%), and asset criticality (20%)
- **Attack-Path Propagation:** Propagates risk along network topology edges
- **Business Unit Rollup:** Recursively aggregates risk scores up the organizational hierarchy
- **Temporal Risk Forecasting:** Projects risk distributions 7, 30, 90 days into the future using the computed TPMs
- **Multi-Algorithm Prioritization:** Ensembles five strategies — Pareto frontier, cost-benefit analysis, time-sensitive urgency, TOPSIS multi-criteria, and rank averaging

### Layer 5 — Presentation

- **REST API** — CRUD endpoints for vulnerabilities, assets, risk pairs, BU rollup, forecasts, freshness reports, data import, and what-if simulation
- **GraphQL API** — Flexible querying via Strawberry-based schema
- **Web Dashboard** — Single-page HTML dashboard with real-time risk visualization
- **What-If Simulator** — Model the effect of patch/mitigate/isolate actions on risk distributions and compute projected risk reduction

---

## Feedback Loop

The core architectural innovation is the **bidirectional ML ↔ Markov feedback loop**:

```
┌─────────────┐         ┌─────────────┐
│  ML Engine  │────────►│Markov Engine │
│  (Layer 2)  │         │  (Layer 3)   │
│             │◄────────│              │
└─────────────┘         └─────────────┘
     ▲                        │
     │    ┌──────────────┐    │
     └────│Feature Store │◄───┘
          │  (Layer 1)   │
          └──────────────┘
```

1. ML predictions inform the Markov transition probability matrix
2. Markov state distributions become features for the next ML inference pass
3. The loop iterates until distributions converge (L1 distance < 0.01) or a maximum iteration count (default: 2)

This coupling ensures that exploitation dynamics, temporal trends, and uncertainty captured by the Markov model are continuously reflected in the ML predictions, and vice versa.

---

## Risk States

| ID  | State                 | Description                                         | Risk Weight |
| --- | --------------------- | --------------------------------------------------- | ----------- |
| 0   | **Unknown**           | Vulnerability exists but not yet publicly disclosed | 0.05        |
| 1   | **Disclosed**         | CVE assigned, publicly known                        | 0.20        |
| 2   | **ExploitAvailable**  | Public exploit or PoC exists                        | 0.60        |
| 3   | **ActivelyExploited** | Being exploited in the wild                         | 0.95        |
| 4   | **Mitigated**         | Workaround or compensating control in place         | 0.15        |
| 5   | **Remediated**        | Fully patched — absorbing state                     | 0.00        |

---

## Prioritization Algorithms

The system implements five complementary prioritization strategies, combined via ensemble:

| Algorithm          | Approach                                                 | Weight |
| ------------------ | -------------------------------------------------------- | ------ |
| **Pareto**         | Multi-objective Pareto frontier (risk vs. cost vs. time) | 25%    |
| **Cost-Benefit**   | Rank by benefit-to-cost ratio (min ratio: 1.5)           | 20%    |
| **Time-Sensitive** | Urgency weighting by exploitation state                  | 25%    |
| **TOPSIS**         | Multi-criteria decision analysis (5 criteria)            | 15%    |
| **Ensemble**       | Rank averaging across all strategies                     | 15%    |

---

## API Reference

### REST API

Base URL: `http://127.0.0.1:8080/api/v1`

| Method | Endpoint                    | Description                                 |
| ------ | --------------------------- | ------------------------------------------- |
| `GET`  | `/vulnerabilities`          | List vulnerabilities (paginated, sortable)  |
| `GET`  | `/vulnerabilities/{id}`     | Get vulnerability details + affected assets |
| `GET`  | `/assets`                   | List assets (paginated)                     |
| `GET`  | `/assets/{id}`              | Get asset details + vulnerabilities         |
| `GET`  | `/risk/pairs`               | List risk-scored vuln–asset pairs           |
| `GET`  | `/risk/summary`             | Aggregate risk statistics                   |
| `GET`  | `/risk/bu-rollup`           | Business unit risk rollup                   |
| `GET`  | `/risk/forecast/{pair_key}` | Temporal risk forecast for a pair           |
| `GET`  | `/prioritization`           | Prioritized remediation list                |
| `POST` | `/what-if`                  | Simulate remediation action effects         |
| `POST` | `/import`                   | Trigger data file import                    |
| `GET`  | `/freshness`                | Data source freshness report                |
| `GET`  | `/health`                   | System health check                         |

### GraphQL API

Endpoint: `http://127.0.0.1:8080/graphql`

```graphql
query {
  vulnerabilities(limit: 10) {
    id
    cvssBaseScore
    hasPublicExploit
    compositeRisk
  }
  riskPairs(limit: 5, minRisk: 0.7) {
    vulnId
    assetId
    compositeRisk
    finalRank
  }
  freshness {
    source
    ageDays
    isStale
  }
}
```

---

## Testing

Run the test suite (24 test cases):

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Tests cover:

- Domain models and enum validation
- Ontology types
- Data freshness monitoring
- Temporal feature computation
- Feature store persistence
- Confidence degradation
- Chapman-Kolmogorov state evolution and absorbing state convergence
- Absorption time analysis
- Markov state management
- Multi-algorithm prioritization (Pareto, cost-benefit, time-sensitive, TOPSIS, ensemble)
- Backtesting engine (ROC-AUC computation)
- Staging package checksum generation

---

## License

See [LICENSE](LICENSE) for details.
