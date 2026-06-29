# 🛡️ FinCrime Platform: AI-Augmented Investigator Workstation

![FinCrime Dashboard](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B) ![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)

An end-to-end, production-style Financial Crime and Anti-Money Laundering (AML) detection platform. This project demonstrates the intersection of traditional regulatory compliance, deterministic rules (JMLSG/NCA mapped), and modern Machine Learning anomaly detection.

## 📌 Business Objective

Traditional Transaction Monitoring (TM) systems suffer from overwhelming false-positive rates (often >90%), leading to massive operational overhead for Level 1 triage analysts. 

This platform acts as an **AI-Augmented Challenger System**. It combines a deterministic rule engine with an XGBoost ML classifier to score transactions, dynamically visualizes 1-degree counterparty networks, and generates structured Suspicious Activity Reports (SARs) to accelerate the investigative workflow.

---

## 🚀 Core Capabilities

1. **JMLSG-Aligned Deterministic Rule Engine:**  
   Detects known typologies like Cash Structuring (Smurfing), Rapid Movement of Funds (Layering), and High-Risk Jurisdiction transfers using hard thresholds mapped to MLR 2017 and NCA guidance.

2. **XGBoost ML Challenger Model with SHAP:**  
   A tree-based gradient boosting model trained on engineered behavioral features (1D/7D/30D volume deviations, velocity ratios). Integrates **SHAP** (SHapley Additive exPlanations) to explicitly tell the investigator *why* a transaction is anomalous, avoiding the "black box" trap.

3. **High-Performance Data Engineering Pipeline:**  
   Utilizes `Polars` to aggregate millions of synthetic transactions instantly, creating temporal baselines and feeding the ML models in a highly scalable `.parquet` format.

4. **Investigator Workstation UX (Streamlit):**  
   A fully interactive dashboard for L1/L2 analysts. Features include triage queues, case management persistence, KPI summaries, and interactive Network Graphing via `NetworkX` & `Plotly`.

5. **Automated SAR Generation:**  
   A one-click tool that synthesizes entity profiles, rule triggers, ML drivers, and investigator notes into a structured draft ready for National Crime Agency (NCA) submission.

---

## 🛠️ Technology Stack

- **Data Engineering:** `Polars`, `Pandas`, `NumPy`, `Parquet`
- **Machine Learning:** `XGBoost`, `Scikit-Learn`, `SHAP`
- **Frontend / UI:** `Streamlit`, `Plotly`, custom CSS design system
- **Network Analysis:** `NetworkX`
- **Configuration:** `YAML`, `JSON`

---

## 💻 Quickstart (Running Locally)

This repository includes a highly realistic synthetic data generator, ensuring you can run the entire end-to-end pipeline locally without needing sensitive PII data.

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/your-username/FinCrime-Platform.git
cd FinCrime-Platform
pip install -r requirements.txt
```

**2. Generate the Data & Train the ML Model:**
```bash
# Generates 1M+ synthetic transactions, applies features, and trains the XGBoost model
python scripts/train_model.py
```

**3. Launch the Investigator Dashboard:**
```bash
# Starts the Streamlit application on localhost:8501
python scripts/run_ui.sh
```

---

## 🧠 Architecture & Project Structure

```text
FinCrime-Platform/
│
├── configs/
│   └── rules.yaml                 # Deterministic typology definitions (JMLSG mapped)
├── data/
│   └── synthetic/                 # Ignored by git; holds local .parquet files and state
├── notebooks/                     # Exploratory Data Analysis & Prototyping
├── scripts/
│   ├── run_ui.sh                  # Execution script for the dashboard
│   └── train_model.py             # End-to-end pipeline orchestrator
├── src/
│   ├── data/
│   │   └── data_generator.py      # Generates synthetic GBP transactions & KYC profiles
│   ├── features/
│   │   └── pipeline.py            # Polars-based temporal feature engineering
│   ├── models/
│   │   └── ml_detector.py         # XGBoost training & SHAP explainer implementation
│   └── ui/
│       ├── app.py                 # Main Streamlit application
│       ├── styles.css             # Custom design system UI styling
│       └── ui_components.py       # Reusable Streamlit components (metric cards, badges)
└── README.md
```

---

## 👤 Author

**Shashank Gogula**  
*MSc Business Analytics | ICA AML Level 2*  
Targeting UK roles in AML, Financial Crime, Fraud Analytics, and Risk Operations.  
[LinkedIn Profile] | [GitHub Profile]
