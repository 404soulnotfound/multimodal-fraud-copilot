# Multimodal Financial Fraud & Identity Spoofing Copilot

An enterprise-grade, end-to-end multimodal fraud investigation copilot and forensic decision-support platform. 

This project tackles coordinated financial crime and identity spoofing by unifying:
1. **Graph Neural Networks (PyG GNN):** Structural ring and mule-network detection across users, devices, and IP subnets.
2. **Computer Vision Document Forensics:** Error Level Analysis (ELA) and layout consistency checking on invoices, pay slips, and identification documents.
3. **Tabular Anomaly Signals:** Structuring heuristics and velocity anomaly detection.
4. **Interactive Forensic Copilot Dashboard (Streamlit + Plotly):** Interactive 2-hop ego-graph visualization, case dossier generator, and enforcement actions.

---

## 🌟 Key Architecture Highlights

* **Heterogeneous Graph Engine:** Formulates transactions as a multi-entity network (`User -> Transacted -> User`, `User -> Shared -> Device`, `User -> Connected -> IP`).
* **PyTorch Graph Convolutional / SAGE Model:** Embeds high-order topological neighborhood properties to flag coordinated syndicates that bypass traditional tabular filters.
* **Digital Splicing & Forgery Detection:** Implements Error Level Analysis (ELA) using PIL and OpenCV to highlight compression discontinuities and digital tampering on supporting PDF/PNG invoices.
* **Unified Risk Fusion:** Combines graph topological risk, document authenticity confidence, and transaction structuring signals into an interpretable forensic audit trail.

---

## 🏗️ Project Structure

```
multimodal_fraud_copilot/
├── data/
│   ├── generate_fraud_data.py          # Generates transaction graph & synthetic forged documents
│   ├── transactions.csv                # Generated transactions dataset
│   └── sample_documents/               # Synthetic clean & tampered invoices
├── src/
│   ├── __init__.py
│   ├── graph_engine.py                 # NetworkX & PyG Graph Neural Network implementation
│   ├── doc_analyzer.py                 # Computer Vision Error Level Analysis & heuristic checks
│   └── copilot_engine.py               # Multimodal risk fusion & case dossier generator
├── app/
│   └── streamlit_app.py                # 4-tab interactive forensic investigation dashboard
├── requirements.txt                    # Dependencies (PyTorch, NetworkX, OpenCV, Streamlit, etc.)
└── README.md                           # Documentation
```

---

## 🚀 Quickstart Guide (Local Setup & Run)

### 1. Navigate to Project & Create Environment
Open PowerShell or your terminal:

```powershell
cd C:\Users\soumili\Desktop\projects\multimodal_fraud_copilot
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Generate Data & Synthetic Documents (Auto-generates if missing)
```powershell
python data/generate_fraud_data.py
```

### 3. Launch the Interactive Dashboard
```powershell
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ What You Can Explore in the Web App

1. **Forensic Case Dossier Tab:**
   * Select flagged transactions from coordinated fraud rings.
   * Review the AI-generated forensic dossier with highlighted evidence (device sharing, GNN risk score, structuring indicators).
   * Interactive enforcement buttons (Freeze Account, Request KYC, Clear False Positive).
2. **Transaction Graph & GNN Tab:**
   * Interactive 2-hop ego-network rendered in Plotly.
   * Visually inspect how multiple accounts cluster around a single shared hardware device ID or IP.
3. **Document Forensic Vision Tab:**
   * Side-by-side comparison of original supporting invoices against the **Error Level Analysis (ELA) heatmap**.
   * Identifies spliced modified totals and font inconsistencies.
4. **Network Macro Analytics Tab:**
   * Breakdown of fraud rates by merchant category (crypto exchanges, luxury goods, P2P transfers).

---

## 🌐 Deployment (e.g. Streamlit Community Cloud / Render)
1. Push this folder to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Multimodal Fraud Copilot"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
2. Link your repo on [share.streamlit.io](https://share.streamlit.io) with entry point `app/streamlit_app.py`.
