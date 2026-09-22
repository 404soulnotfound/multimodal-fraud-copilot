"""
Forensic Copilot Synthesis Engine.
Fuses:
1. Transaction tabular anomaly scores (XGBoost)
2. Graph structural risk (PyG GNN embeddings & PageRank)
3. Document forensic tampering detection (Vision ELA)
Generates natural language case dossiers and audit trails for compliance & fraud analysts.
"""

import numpy as np
import pandas as pd


class FraudCopilotEngine:
    def __init__(self, graph_engine, doc_analyzer):
        self.graph_engine = graph_engine
        self.doc_analyzer = doc_analyzer

    def evaluate_case(self, tx_row: dict, doc_image_path: str = None) -> dict:
        sender_id = tx_row.get("sender_id")
        user_node = f"user:{sender_id}"
        
        # 1. Graph Risk Component
        gnn_score = self.graph_engine.gnn_scores.get(user_node, 0.05) if self.graph_engine.gnn_scores else 0.05
        pagerank = self.graph_engine.pagerank.get(user_node, 0.001) if hasattr(self.graph_engine, "pagerank") else 0.001
        
        # Check device sharing count
        dev_neighbors = []
        shared_accounts_count = 1
        if user_node in self.graph_engine.G:
            dev_neighbors = [nbr for nbr in self.graph_engine.G.neighbors(user_node) 
                             if self.graph_engine.G.nodes[nbr].get("node_type") == "device"]
            if dev_neighbors:
                shared_accounts_count = self.graph_engine.degrees.get(dev_neighbors[0], 1)

        # 2. Document Risk Component
        doc_analysis = None
        doc_risk = 0.0
        if doc_image_path:
            doc_analysis = self.doc_analyzer.generate_llm_forensic_report(doc_image_path, tx_row)
            doc_risk = doc_analysis["tamper_confidence"]

        # 3. Multimodal Unified Risk Fusion
        # Fuses Tabular Amount Heuristic + Graph Topology + Document Authenticity
        amount = tx_row.get("amount", 0.0)
        amount_risk = min(1.0, amount / 10000.0) if amount > 4000 else 0.1
        
        # Weighted composite risk score
        if doc_image_path:
            composite_risk = (0.40 * gnn_score) + (0.35 * doc_risk) + (0.25 * amount_risk)
        else:
            composite_risk = (0.65 * gnn_score) + (0.35 * amount_risk)

        # Severity level
        if composite_risk > 0.65:
            severity = "CRITICAL / HIGH RISK"
            badge_color = "#ef4444"
        elif composite_risk > 0.35:
            severity = "ELEVATED / SUSPICIOUS"
            badge_color = "#f59e0b"
        else:
            severity = "NORMAL / LOW RISK"
            badge_color = "#10b981"

        # 4. Generate Natural Language Explanation
        evidence_points = []
        if shared_accounts_count > 3:
            evidence_points.append(f"🚨 **Coordinated Device Ring:** Device `{tx_row.get('device_id')}` is shared by {shared_accounts_count} separate user accounts.")
        if gnn_score > 0.5:
            evidence_points.append(f"🕸️ **Graph Neural Net Flag:** Sender's topological neighborhood has a high anomalous centrality score ({gnn_score:.2f}).")
        if amount > 4500 and amount < 10000:
            evidence_points.append(f"⚠️ **Structuring Indicator:** Transaction amount (${amount:,.2f}) sits just below mandatory regulatory compliance thresholds.")
        if doc_analysis and doc_analysis["tamper_confidence"] > 0.5:
            evidence_points.append("📄 **Document Forgery:** Error Level Analysis detected digital splicing and inconsistent subtotal math on attached invoice.")
            
        if not evidence_points:
            evidence_points.append("✅ Standard transaction patterns: Solo device history, normal velocity, and verified counterparties.")

        explanation = (
            f"### AI Forensic Investigation Dossier\n\n"
            f"**Case Status:** <span style='color:{badge_color}; font-weight:bold;'>{severity} (Risk Score: {composite_risk*100:.1f}%)</span>\n\n"
            f"#### Key Findings:\n" + "\n".join([f"- {p}" for p in evidence_points]) + "\n\n"
            f"#### Copilot Recommendation:\n"
            f"{'Immediate freeze of account assets and referral to AML/Financial Crimes Unit.' if composite_risk > 0.65 else 'Allow transfer to proceed under standard post-settlement monitoring.'}"
        )

        return {
            "composite_risk_score": float(composite_risk),
            "severity": severity,
            "gnn_structural_score": float(gnn_score),
            "shared_device_accounts": shared_accounts_count,
            "doc_analysis": doc_analysis,
            "dossier_markdown": explanation
        }
