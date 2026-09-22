"""
Multimodal Financial Fraud & Identity Spoofing Investigation Copilot Dashboard.
Built with Streamlit, Plotly, PyTorch GNN, and Computer Vision.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from data.generate_fraud_data import generate_fraud_dataset, create_sample_forensic_documents
from src.graph_engine import TransactionGraphEngine
from src.doc_analyzer import DocumentForensicAnalyzer
from src.copilot_engine import FraudCopilotEngine


# Page configuration
st.set_page_config(
    page_title="Multimodal Fraud & Forensic Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-container {
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        background-color: #0f172a;
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-safe {
        background-color: #10b981;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_or_create_dataset():
    data_dir = PROJECT_ROOT / "data"
    tx_file = data_dir / "transactions.csv"
    if not tx_file.exists():
        df = generate_fraud_dataset()
        df.to_csv(tx_file, index=False)
        create_sample_forensic_documents(data_dir)
    else:
        df = pd.read_csv(tx_file)
    return df


@st.cache_resource
def initialize_engines():
    df = load_or_create_dataset()
    graph_engine = TransactionGraphEngine()
    graph_engine.build_graph(df)
    graph_engine.train_gnn(df, epochs=50)
    
    doc_analyzer = DocumentForensicAnalyzer()
    copilot = FraudCopilotEngine(graph_engine, doc_analyzer)
    return graph_engine, doc_analyzer, copilot


def render_network_graph(subgraph_data: dict, target_user: str):
    """Renders an interactive 2D network subgraph with Plotly."""
    nodes = subgraph_data["nodes"]
    edges = subgraph_data["edges"]
    
    # 1. Edge traces
    edge_x = []
    edge_y = []
    for e in edges:
        edge_x.extend([e["x0"], e["x1"], None])
        edge_y.extend([e["y0"], e["y1"], None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="#64748b"),
        hoverinfo="none",
        mode="lines"
    )
    
    # 2. Node traces by type
    node_x = [n["x"] for n in nodes]
    node_y = [n["y"] for n in nodes]
    node_colors = []
    node_sizes = []
    node_texts = []
    
    for n in nodes:
        ntype = n["type"]
        if n["is_target"]:
            node_colors.append("#ef4444")  # Red for target
            node_sizes.append(24)
        elif ntype == "device":
            node_colors.append("#3b82f6")  # Blue for device
            node_sizes.append(18)
        elif ntype == "ip":
            node_colors.append("#a855f7")  # Purple for IP
            node_sizes.append(14)
        else:
            # User node colored by fraud risk
            risk = n.get("fraud_risk", 0.0)
            node_colors.append(f"rgba({int(risk*255)}, {int((1-risk)*200)}, 50, 0.9)")
            node_sizes.append(16)
            
        node_texts.append(f"<b>{n['id']}</b><br>Type: {ntype}<br>GNN Risk: {n.get('fraud_risk', 0.0)*100:.1f}%")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[n["label"] if n["is_target"] or n["type"] == "device" else "" for n in nodes],
        textposition="top center",
        hovertext=node_texts,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1.5, color="#ffffff")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=420,
                        plot_bgcolor="#0b0f19",
                        paper_bgcolor="#0b0f19"
                    ))
    return fig


def main():
    st.sidebar.title("🛡️ Multimodal Fraud Copilot")
    st.sidebar.markdown("**Graph Neural Network & Forensic Vision Engine**")
    st.sidebar.markdown("---")
    
    df = load_or_create_dataset()
    graph_engine, doc_analyzer, copilot = initialize_engines()
    
    # Sidebar Filters
    fraud_only = st.sidebar.checkbox("Filter to High-Risk Flagged Transactions", value=True)
    if fraud_only:
        filtered_df = df[df["is_fraud"] == 1]
    else:
        filtered_df = df
        
    selected_tx_id = st.sidebar.selectbox(
        "Select Transaction for Forensic Case Audit:",
        options=filtered_df["transaction_id"].tolist()
    )
    
    tx_row = df[df["transaction_id"] == selected_tx_id].iloc[0].to_dict()
    
    # App Tabs
    tab_case, tab_graph, tab_docs, tab_analytics = st.tabs([
        "🔍 Forensic Case Dossier",
        "🕸️ Transaction Graph & GNN",
        "📄 Document Forensic Vision",
        "📊 Network Macro Analytics"
    ])

    # ==========================================
    # TAB 1: FORENSIC CASE DOSSIER
    # ==========================================
    with tab_case:
        st.subheader(f"Case Audit: Transaction `{tx_row['transaction_id']}`")
        
        # Determine attached document
        doc_dir = PROJECT_ROOT / "data" / "sample_documents"
        if tx_row["is_fraud"] == 1:
            doc_file = doc_dir / "invoice_tampered_forgery.png"
        else:
            doc_file = doc_dir / "invoice_legitimate_clean.png"
            
        case_result = copilot.evaluate_case(tx_row, str(doc_file))
        
        # Top KPI Banner
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transaction Amount", f"${tx_row['amount']:,.2f}")
        c2.metric("Sender Account", tx_row["sender_id"])
        c3.metric("Counterparty", tx_row["receiver_id"])
        c4.metric("Shared Device Accounts", f"{case_result['shared_device_accounts']} linked")
        
        st.markdown("---")
        
        col_dos_l, col_dos_r = st.columns([3, 2])
        
        with col_dos_l:
            st.markdown(case_result["dossier_markdown"], unsafe_allow_html=True)
            
            st.markdown("#### Quick Enforcement Action")
            act_col1, act_col2, act_col3 = st.columns(3)
            with act_col1:
                if st.button("🚨 Freeze Sender Account", type="primary", use_container_width=True):
                    st.error(f"Account {tx_row['sender_id']} has been locked. Audit ticket submitted.")
            with act_col2:
                if st.button("⚠️ Request Identity Verification", use_container_width=True):
                    st.warning(f"KYC Challenge dispatched to {tx_row['sender_id']}.")
            with act_col3:
                if st.button("✅ Clear False Positive", use_container_width=True):
                    st.success(f"Transaction {tx_row['transaction_id']} cleared.")

        with col_dos_r:
            st.markdown("#### Risk Signal Breakdown")
            risk_breakdown = pd.DataFrame([
                {"Signal": "Graph Topology (GNN)", "Risk Weight": case_result["gnn_structural_score"] * 100},
                {"Signal": "Document Tampering (Vision)", "Risk Weight": (case_result["doc_analysis"]["tamper_confidence"] if case_result["doc_analysis"] else 0.1) * 100},
                {"Signal": "Structuring Velocity", "Risk Weight": min(100.0, (tx_row["amount"] / 10000.0) * 100)}
            ])
            fig_bar = go.Figure(go.Bar(
                x=risk_breakdown["Risk Weight"],
                y=risk_breakdown["Signal"],
                orientation="h",
                marker_color=["#3b82f6", "#ef4444", "#f59e0b"]
            ))
            fig_bar.update_layout(xaxis_title="Risk Contribution (%)", height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

    # ==========================================
    # TAB 2: TRANSACTION GRAPH & GNN
    # ==========================================
    with tab_graph:
        st.subheader("Interactive 2-Hop Ego Graph & Collusion Rings")
        st.markdown(
            "Visualizing the topological neighborhood of the target account: "
            "**Red** = Target Account | **Blue** = Device ID | **Purple** = IP Subnet"
        )
        
        target_node = f"user:{tx_row['sender_id']}"
        subgraph = graph_engine.get_ego_subgraph(target_node, radius=2)
        
        if subgraph["num_nodes"] > 0:
            fig_graph = render_network_graph(subgraph, target_user=target_node)
            st.plotly_chart(fig_graph, use_container_width=True)
            
            st.info(
                f"**Topological Insights:** Node `{target_node}` connects with {subgraph['num_nodes']} neighboring entities. "
                f"GNN structural anomaly score: `{graph_engine.gnn_scores.get(target_node, 0.0):.3f}`"
            )
        else:
            st.warning("No localized subgraph available for this entity.")

    # ==========================================
    # TAB 3: DOCUMENT FORENSIC VISION
    # ==========================================
    with tab_docs:
        st.subheader("Multimodal Invoice Inspection & Error Level Analysis (ELA)")
        st.markdown(
            "Combines **Computer Vision compression analysis (ELA)** with **OCR/Layout parsing** to detect digitally spliced numbers and forged totals."
        )
        
        doc_c1, doc_c2 = st.columns(2)
        
        with doc_c1:
            st.markdown("#### Original Supporting Document")
            if doc_file.exists():
                orig_img = Image.open(doc_file)
                st.image(orig_img, caption=f"Document File: {doc_file.name}", use_container_width=True)
                
        with doc_c2:
            st.markdown("#### Computer Vision Error Level Analysis (ELA)")
            if doc_file.exists():
                ela_img = doc_analyzer.compute_error_level_analysis(doc_file)
                st.image(ela_img, caption="ELA Heatmap: Bright areas show compression discrepancies and digital tampering boundaries.", use_container_width=True)

        if doc_file.exists():
            heuristics = doc_analyzer.analyze_document_heuristics(doc_file)
            st.markdown("#### Forensic Inspection Results")
            for finding in heuristics["forensic_findings"]:
                if heuristics["is_suspicious"]:
                    st.error(f"⚠️ {finding}")
                else:
                    st.success(f"✅ {finding}")

    # ==========================================
    # TAB 4: NETWORK MACRO ANALYTICS
    # ==========================================
    with tab_analytics:
        st.subheader("Global Transaction Risk & Ring Detection Summary")
        
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Total Transactions Monitored", f"{len(df):,}")
        c_m2.metric("Flagged Fraud Ring Transactions", f"{df['is_fraud'].sum():,}")
        c_m3.metric("Syndicate Detection Rate", f"{df['is_fraud'].mean()*100:.1f}%")
        
        st.markdown("#### Transaction Distribution by Merchant Category")
        cat_counts = df.groupby(["merchant_category", "is_fraud"]).size().unstack(fill_value=0).reset_index()
        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(x=cat_counts["merchant_category"], y=cat_counts[0], name="Legitimate", marker_color="#10b981"))
        fig_cat.add_trace(go.Bar(x=cat_counts["merchant_category"], y=cat_counts[1], name="Flagged Fraud", marker_color="#ef4444"))
        fig_cat.update_layout(barmode="stack", height=380, xaxis_title="Merchant Category", yaxis_title="Number of Transactions")
        st.plotly_chart(fig_cat, use_container_width=True)


if __name__ == "__main__":
    main()
