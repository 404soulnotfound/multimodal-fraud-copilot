"""
Multimodal Financial Fraud & Identity Spoofing Investigation Copilot Dashboard.
Redesigned with Modern FinTech Risk & Trust Console UI/UX (Option 1: Stripe Radar / Unit21 Clean Light).
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
    page_title="ApexRisk // Fraud & Forensic Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Option 1: Modern FinTech Risk & Trust Console (Clean Light Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main App Light Background */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    
    /* Global Card Container */
    .fintech-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
    }
    
    /* Header Top Bar */
    .top-banner {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    /* Badges */
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .badge-elevated {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    .meta-mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #475569;
    }
    
    /* Clean Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 4px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 600;
        color: #64748B;
        border-radius: 8px;
        padding: 8px 16px;
        background: transparent;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    
    /* Clean Metric Box */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 800 !important;
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


def render_light_network_graph(subgraph_data: dict, target_user: str):
    """Renders a clean, light-mode interactive network graph for FinTech consoles."""
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
        line=dict(width=1.2, color="#CBD5E1"),
        hoverinfo="none",
        mode="lines"
    )
    
    # 2. Node traces
    node_x = [n["x"] for n in nodes]
    node_y = [n["y"] for n in nodes]
    node_colors = []
    node_sizes = []
    node_borders = []
    node_texts = []
    
    for n in nodes:
        ntype = n["type"]
        if n["is_target"]:
            node_colors.append("#EF4444")  # Vivid red target
            node_sizes.append(26)
            node_borders.append("#991B1B")
        elif ntype == "device":
            node_colors.append("#3B82F6")  # Crisp Blue device
            node_sizes.append(20)
            node_borders.append("#1D4ED8")
        elif ntype == "ip":
            node_colors.append("#8B5CF6")  # Purple IP
            node_sizes.append(15)
            node_borders.append("#6D28D9")
        else:
            risk = n.get("fraud_risk", 0.0)
            if risk > 0.6:
                node_colors.append("#F87171")
                node_borders.append("#DC2626")
            else:
                node_colors.append("#10B981")
                node_borders.append("#059669")
            node_sizes.append(18)
            
        node_texts.append(
            f"<b>{n['id']}</b><br>"
            f"Entity Type: <b>{ntype.upper()}</b><br>"
            f"GNN Anomaly Risk: <b>{n.get('fraud_risk', 0.0)*100:.1f}%</b>"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[n["label"] if n["is_target"] or n["type"] == "device" else "" for n in nodes],
        textposition="top center",
        hovertext=node_texts,
        textfont=dict(family="Plus Jakarta Sans", size=11, color="#334155"),
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color=node_borders)
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=10, l=10, r=10, t=10),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=420,
                        plot_bgcolor="#FFFFFF",
                        paper_bgcolor="#FFFFFF"
                    ))
    return fig


def main():
    # Sidebar
    st.sidebar.markdown("### 🛡️ **ApexRisk Console**")
    st.sidebar.caption("Enterprise Fraud & Forensics Engine")
    st.sidebar.markdown("---")
    
    df = load_or_create_dataset()
    graph_engine, doc_analyzer, copilot = initialize_engines()
    
    # Filter selection
    st.sidebar.markdown("**Case Queue Filter**")
    fraud_only = st.sidebar.toggle("Show High-Risk Flagged Transactions Only", value=True)
    if fraud_only:
        filtered_df = df[df["is_fraud"] == 1]
    else:
        filtered_df = df
        
    selected_tx_id = st.sidebar.selectbox(
        "Select Transaction Case ID:",
        options=filtered_df["transaction_id"].tolist()
    )
    
    tx_row = df[df["transaction_id"] == selected_tx_id].iloc[0].to_dict()
    
    # Attached document resolution
    doc_dir = PROJECT_ROOT / "data" / "sample_documents"
    if tx_row["is_fraud"] == 1:
        doc_file = doc_dir / "invoice_tampered_forgery.png"
    else:
        doc_file = doc_dir / "invoice_legitimate_clean.png"
        
    case_result = copilot.evaluate_case(tx_row, str(doc_file))
    risk_pct = case_result["composite_risk_score"] * 100
    
    if risk_pct > 65:
        badge_html = f'<span class="badge-critical">● CRITICAL RISK ({risk_pct:.1f}%)</span>'
    elif risk_pct > 35:
        badge_html = f'<span class="badge-elevated">▲ ELEVATED RISK ({risk_pct:.1f}%)</span>'
    else:
        badge_html = f'<span class="badge-normal">✓ LOW RISK ({risk_pct:.1f}%)</span>'

    # Top Brand Ribbon
    st.markdown(f"""
    <div class="top-banner">
        <div>
            <h2 style="margin:0; font-size: 22px; font-weight: 800; color: #0F172A;">
                Case Audit: <span style="font-family:'JetBrains Mono'; color:#2563EB;">{tx_row['transaction_id']}</span>
            </h2>
            <div style="margin-top: 4px; font-size: 13px; color: #64748B;">
                Timestamp: <b>{tx_row['timestamp']}</b> &nbsp;|&nbsp; Category: <b>{tx_row['merchant_category'].replace('_', ' ').title()}</b>
            </div>
        </div>
        <div>
            {badge_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Top KPI Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Transaction Volume", f"${tx_row['amount']:,.2f}")
    col_kpi2.metric("Sender Account", tx_row["sender_id"])
    col_kpi3.metric("Counterparty", tx_row["receiver_id"])
    col_kpi4.metric("Shared Device Accounts", f"{case_result['shared_device_accounts']} linked")
    
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Main Tabs
    tab_dossier, tab_graph, tab_docs, tab_analytics = st.tabs([
        "📋 Forensic Case Dossier",
        "🕸️ 2-Hop Network Graph & GNN",
        "🔍 Document Forensic & ELA Viewer",
        "📊 Portfolio Risk Analytics"
    ])

    # ==========================================
    # TAB 1: FORENSIC CASE DOSSIER
    # ==========================================
    with tab_dossier:
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("### AI Copilot Risk Dossier")
            st.markdown(case_result["dossier_markdown"], unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### Quick Enforcement Actions")
            act1, act2, act3 = st.columns(3)
            with act1:
                if st.button("🚨 Freeze Sender Account", type="primary", use_container_width=True):
                    st.error(f"Account {tx_row['sender_id']} locked. Automated SAR filed with FinCEN.")
            with act2:
                if st.button("⚠️ Request Enhanced KYC", use_container_width=True):
                    st.warning(f"KYC Challenge triggered for {tx_row['sender_id']}.")
            with act3:
                if st.button("✅ Clear False Positive", use_container_width=True):
                    st.success(f"Case {tx_row['transaction_id']} marked as verified legitimate.")

        with col_right:
            st.markdown("### Multimodal Risk Decomposition")
            risk_breakdown = pd.DataFrame([
                {"Signal": "Graph Topology (GNN)", "Risk Weight": case_result["gnn_structural_score"] * 100},
                {"Signal": "Document Splicing (Vision ELA)", "Risk Weight": (case_result["doc_analysis"]["tamper_confidence"] if case_result["doc_analysis"] else 0.1) * 100},
                {"Signal": "Structuring Velocity", "Risk Weight": min(100.0, (tx_row["amount"] / 10000.0) * 100)}
            ])
            fig_bar = go.Figure(go.Bar(
                x=risk_breakdown["Risk Weight"],
                y=risk_breakdown["Signal"],
                orientation="h",
                marker=dict(
                    color=["#2563EB", "#EF4444", "#F59E0B"],
                    line=dict(width=0)
                )
            ))
            fig_bar.update_layout(
                xaxis=dict(title="Risk Contribution Score (%)", range=[0, 100], gridcolor="#F1F5F9"),
                yaxis=dict(autorange="reversed"),
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("### Device & IP Footprint")
            st.markdown(f"""
            - **Device ID:** `{tx_row['device_id']}`
            - **Origin IP:** `{tx_row['ip_address']}`
            - **Cluster Association:** {'🚨 Coordinated Syndicate Ring A' if 'FRAUD' in tx_row['device_id'] else 'Standard Solo Hardware'}
            """)

    # ==========================================
    # TAB 2: NETWORK GRAPH & GNN
    # ==========================================
    with tab_graph:
        st.subheader("Heterogeneous 2-Hop Ego Graph")
        st.markdown(
            "Graph topology around the target account: "
            "<span style='color:#EF4444; font-weight:700;'>● Target</span> | "
            "<span style='color:#3B82F6; font-weight:700;'>● Device ID</span> | "
            "<span style='color:#8B5CF6; font-weight:700;'>● IP Subnet</span> | "
            "<span style='color:#10B981; font-weight:700;'>● Legitimate Accounts</span>",
            unsafe_allow_html=True
        )
        
        target_node = f"user:{tx_row['sender_id']}"
        subgraph = graph_engine.get_ego_subgraph(target_node, radius=2)
        
        if subgraph["num_nodes"] > 0:
            fig_graph = render_light_network_graph(subgraph, target_user=target_node)
            st.plotly_chart(fig_graph, use_container_width=True)
            
            st.info(
                f"**Graph Analytics:** Node `{target_node}` links to **{subgraph['num_nodes']}** neighboring entities. "
                f"Graph Neural Network anomaly score: **`{graph_engine.gnn_scores.get(target_node, 0.0):.3f}`**"
            )
        else:
            st.warning("No localized subgraph available for this entity.")

    # ==========================================
    # TAB 3: DOCUMENT FORENSIC & ELA VIEWER
    # ==========================================
    with tab_docs:
        st.subheader("Supporting Invoice Inspection: Original vs. Error Level Analysis (ELA)")
        st.markdown(
            "Combines **Computer Vision compression analysis (ELA)** with **OCR mathematical reconciliation** to detect digitally spliced totals."
        )
        
        doc_c1, doc_c2 = st.columns(2)
        with doc_c1:
            st.markdown("#### 1. Uploaded Invoice Document")
            if doc_file.exists():
                orig_img = Image.open(doc_file)
                st.image(orig_img, caption=f"File: {doc_file.name}", use_container_width=True)
                
        with doc_c2:
            st.markdown("#### 2. Computer Vision ELA Compression Residual")
            if doc_file.exists():
                ela_img = doc_analyzer.compute_error_level_analysis(doc_file)
                st.image(ela_img, caption="ELA Heatmap: Bright regions indicate compression discontinuities and digital tampering boundaries.", use_container_width=True)

        if doc_file.exists():
            heuristics = doc_analyzer.analyze_document_heuristics(doc_file)
            st.markdown("#### Forensic Inspection Checkpoints")
            for finding in heuristics["forensic_findings"]:
                if heuristics["is_suspicious"]:
                    st.error(f"⚠️ {finding}")
                else:
                    st.success(f"✅ {finding}")

    # ==========================================
    # TAB 4: PORTFOLIO RISK ANALYTICS
    # ==========================================
    with tab_analytics:
        st.subheader("Global Portfolio Fraud Distribution & Merchant Risk")
        
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("Total Monitored Transactions", f"{len(df):,}")
        m_c2.metric("Flagged Syndicate Transactions", f"{df['is_fraud'].sum():,}")
        m_c3.metric("Global Fraud Exposure Rate", f"{df['is_fraud'].mean()*100:.1f}%")
        
        cat_counts = df.groupby(["merchant_category", "is_fraud"]).size().unstack(fill_value=0).reset_index()
        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(x=cat_counts["merchant_category"], y=cat_counts[0], name="Legitimate", marker_color="#10B981"))
        fig_cat.add_trace(go.Bar(x=cat_counts["merchant_category"], y=cat_counts[1], name="Flagged Fraud", marker_color="#EF4444"))
        fig_cat.update_layout(
            barmode="stack",
            height=360,
            xaxis_title="Merchant Category",
            yaxis_title="Transaction Count",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cat, use_container_width=True)


if __name__ == "__main__":
    main()
