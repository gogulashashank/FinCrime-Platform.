import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import os
import yaml
import json
from pathlib import Path

# Must be the very first Streamlit command
st.set_page_config(page_title="FinCrime Platform | Investigator Workstation", layout="wide")

from src.ui.ui_components import metric_card, badge, sar_textarea, style_network_figure, ICONS

# Inject Custom CSS
try:
    st.markdown(f"<style>{Path('src/ui/styles.css').read_text()}</style>", unsafe_allow_html=True)
except Exception:
    pass

@st.cache_resource
def load_rule_catalog():
    path = 'configs/rules.yaml'
    if os.path.exists(path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('rules', [])
    return []

# Persistence logic for Case Management
CASE_STATE_FILE = 'data/synthetic/case_state.json'

def load_case_state():
    if os.path.exists(CASE_STATE_FILE):
        with open(CASE_STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_case_state(state_dict):
    os.makedirs(os.path.dirname(CASE_STATE_FILE), exist_ok=True)
    with open(CASE_STATE_FILE, 'w') as f:
        json.dump(state_dict, f)

@st.cache_data
def load_data():
    ml_path = 'data/synthetic/ml_scored_transactions.parquet'
    alerts_path = 'data/synthetic/fact_alerts.parquet'
    
    if not os.path.exists(ml_path):
        st.error(f"Data file not found at {ml_path}. Run data generation and ML pipelines first.")
        return pd.DataFrame()
        
    df = pd.read_parquet(ml_path)
    
    if os.path.exists(alerts_path):
        alerts = pd.read_parquet(alerts_path)[['transaction_id', 'risk_score', 'triggered_rules']]
        df = df.merge(alerts, on='transaction_id', how='left')
        
    df['risk_score'] = df.get('risk_score', pd.Series(0, index=df.index)).fillna(0)
    df['triggered_rules'] = df.get('triggered_rules', pd.Series('None', index=df.index)).fillna('None')
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp_fmt'] = df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
        df['alert_id'] = 'ALT-' + df['timestamp'].dt.strftime('%Y-%m') + '-' + df['transaction_id'].astype(str).str[-6:]
    else:
        df['alert_id'] = df['transaction_id']
        
    if 'combined_risk' not in df.columns:
        df['combined_risk'] = df[['risk_score', 'ml_risk_score']].max(axis=1)
        
    return df

def generate_network_graph(df_context, center_node):
    """Generate an annotated Plotly Network Graph"""
    G = nx.DiGraph()
    for _, row in df_context.iterrows():
        G.add_edge(row['sender_account_id'], row['receiver_account_id'], 
                   amount=f"£{row['amount']:,.2f}", 
                   date=row['timestamp_fmt'])
        
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    edge_texts = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        edge_texts.append(dict(
            x=(x0+x1)/2, y=(y0+y1)/2,
            text=edge[2]['amount'],
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="white"
        ))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Acct: {str(node)[-8:]}")
        
        if node == center_node:
            node_color.append('#d62728')
        else:
            node_color.append('#1f77b4')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        textposition="bottom center",
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=22,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=dict(text='<br>Entity Transaction Network (1-Degree)', font=dict(size=16)),
                showlegend=False,
                hovermode='closest',
                annotations=edge_texts,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
                
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#d62728'), name='Primary Subject'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#1f77b4'), name='Counterparty'))
    fig.update_layout(showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    
    # Apply custom styling
    fig = style_network_figure(fig)
    return fig

def format_rules(rules_str, rule_catalog):
    if rules_str == 'None' or not rules_str: return "None"
    
    rule_map = {r['id']: r['name'] for r in rule_catalog}
    try:
        rule_ids = json.loads(rules_str)
        return ", ".join([f"{rid}: {rule_map.get(rid, rid)}" for rid in rule_ids])
    except:
        return rules_str

def main():
    st.title("🛡️ FinCrime Investigator Workstation")
    st.markdown("An AI-Augmented Financial Crime Detection & Triage Platform.")
    
    df = load_data()
    rule_catalog = load_rule_catalog()
    
    if df.empty:
        return
        
    tab1, tab2 = st.tabs(["🚨 Alert Triage", "📚 Rules Catalogue"])
    
    with tab1:
        if 'case_statuses' not in st.session_state:
            st.session_state.case_statuses = load_case_state()
            
        df['status'] = df['alert_id'].map(lambda x: st.session_state.case_statuses.get(x, 'Open'))
            
        st.sidebar.header("Triage Filters")
        min_risk = st.sidebar.slider("Minimum Combined Risk Score", 0, 100, 75)
        
        all_jurisdictions = df['jurisdiction'].dropna().unique().tolist()
        sel_jurisdictions = st.sidebar.multiselect("Jurisdiction", all_jurisdictions, default=[])
        sel_status = st.sidebar.multiselect("Case Status", ["Open", "In Review", "Escalated", "Closed"], default=["Open", "In Review"])
        
        base_queue = df[df['combined_risk'] >= min_risk]
        
        alerts = base_queue.copy()
        if sel_jurisdictions:
            alerts = alerts[alerts['jurisdiction'].isin(sel_jurisdictions)]
        if sel_status:
            alerts = alerts[alerts['status'].isin(sel_status)]
            
        alerts = alerts.sort_values(by='combined_risk', ascending=False)
        
        st.sidebar.markdown(f"**Alerts in Current View:** {len(alerts)}")
        
        total_open = len(base_queue[base_queue['status'] == 'Open'])
        high_risk_count = len(base_queue[base_queue['combined_risk'] >= 90])
        escalated = len(base_queue[base_queue['status'] == 'Escalated'])
        avg_risk = base_queue['combined_risk'].mean() if not base_queue.empty else 0
        
        # UX Improvement: Wrap KPIs in metric_card components
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1: metric_card("Total Open Alerts", total_open, ICONS['queue'])
        with col_kpi2: metric_card("Critical Risk (>90)", high_risk_count, ICONS['high_risk'])
        with col_kpi3: metric_card("Escalated Cases", escalated, ICONS['medium_risk'])
        with col_kpi4: metric_card("Avg Queue Risk", f"{avg_risk:.1f}", ICONS['document'])
        
        if len(alerts) == 0:
            st.success("No alerts match the current filter criteria.")
        else:
            st.subheader("🚨 Alert Triage Queue")
            
            display_df = alerts.copy()
            display_df['amount_fmt'] = display_df['amount'].apply(lambda x: f"£{x:,.2f}")
            
            # Show analyst-friendly alert_id instead of raw transaction_id
            display_cols = ['alert_id', 'timestamp_fmt', 'sender_account_id', 'amount_fmt', 'combined_risk', 'status']
            st.dataframe(display_df[display_cols].head(100), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("🔎 Alert Investigation Panel")
            
            selected_alert_id = st.selectbox("Select Alert to Investigate", alerts['alert_id'].head(100).tolist())
            
            if selected_alert_id:
                txn_details = alerts[alerts['alert_id'] == selected_alert_id].iloc[0]
                
                # Render the badge in the UI
                st.markdown(f"Current Status: {badge(txn_details['status'])}", unsafe_allow_html=True)
                
                status_col1, status_col2 = st.columns([3, 1])
                with status_col2:
                    new_status = st.selectbox("Update Case Status", ["Open", "In Review", "Escalated", "Closed"], 
                                              index=["Open", "In Review", "Escalated", "Closed"].index(txn_details['status']))
                    if new_status != txn_details['status']:
                        st.session_state.case_statuses[selected_alert_id] = new_status
                        save_case_state(st.session_state.case_statuses) # Persist to JSON
                        st.rerun()
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### Transaction & Entity Context")
                    st.markdown(f"**Alert ID:** `{txn_details['alert_id']}`")
                    st.markdown(f"**Raw TXN ID:** `{txn_details['transaction_id']}`")
                    st.markdown(f"**Date:** `{txn_details['timestamp_fmt']}`")
                    st.markdown(f"**Amount:** £{txn_details['amount']:,.2f}")
                    st.markdown(f"**Sender Account:** `{txn_details['sender_account_id']}`")
                    st.markdown(f"**Receiver Account:** `{txn_details['receiver_account_id']}`")
                    
                    st.markdown("#### KYC Profile (Sender)")
                    risk_rating = txn_details.get('risk_rating', 'Unknown')
                    color = "red" if risk_rating == "High" else "orange" if risk_rating == "Medium" else "green"
                    st.markdown(f"**Risk Rating:** <span style='color:{color}'>{risk_rating}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Jurisdiction:** {txn_details.get('jurisdiction', 'Unknown')}")
                    
                    is_pep = txn_details.get('is_pep', False)
                    st.markdown(f"**PEP/Sanctions Flag:** {'🔴 YES' if is_pep else '🟢 NO'}")
                    st.markdown(f"**Onboarding Date:** `{txn_details.get('onboarding_date', 'N/A')}`")
                    st.markdown(f"**Last Review Date:** `{txn_details.get('last_review_date', 'N/A')}`")
                    
                    st.markdown("#### Historical Behavior (Deviation vs Baseline)")
                    st.markdown(f"- **Txn Count (1D / 7D / 30D):** {txn_details.get('txn_count_1d', 0)} / {txn_details.get('txn_count_7d', 0)} / {txn_details.get('txn_count_30d', 0)}")
                    st.markdown(f"- **Volume (1D / 7D / 30D):** £{txn_details.get('total_amount_1d', 0):,.2f} / £{txn_details.get('total_amount_7d', 0):,.2f} / £{txn_details.get('total_amount_30d', 0):,.2f}")
                    st.markdown(f"- **Velocity (In/Out Ratio):** {txn_details.get('velocity_in_out_ratio', 0):.2f}")

                with col2:
                    st.markdown("### Detection Intelligence")
                    
                    friendly_rules = format_rules(txn_details.get('triggered_rules', 'None'), rule_catalog)
                    
                    # Instead of metric card CSS manually, using native st.metric or a custom section component if preferred
                    # We'll use Streamlit's native metric, it looks good inside columns
                    st.metric("Rule Engine Score", f"{txn_details['risk_score']}/100")
                    st.markdown(f"**Triggered Typologies:** {friendly_rules}")
                    
                    st.markdown("---")
                                
                    ml_score = txn_details['ml_risk_score']
                    reasons = txn_details.get('ml_top_reasons', 'N/A')
                    
                    st.metric("ML Anomaly Score", f"{ml_score}/100")
                    st.markdown(f"**SHAP Explainability (Top Drivers):**<br> {reasons}", unsafe_allow_html=True)
                                
                st.markdown("---")
                st.markdown("### Transaction Network View")
                
                sender_id = txn_details['sender_account_id']
                context_txns = df[(df['sender_account_id'] == sender_id) | (df['receiver_account_id'] == sender_id)].head(50)
                
                fig = generate_network_graph(context_txns, center_node=sender_id)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 📝 NCA Suspicious Activity Report (SAR) Assistant")
                case_notes = sar_textarea("Investigator Notes (Description of Activity)", placeholder="Document your findings, source of funds, and rationale for escalation...")
                
                if st.button("Generate Structured SAR Draft"):
                    sar_narrative = f"""
### Suspicious Activity Report (SAR) Draft

**1. Subject Details**
- **Account ID:** {sender_id}
- **Jurisdiction:** {txn_details.get('jurisdiction', 'Unknown')}
- **PEP Status:** {'YES' if is_pep else 'NO'}

**2. Reason for Suspicion**
The account triggered automated AML detection systems combining deterministic rules (Score: {txn_details['risk_score']}) and ML anomaly detection (Score: {txn_details['ml_risk_score']}).
- **Typologies Identified:** {friendly_rules}

**3. Activity Description**
{case_notes if case_notes else '[No investigator notes provided]'}

**4. Supporting Evidence**
- **Flagged Transaction:** {txn_details['alert_id']} executed on {txn_details['timestamp_fmt']} for £{txn_details['amount']:,.2f}.
- **ML Anomaly Drivers:** {reasons}.
- **Behavioral Deviation:** 1D Volume: £{txn_details.get('total_amount_1d',0):,.2f} vs 30D Volume: £{txn_details.get('total_amount_30d',0):,.2f}.

**5. Accounts Involved**
- **Sender:** {sender_id}
- **Counterparty:** {txn_details['receiver_account_id']}

**6. Date Range**
- **Trigger Date:** {txn_details['timestamp_fmt']}

**7. NCA Escalation Recommendation**
Escalate for Level 2 EDD / Proceed with external SAR filing to the National Crime Agency.
                    """
                    st.info(sar_narrative)

    with tab2:
        st.subheader("📚 Detection Logic & Typology Catalogue")
        st.markdown("This catalogue provides transparency into the deterministic rule engine, mapping alerts to regulatory typologies (JMLSG, NCA) and explicit thresholds.")
        
        if rule_catalog:
            rules_df = pd.DataFrame(rule_catalog)
            display_rules = rules_df[['id', 'name', 'typology', 'threshold_logic', 'regulatory_reference', 'score', 'description']]
            display_rules.columns = ['Rule ID', 'Rule Name', 'Typology (JMLSG/NCA)', 'Threshold Logic', 'Regulatory Reference', 'Risk Score', 'Description']
            st.dataframe(display_rules, use_container_width=True, hide_index=True)
        else:
            st.info("No rules found in configs/rules.yaml")

if __name__ == "__main__":
    main()
