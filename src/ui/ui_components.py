import streamlit as st

ICONS = {
    'high_risk': '🚨',
    'medium_risk': '⚠️',
    'low_risk': '✅',
    'queue': '📋',
    'network': '🕸️',
    'document': '📄'
}

def metric_card(label, value, icon=""):
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def badge(status):
    status_class = "badge-" + status.lower().replace(" ", "-")
    return f'<span class="badge {status_class}">{status}</span>'

def alert_row(row):
    """Fallback if we want to build a raw HTML queue, but st.dataframe is often better."""
    pass

def sar_textarea(label, placeholder):
    return st.text_area(label, placeholder=placeholder, height=200)

def style_network_figure(fig):
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, sans-serif", color="#1E2A38"),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig
