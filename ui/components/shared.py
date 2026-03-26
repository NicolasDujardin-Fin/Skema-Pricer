"""
Shared UI components — CSS, chart builders, section headers, Q&A renderer.
"""

import json
import os

import plotly.graph_objects as go
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
/* ── Header: transparent, only keep sidebar toggle ── */
[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.5rem !important;
}
[data-testid="stHeader"] [data-testid="stToolbar"] {
    display: none !important;
}

/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #FAFAF9;
    font-size: 13px !important;
}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] td,
[data-testid="stAppViewContainer"] th,
[data-testid="stAppViewContainer"] label {
    font-size: 13px !important;
}
section[data-testid="stSidebar"] {
    background: #ECEAE7;
    border-right: 2px solid #C8C4BF;
    font-size: 11px !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    font-size: 11px !important;
}
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stToggle label {
    font-size: 11px !important;
    color: #555;
    margin-bottom: 0;
}
/* ── Sidebar input boxes: smaller text & height ── */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
    font-size: 11px !important;
    min-height: 0 !important;
    padding-top: 4px !important;
    padding-bottom: 4px !important;
}
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stDateInput {
    margin-bottom: -10px;
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 0.3rem !important;
}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    font-size: 11px !important;
    padding: 2px 6px !important;
}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
    font-size: 10px !important;
}

/* ── Force compact everywhere ── */
.main .block-container {
    padding-top: 0.8rem;
    font-size: 12px !important;
}
.main .block-container p,
.main .block-container span,
.main .block-container label,
.main .block-container td,
.main .block-container th,
.main .block-container li,
.main .block-container input,
.main .block-container select,
.main .block-container button {
    font-size: 12px !important;
}
.main .block-container input {
    padding: 3px 8px !important;
    height: auto !important;
}
.main .block-container [data-baseweb="input"] {
    min-height: 0 !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e8ecf0;
    border-radius: 4px;
    padding: 5px 8px 3px 8px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
[data-testid="stMetricLabel"] {
    font-size: 9px !important;
    color: #6B6B6B !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    line-height: 1.1 !important;
    margin-bottom: 0 !important;
    min-height: 0 !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 9px !important;
    line-height: 1.1 !important;
    margin: 0 !important;
}
[data-testid="stMetricValue"] {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #1D1D1B !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricValue"] div {
    font-size: 13px !important;
}

/* ── Hero metrics (first row) bigger ── */
.hero-metric [data-testid="stMetricValue"],
.hero-metric [data-testid="stMetricValue"] div {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #1D1D1B !important;
}
.hero-metric [data-testid="stMetric"] {
    border-left: 2px solid #E63329;
    background: #FEF7F6;
}

/* ── Section headers ── */
.section-header {
    font-size: 9px;
    font-weight: 500;
    color: #6B6B6B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.6rem;
    margin-bottom: 0.15rem;
    padding-bottom: 0.15rem;
    border-bottom: 1px solid #E0DEDA;
}

/* ── Reduce vertical gaps between elements ── */
.main .block-container [data-testid="stVerticalBlock"] > div {
    gap: 0.4rem !important;
}
.main .block-container [data-testid="stHorizontalBlock"] {
    gap: 0.4rem !important;
}

/* ── Expanders ── */
details[data-testid="stExpander"] {
    border: 1px solid #E0DEDA !important;
    border-radius: 4px !important;
    background: white !important;
}
details[data-testid="stExpander"] summary {
    font-weight: 500;
    font-size: 11px;
    color: #1D1D1B;
    padding: 6px 10px !important;
}
details[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 6px 10px !important;
}

/* ── Delta hedge bar ── */
.hedge-bar {
    background: #F7F6F5;
    border-left: 3px solid #E63329;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 0.73rem;
    color: #1D1D1B;
    margin-bottom: 0.5rem;
}
.hedge-bar b { color: #1D1D1B; }

/* ── Title styling ── */
.app-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1D1D1B;
    margin-bottom: 0;
    letter-spacing: -0.01em;
}
.app-badge {
    display: inline-block;
    background: #FDECEA;
    color: #E63329;
    font-size: 0.56rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
}
</style>
"""


def inject_css():
    """Inject the global Bloomberg-style CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PLOTLY CHART THEME
# ═══════════════════════════════════════════════════════════════════════════

PLOT_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#4a5568"),
    margin=dict(l=45, r=15, t=36, b=42),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left",
                font=dict(size=11)),
    xaxis=dict(gridcolor="#f0f0f0", linecolor="#ddd", linewidth=1,
               zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#f0f0f0", linecolor="#ddd", linewidth=1,
               zeroline=False, tickfont=dict(size=10)),
)


def make_line_chart(
    data: list,
    x_key: str,
    lines: list,
    title: str = "",
    x_label: str = "",
    vline: float = None,
    hline: float = None,
    height: int = 370,
    legend_below: bool = False,
) -> go.Figure:
    fig = go.Figure()
    xs = [d[x_key] for d in data]
    for dk, name, color in lines:
        ys = [d.get(dk) for d in data]
        if any(y is not None for y in ys):
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=name,
                line=dict(color=color, width=2.2),
                hovertemplate=f"{name}: %{{y:,.4f}}<extra></extra>",
            ))
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dot", line_color="#a0aec0", line_width=1,
                      annotation_text="ATM", annotation_font_size=10,
                      annotation_font_color="#a0aec0")
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dot", line_color="#a0aec0", line_width=1,
                      annotation_text="Par", annotation_font_size=10,
                      annotation_font_color="#a0aec0")
    layout = dict(**PLOT_LAYOUT)
    if legend_below:
        layout["margin"] = dict(l=45, r=15, t=60, b=42)
        layout["legend"] = dict(orientation="h", yanchor="top", y=-0.25,
                                xanchor="center", x=0.5, font=dict(size=11))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#2d3748"), x=0, xanchor="left"),
        xaxis_title=x_label,
        height=height,
        **layout,
    )
    return fig


def make_bar_chart(
    data: list,
    x_key: str,
    bars: list,
    title: str = "",
    x_label: str = "",
    stacked: bool = True,
    height: int = 370,
) -> go.Figure:
    fig = go.Figure()
    xs = [d[x_key] for d in data]
    for dk, name, color in bars:
        ys = [d.get(dk, 0) for d in data]
        fig.add_trace(go.Bar(x=xs, y=ys, name=name, marker_color=color,
                             marker_line_width=0))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#2d3748"), x=0, xanchor="left"),
        xaxis_title=x_label,
        barmode="stack" if stacked else "group",
        height=height,
        **PLOT_LAYOUT,
    )
    return fig


def section(label: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Q&A RENDERER
# ═══════════════════════════════════════════════════════════════════════════

_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "questions.json")
with open(_QUESTIONS_PATH, encoding="utf-8") as _f:
    _QA_DATA = json.load(_f)


def render_qa(section_key: str):
    """Render a Q&A section from questions.json."""
    qa_section = _QA_DATA.get(section_key)
    if not qa_section:
        return
    section(qa_section["title"])
    if qa_section.get("subtitle"):
        st.caption(qa_section["subtitle"])
    for i, item in enumerate(qa_section["questions"], 1):
        with st.expander(f"**Q{i}. {item['q']}**"):
            st.markdown(item["a"])
