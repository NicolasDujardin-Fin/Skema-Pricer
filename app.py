"""
app.py — Streamlit GUI for the Skema Pricer.
Bloomberg-style professional layout.
"""

import datetime
import json
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bs_engine import (
    bs_greeks,
    bs_price,
    cash_delta_spot_ladder,
    delta_vol_ladder,
    gamma_spot_ladder,
    gamma_vol_ladder,
)
from bond_engine import (
    bond_price_from_ytm,
    bond_price_yield_curve,
    callable_bond_tree,
    callable_bond_tree_detail,
    callable_bond_yield_curve,
    yield_to_call,
)
from discount_engine import (
    discount_certificate_price,
    discount_payoff_data,
    dc_price_across_vols,
    dc_price_across_caps,
)
from bonus_engine import (
    bonus_certificate_price,
    bonus_payoff_data,
    bc_price_across_vols,
    bc_price_across_time,
)
from numerical_engine import american_binomial_tree
from rates_engine import price_forward

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Skema Pricer", layout="wide", page_icon="📈")

# -- Inject custom CSS for Bloomberg-style look --
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #fafbfc;
}
section[data-testid="stSidebar"] {
    background: #f0f2f5;
    border-right: 1px solid #dfe3e8;
}
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stDateInput label {
    font-size: 0.78rem;
    color: #555;
    margin-bottom: 0;
}
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stDateInput {
    margin-bottom: -8px;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e8ecf0;
    border-radius: 6px;
    padding: 12px 14px 8px 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    color: #8b95a5 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #1a1f36 !important;
}

/* ── Hero metrics (first row) bigger ── */
.hero-metric [data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #0b1222 !important;
}
.hero-metric [data-testid="stMetric"] {
    border-left: 3px solid #3182ce;
    background: #f7faff;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.82rem;
    font-weight: 600;
    color: #8b95a5;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1.2rem;
    margin-bottom: 0.4rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #e8ecf0;
}

/* ── Containers ── */
.block-container {
    padding-top: 1.5rem;
}

/* ── Expanders ── */
details[data-testid="stExpander"] {
    border: 1px solid #e8ecf0 !important;
    border-radius: 6px !important;
    background: white !important;
}
details[data-testid="stExpander"] summary {
    font-weight: 600;
    font-size: 0.85rem;
    color: #2d3748;
}

/* ── Delta hedge bar ── */
.hedge-bar {
    background: #f0f4f8;
    border-left: 3px solid #38a169;
    border-radius: 4px;
    padding: 8px 14px;
    font-size: 0.85rem;
    color: #2d3748;
    margin-bottom: 0.5rem;
}
.hedge-bar b { color: #1a202c; }

/* ── Title styling ── */
.app-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1a202c;
    margin-bottom: 0;
    letter-spacing: -0.01em;
}
.app-badge {
    display: inline-block;
    background: #ebf4ff;
    color: #3182ce;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PLOTLY CHART THEME
# ═══════════════════════════════════════════════════════════════════════════

_PLOT_LAYOUT = dict(
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


def _make_line_chart(
    data: list[dict],
    x_key: str,
    lines: list[tuple[str, str, str]],
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
    layout = dict(**_PLOT_LAYOUT)
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


def _make_bar_chart(
    data: list[dict],
    x_key: str,
    bars: list[tuple[str, str, str]],
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
        **_PLOT_LAYOUT,
    )
    return fig


def _section(label: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)


# Load Q&A from external file
_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
with open(_QUESTIONS_PATH, encoding="utf-8") as _f:
    _QA_DATA = json.load(_f)


def _render_qa(section_key: str):
    """Render a Q&A section from questions.json."""
    section = _QA_DATA.get(section_key)
    if not section:
        return
    _section(section["title"])
    if section.get("subtitle"):
        st.caption(section["subtitle"])
    for i, item in enumerate(section["questions"], 1):
        with st.expander(f"**Q{i}. {item['q']}**"):
            st.markdown(item["a"])


# ═══════════════════════════════════════════════════════════════════════════
# COMPUTE HELPERS (shared between vol / maturity axis)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _compute_greeks_across_vols(S, K, T, r, q, repo, n_lots, multiplier, option_type):
    data = []
    for sv in np.linspace(0.05, 0.60, 31):
        sv = max(0.001, float(sv))
        gv = bs_greeks(S, K, T, r, q, repo, sv, option_type)
        pc = bs_price(S, K, T, r, q, repo, sv, "call")
        pp = bs_price(S, K, T, r, q, repo, sv, "put")
        T1 = max(0.001, T - 1 / 365)
        d1 = bs_greeks(S, K, T1, r, q, repo, sv, option_type)["delta"]
        ds = 0.001
        du = bs_greeks(S, K, T, r, q, repo, sv + ds, option_type)["delta"]
        dd = bs_greeks(S, K, T, r, q, repo, max(0.001, sv - ds), option_type)["delta"]
        dr = 0.001
        pu = bs_price(S, K, T, r + dr, q, repo, sv, option_type)
        pd_ = bs_price(S, K, T, r - dr, q, repo, sv, option_type)
        data.append({
            "vol": round(sv * 100, 2),
            "call_price": round(pc, 4), "put_price": round(pp, 4),
            "cash_delta": round(n_lots * gv["delta"] * multiplier * S, 2),
            "cash_gamma": round(n_lots * gv["gamma"] * multiplier * S ** 2 / 100, 2),
            "cash_theta": round(n_lots * gv["theta"] * multiplier, 2),
            "cash_vega": round(n_lots * gv["vega"] * multiplier, 2),
            "cash_charm": round((d1 - gv["delta"]) * n_lots * multiplier * S, 2),
            "cash_vanna": round((du - dd) / (2 * ds) * n_lots * multiplier * S / 100, 2),
            "cash_rho": round((pu - pd_) / (2 * dr) * n_lots * multiplier / 100, 2),
        })
    return data


@st.cache_data(show_spinner=False)
def _compute_greeks_across_mats(S, K, r, q, repo, sigma, n_lots, multiplier, option_type):
    data = []
    for Tm in [i / 12.0 for i in range(1, 37)]:
        Tm = max(0.001, Tm)
        gm = bs_greeks(S, K, Tm, r, q, repo, sigma, option_type)
        pc = bs_price(S, K, Tm, r, q, repo, sigma, "call")
        pp = bs_price(S, K, Tm, r, q, repo, sigma, "put")
        T1 = max(0.001, Tm - 1 / 365)
        d1 = bs_greeks(S, K, T1, r, q, repo, sigma, option_type)["delta"]
        ds = 0.001
        du = bs_greeks(S, K, Tm, r, q, repo, sigma + ds, option_type)["delta"]
        dd = bs_greeks(S, K, Tm, r, q, repo, max(0.001, sigma - ds), option_type)["delta"]
        dr = 0.001
        pu = bs_price(S, K, Tm, r + dr, q, repo, sigma, option_type)
        pd_ = bs_price(S, K, Tm, r - dr, q, repo, sigma, option_type)
        data.append({
            "maturity": round(Tm, 2),
            "call_price": round(pc, 4), "put_price": round(pp, 4),
            "cash_delta": round(n_lots * gm["delta"] * multiplier * S, 2),
            "cash_gamma": round(n_lots * gm["gamma"] * multiplier * S ** 2 / 100, 2),
            "cash_theta": round(n_lots * gm["theta"] * multiplier, 2),
            "cash_vega": round(n_lots * gm["vega"] * multiplier, 2),
            "cash_charm": round((d1 - gm["delta"]) * n_lots * multiplier * S, 2),
            "cash_vanna": round((du - dd) / (2 * ds) * n_lots * multiplier * S / 100, 2),
            "cash_rho": round((pu - pd_) / (2 * dr) * n_lots * multiplier / 100, 2),
        })
    return data


@st.cache_data(show_spinner=False)
def _cached_callable_yield_curve(face, coupon, mat, freq, call_price, first_call, rate_vol):
    return callable_bond_yield_curve(face, coupon, mat, freq, call_price, first_call, rate_vol)


@st.cache_data(show_spinner=False)
def _cached_bond_yield_curve(face, coupon, mat, freq):
    return bond_price_yield_curve(face, coupon, mat, freq)


@st.cache_data(show_spinner=False)
def _cached_dc_vols(S, cap, T, r, q, parity):
    return dc_price_across_vols(S, cap, T, r, q, parity)


@st.cache_data(show_spinner=False)
def _cached_dc_caps(S, T, r, q, sigma, parity):
    return dc_price_across_caps(S, T, r, q, sigma, parity)


@st.cache_data(show_spinner=False)
def _cached_bc_vols(S, bonus, barrier, T, r, q, cap, parity, sigma_call):
    return bc_price_across_vols(S, bonus, barrier, T, r, q, cap, parity, sigma_call=sigma_call)


@st.cache_data(show_spinner=False)
def _cached_bc_time(S, bonus, barrier, r, q, sigma_put, cap, parity, sigma_call):
    return bc_price_across_time(S, bonus, barrier, r, q, sigma_put, cap, parity, sigma_call=sigma_call)


@st.cache_data(show_spinner=False)
def _raw_greeks_spot_vol(S, K, T, r, q, repo, option_type, S_min, S_max):
    """Delta, Gamma, Vega vs Spot for 3 different vols."""
    vols = [0.10, 0.20, 0.40]
    spots = np.linspace(S_min, S_max, 51)
    data = []
    for s in spots:
        row = {"spot": round(float(s), 2)}
        for sv in vols:
            g = bs_greeks(float(s), K, T, r, q, repo, sv, option_type)
            tag = f"{int(sv*100)}%"
            row[f"delta_{tag}"] = round(g["delta"], 6)
            row[f"gamma_{tag}"] = round(g["gamma"], 6)
            row[f"vega_{tag}"] = round(g["vega"], 6)
        data.append(row)
    return data, vols


@st.cache_data(show_spinner=False)
def _raw_greeks_time(S, K, T_max, r, q, repo, sigma, option_type):
    """Delta, Gamma, Vega vs Time for OTM / ATM / ITM."""
    moneyness = [
        (round(K * 0.90, 2), "OTM"),
        (round(K * 1.00, 2), "ATM"),
        (round(K * 1.10, 2), "ITM"),
    ] if option_type == "call" else [
        (round(K * 1.10, 2), "OTM"),
        (round(K * 1.00, 2), "ATM"),
        (round(K * 0.90, 2), "ITM"),
    ]
    times = np.linspace(0.02, min(T_max, 3.0), 51)
    data = []
    for t in times:
        t = max(0.001, float(t))
        row = {"time": round(t, 3)}
        for spot, tag in moneyness:
            g = bs_greeks(spot, K, t, r, q, repo, sigma, option_type)
            row[f"delta_{tag}"] = round(g["delta"], 6)
            row[f"gamma_{tag}"] = round(g["gamma"], 6)
            row[f"vega_{tag}"] = round(g["vega"], 6)
        data.append(row)
    return data, moneyness


@st.cache_data(show_spinner=False)
def _compute_spot_ladder(S_min, S_max, K, T, r, q, repo, sigma, n_lots, multiplier, option_type):
    """Compute enriched spot ladder with all cash Greeks."""
    spot_data = cash_delta_spot_ladder(
        S_min, S_max, 21, K, T, r, q, repo, sigma, n_lots, multiplier,
    )
    for row in spot_data:
        Ss = row["spot"]
        gs = bs_greeks(Ss, K, T, r, q, repo, sigma, option_type)
        row["cash_gamma"] = n_lots * gs["gamma"] * multiplier * Ss ** 2 / 100
        row["cash_theta"] = n_lots * gs["theta"] * multiplier
        row["cash_vega"] = n_lots * gs["vega"] * multiplier
        T1s = max(0.001, T - 1 / 365)
        gs1 = bs_greeks(Ss, K, T1s, r, q, repo, sigma, option_type)
        row["cash_charm"] = (gs1["delta"] - gs["delta"]) * n_lots * multiplier * Ss
        dss = 0.001
        d_ups = bs_greeks(Ss, K, T, r, q, repo, sigma + dss, option_type)["delta"]
        d_dns = bs_greeks(Ss, K, T, r, q, repo, max(0.001, sigma - dss), option_type)["delta"]
        row["cash_vanna"] = (d_ups - d_dns) / (2 * dss) * n_lots * multiplier * Ss / 100
        drs = 0.001
        p_ups = bs_price(Ss, K, T, r + drs, q, repo, sigma, option_type)
        p_dns = bs_price(Ss, K, T, r - drs, q, repo, sigma, option_type)
        row["cash_rho"] = (p_ups - p_dns) / (2 * drs) * n_lots * multiplier / 100
    return spot_data


# ═══════════════════════════════════════════════════════════════════════════
# OPTIONS TAB
# ═══════════════════════════════════════════════════════════════════════════

def options_tab():

    # ── SIDEBAR INPUTS ──
    with st.sidebar:
        st.markdown('<p class="app-title">Parameters</p>', unsafe_allow_html=True)
        st.caption("Market & Contract")

        S = st.number_input("Spot (S)", value=100.0, step=1.0, format="%.2f")
        K = st.number_input("Strike (K)", value=100.0, step=1.0, format="%.2f")

        mat_date = st.date_input("Maturity",
                                 value=datetime.date.today() + datetime.timedelta(days=365))
        T = max(0.001, (mat_date - datetime.date.today()).days / 365.25)
        st.caption(f"T = {T:.4f} y")

        c1, c2 = st.columns(2)
        r = c1.number_input("r %", value=5.0, step=0.1, format="%.2f") / 100
        q = c2.number_input("q %", value=2.0, step=0.1, format="%.2f") / 100

        c3, c4 = st.columns(2)
        repo = c3.number_input("repo %", value=0.0, step=0.1, format="%.2f") / 100
        sigma = max(0.001, c4.number_input("σ %", value=20.0, step=0.5, format="%.2f") / 100)

        st.caption("Position")
        c5, c6 = st.columns(2)
        n_lots = c5.number_input("Lots", value=1.0, step=1.0, format="%.1f")
        multiplier = c6.number_input("Mult.", value=100.0, step=1.0, format="%.0f")

        c7, c8 = st.columns(2)
        tick_size = c7.number_input("Tick", value=0.01, step=0.01, format="%.4f")
        option_type = c8.selectbox("Type", ["call", "put"])

    # ── COMPUTE ──
    try:
        g = bs_greeks(S, K, T, r, q, repo, sigma, option_type)
        p = bs_price(S, K, T, r, q, repo, sigma, option_type)
        fwd = price_forward(S, r, q, repo, T)
    except Exception:
        g = {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}
        p = 0.0
        fwd = 0.0

    cd = n_lots * g["delta"] * multiplier * S
    cg = n_lots * g["gamma"] * multiplier * S ** 2 / 100
    ct = n_lots * g["theta"] * multiplier
    cv = n_lots * g["vega"] * multiplier
    n_shares = n_lots * g["delta"] * multiplier

    dt_1d = 1 / 365
    T1 = max(0.001, T - dt_1d)
    try:
        d_tm1 = bs_greeks(S, K, T1, r, q, repo, sigma, option_type)["delta"]
        cash_charm = (d_tm1 - g["delta"]) * n_lots * multiplier * S
        delta_tx = (d_tm1 - g["delta"]) * n_lots * multiplier
    except Exception:
        cash_charm = delta_tx = 0.0

    try:
        ds = 0.001
        d_up = bs_greeks(S, K, T, r, q, repo, sigma + ds, option_type)["delta"]
        d_dn = bs_greeks(S, K, T, r, q, repo, max(0.001, sigma - ds), option_type)["delta"]
        cash_vanna = (d_up - d_dn) / (2 * ds) * n_lots * multiplier * S / 100
    except Exception:
        cash_vanna = 0.0

    try:
        dr = 0.001
        p_up = bs_price(S, K, T, r + dr, q, repo, sigma, option_type)
        p_dn = bs_price(S, K, T, r - dr, q, repo, sigma, option_type)
        rho_unit = (p_up - p_dn) / (2 * dr) / 100
        cash_rho = (p_up - p_dn) / (2 * dr) * n_lots * multiplier / 100
    except Exception:
        rho_unit = cash_rho = 0.0

    # ── PRIMARY METRICS (hero row) ──
    _section("Pricing")
    st.markdown('<div class="hero-metric">', unsafe_allow_html=True)
    h1, h2, h3, _sp, h4, h5 = st.columns([1, 1, 1, 0.3, 1, 1])
    h1.metric("BS Price", f"{p:.4f}")
    h2.metric("Forward", f"{fwd:.4f}")
    h3.metric("Cash Delta", f"{cd:,.2f}")
    h4.metric("Delta Hedge", f"{n_shares:,.1f} shrs")
    h5.metric("Δ t+1d", f"{delta_tx:,.2f} shrs/day")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECONDARY GREEKS (compact grid) ──
    _section("Cash Greeks")
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    g1.metric("Gamma / 1%", f"{cg:,.2f}")
    g2.metric("Theta / day", f"{ct:,.2f}")
    g3.metric("Vega / 1%", f"{cv:,.2f}")
    g4.metric("Charm / day", f"{cash_charm:,.2f}")
    g5.metric("Vanna / 1%", f"{cash_vanna:,.2f}")
    g6.metric("Rho / 1%", f"{cash_rho:,.2f}")

    st.markdown("")  # breathing

    # ── RISK ANALYSIS (expanders) ──
    with st.expander("**Gamma PnL Calculator**", expanded=True):
        rc1, rc2, rc3, rc4, rc5 = st.columns([1, 1, 1, 1, 1])
        spot_move_pct = rc1.number_input("Spot move %", value=1.0, step=0.5,
                                          format="%.2f", key="spot_move")
        new_cd = cg * spot_move_pct
        gamma_pnl = 0.5 * new_cd * (spot_move_pct / 100)
        rc2.metric("New Δ Cash", f"{new_cd:,.0f}")
        rc3.metric("Gamma PnL", f"{gamma_pnl:,.2f}")
        iv_for_move = rc4.number_input("IV %", value=20.0, step=1.0,
                                        format="%.1f", key="iv_move")
        daily_move = iv_for_move / math.sqrt(252)
        rc5.metric("Daily Move", f"{daily_move:.2f}%")

    with st.expander("**Trading Shortcuts**", expanded=False):
        tc1, tc2, tc3, tc4 = st.columns(4)

        try:
            be_spot = K + p if option_type == "call" else K - p
            be_pct = (be_spot / S - 1) * 100
            tc1.metric("BE Spot", f"{be_spot:.2f}", delta=f"{be_pct:+.2f}%")
        except Exception:
            tc1.metric("BE Spot", "—")

        try:
            gamma_v = g["gamma"]
            theta_v = g["theta"]
            if gamma_v > 0:
                dm = math.sqrt(2 * abs(theta_v) / gamma_v)
                be_vol = (dm / S) * math.sqrt(252) * 100
                s_lo, s_hi = S - dm, S + dm
                ticks = dm / tick_size if tick_size > 0 else 0
                tc2.metric("BE Vol (realized)", f"{be_vol:.2f}%")
                tc2.caption(f"[{s_lo:.2f}, {s_hi:.2f}] · {ticks:.0f} ticks")
            else:
                tc2.metric("BE Vol (realized)", "—")
        except Exception:
            tc2.metric("BE Vol (realized)", "—")

        try:
            ratio = cg / abs(ct) if abs(ct) > 1e-10 else 0
            tc3.metric("Gamma / Theta", f"{ratio:.2f}")
        except Exception:
            tc3.metric("Gamma / Theta", "—")

        try:
            if cg > 0:
                m = math.sqrt(200 * abs(ct) / cg)
                tc4.metric("Theta Earn Move", f"{m:.2f}%")
            else:
                tc4.metric("Theta Earn Move", "—")
        except Exception:
            tc4.metric("Theta Earn Move", "—")

    with st.expander("**Quick Calc — Gamma → Theta Bill**", expanded=False):
        qc1, qc2, qc3, qc4 = st.columns([2, 1, 2, 1])
        qa_gamma = qc1.number_input("Gamma $ notional", value=20_000_000.0,
                                     step=1_000_000.0, format="%.0f", key="qa_g")
        qa_vol = qc2.number_input("Vol %", value=22.0, step=1.0,
                                   format="%.1f", key="qa_v")
        qa_theta = qa_gamma * qa_vol ** 2 / 50_400
        qc3.metric("θ / day", f"${qa_theta:,.0f}")
        qc3.caption("= CG × σ² / 50,400")
        qa_dm = qa_vol / math.sqrt(252)
        qc4.metric("Daily Move", f"{qa_dm:.2f}%")

    # ── AMERICAN vs EUROPEAN ──
    with st.expander("**Early Exercise Analysis — American Style**", expanded=False):
        ac1, ac2 = st.columns([3, 1])
        with ac2:
            bin_steps = st.slider("Tree steps", 10, 500, 100, step=10, key="bin_steps")

        try:
            b = r - q - repo
            am = american_binomial_tree(S, K, T, r, b, sigma, bin_steps, option_type)
        except Exception:
            am = {"price": 0.0, "eu_price": 0.0, "delta": 0.0}

        prem = am["price"] - am["eu_price"]
        prem_pct = (prem / am["eu_price"] * 100) if am["eu_price"] > 0.0001 else 0

        with ac1:
            am_df = pd.DataFrame({
                "": ["European Price (tree)", "American Price (CRR)",
                     "Early Exercise Premium", "Premium", "American Delta"],
                "Value": [f"{am['eu_price']:.4f}", f"**{am['price']:.4f}**",
                          f"{prem:.4f}", f"{prem_pct:.2f}%", f"{am['delta']:.6f}"],
            })
            st.dataframe(am_df, hide_index=True, use_container_width=True)

            if prem > 0.0001:
                reason = ("Deep ITM put: remaining time value < interest earned by exercising "
                          "now and collecting K."
                          if option_type == "put" else
                          "Call with dividend: dividend lost by not holding the underlying "
                          "exceeds remaining time value.")
                st.warning(reason, icon="⚠️")

    # ── GREEKS TABLE ──
    _section("Unit Greeks")
    greeks_df = pd.DataFrame({
        "Greek": ["Delta", "Gamma", "Vega /1%", "Theta /day", "Rho /1%"],
        "Value": [f"{g['delta']:.6f}", f"{g['gamma']:.6f}", f"{g['vega']:.6f}",
                  f"{g['theta']:.6f}", f"{rho_unit:.6f}"],
    })
    st.dataframe(greeks_df, hide_index=True, width=360)

    st.markdown("")  # breathing

    # ── CHARTS ──
    _section("Charts")

    # Compute spot ladder
    center = K
    half = max(0.30 * center, abs(S - K) * 1.3)
    S_min, S_max = round(center - half, 4), round(center + half, 4)

    try:
        spot_data = _compute_spot_ladder(S_min, S_max, K, T, r, q, repo, sigma,
                                          n_lots, multiplier, option_type)
    except Exception:
        spot_data = []

    _CHART_MAP = {
        "Price": [("call_price", "Call", "#3182ce"), ("put_price", "Put", "#e53e3e")],
        "Cash Delta": [("call_cash_delta", "Call Cash$", "#38a169"),
                       ("put_cash_delta", "Put Cash$", "#dd6b20")],
        "Cash Gamma": [("cash_gamma", "Cash Gamma", "#805ad5")],
        "Cash Theta": [("cash_theta", "Cash Theta", "#c05621")],
        "Cash Vega": [("cash_vega", "Cash Vega", "#2b6cb0")],
        "Cash Charm": [("cash_charm", "Cash Charm", "#276749")],
        "Cash Vanna": [("cash_vanna", "Cash Vanna", "#97266d")],
        "Cash Rho": [("cash_rho", "Cash Rho", "#b83280")],
    }

    _CHART2_MAP = {
        "Price": [("call_price", "Call", "#3182ce"), ("put_price", "Put", "#e53e3e")],
        "Cash Delta": [("cash_delta", "Cash Delta", "#38a169")],
        "Cash Gamma": [("cash_gamma", "Cash Gamma", "#805ad5")],
        "Cash Theta": [("cash_theta", "Cash Theta", "#c05621")],
        "Cash Vega": [("cash_vega", "Cash Vega", "#2b6cb0")],
        "Cash Charm": [("cash_charm", "Cash Charm", "#276749")],
        "Cash Vanna": [("cash_vanna", "Cash Vanna", "#97266d")],
        "Cash Rho": [("cash_rho", "Cash Rho", "#b83280")],
    }

    ch1, _sp, ch2 = st.columns([5, 0.3, 5])

    with ch1:
        sel_chart1 = st.selectbox("vs Spot", list(_CHART_MAP.keys()), key="ch1")
        if spot_data:
            fig1 = _make_line_chart(spot_data, "spot", _CHART_MAP[sel_chart1],
                                    title=f"{sel_chart1} vs Spot",
                                    x_label="Spot", vline=K)
            st.plotly_chart(fig1, use_container_width=True)

    with ch2:
        sc2a, sc2b = st.columns(2)
        sel_chart2 = sc2a.selectbox("Chart", list(_CHART2_MAP.keys()), key="ch2")
        sel_axis2 = sc2b.selectbox("vs", ["Vol", "Maturity"], key="ch2ax")

        try:
            if sel_axis2 == "Vol":
                vol_data = _compute_greeks_across_vols(
                    S, K, T, r, q, repo, n_lots, multiplier, option_type)
                fig2 = _make_line_chart(vol_data, "vol", _CHART2_MAP[sel_chart2],
                                        title=f"{sel_chart2} vs Vol", x_label="Vol (%)")
            else:
                mat_data = _compute_greeks_across_mats(
                    S, K, r, q, repo, sigma, n_lots, multiplier, option_type)
                fig2 = _make_line_chart(mat_data, "maturity", _CHART2_MAP[sel_chart2],
                                        title=f"{sel_chart2} vs Maturity",
                                        x_label="Maturity (y)")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.error("Error computing chart data")

    # ── RAW GREEKS: Impact of Vol & Time on Delta, Gamma, Vega ──
    st.markdown("")
    _section("Unit Greeks Sensitivity — Impact of Volatility & Time to Maturity")

    try:
        raw_sv_data, vols_used = _raw_greeks_spot_vol(S, K, T, r, q, repo, option_type, S_min, S_max)
        raw_time_data, moneyness_used = _raw_greeks_time(S, K, T, r, q, repo, sigma, option_type)
    except Exception:
        raw_sv_data, vols_used = [], [0.10, 0.20, 0.40]
        raw_time_data, moneyness_used = [], []

    # Color palettes
    _VOL_COLORS = {"10%": "#3182ce", "20%": "#805ad5", "40%": "#e53e3e"}
    _MON_COLORS = {"OTM": "#e53e3e", "ATM": "#3182ce", "ITM": "#38a169"}

    # Row 1: Delta
    d_c1, _sp1, d_c2 = st.columns([5, 0.3, 5])
    if raw_sv_data:
        with d_c1:
            lines = [(f"delta_{int(v*100)}%", f"σ = {int(v*100)}%", _VOL_COLORS[f"{int(v*100)}%"])
                     for v in vols_used]
            fig = _make_line_chart(raw_sv_data, "spot", lines,
                                   title="Delta vs Spot — by Volatility",
                                   x_label="Spot", vline=K, height=340,
                                   legend_below=True)
            st.plotly_chart(fig, use_container_width=True)
    if raw_time_data:
        with d_c2:
            lines = [(f"delta_{tag}", f"{tag} (S={s:.0f})", _MON_COLORS[tag])
                     for s, tag in moneyness_used]
            fig = _make_line_chart(raw_time_data, "time", lines,
                                   title="Delta vs Time — by Moneyness",
                                   x_label="Time to Maturity (y)", height=340,
                                   legend_below=True)
            st.plotly_chart(fig, use_container_width=True)

    # Row 2: Gamma
    g_c1, _sp2, g_c2 = st.columns([5, 0.3, 5])
    if raw_sv_data:
        with g_c1:
            lines = [(f"gamma_{int(v*100)}%", f"σ = {int(v*100)}%", _VOL_COLORS[f"{int(v*100)}%"])
                     for v in vols_used]
            fig = _make_line_chart(raw_sv_data, "spot", lines,
                                   title="Gamma vs Spot — by Volatility",
                                   x_label="Spot", vline=K, height=340,
                                   legend_below=True)
            st.plotly_chart(fig, use_container_width=True)
    if raw_time_data:
        with g_c2:
            lines = [(f"gamma_{tag}", f"{tag} (S={s:.0f})", _MON_COLORS[tag])
                     for s, tag in moneyness_used]
            fig = _make_line_chart(raw_time_data, "time", lines,
                                   title="Gamma vs Time — by Moneyness",
                                   x_label="Time to Maturity (y)", height=340,
                                   legend_below=True)
            st.plotly_chart(fig, use_container_width=True)

    # Row 3: Vega
    v_c1, _sp3, v_c2 = st.columns([5, 0.3, 5])
    if raw_sv_data:
        with v_c1:
            lines = [(f"vega_{int(v*100)}%", f"σ = {int(v*100)}%", _VOL_COLORS[f"{int(v*100)}%"])
                     for v in vols_used]
            fig = _make_line_chart(raw_sv_data, "spot", lines,
                                   title="Vega vs Spot — by Volatility",
                                   x_label="Spot", vline=K, height=340,
                                   legend_below=True)
            st.plotly_chart(fig, use_container_width=True)
    if raw_time_data:
        with v_c2:
            lines = [(f"vega_{tag}", f"{tag} (S={s:.0f})", _MON_COLORS[tag])
                     for s, tag in moneyness_used]
            fig = _make_line_chart(raw_time_data, "time", lines,
                                   title="Vega vs Time — by Moneyness",
                                   x_label="Time to Maturity (y)", height=340,
                                   legend_below=True)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("")
    _render_qa("options")


# ═══════════════════════════════════════════════════════════════════════════
# BONDS TAB
# ═══════════════════════════════════════════════════════════════════════════

def _build_tree_svg(nodes: list[dict], n: int) -> str:
    margin_x, margin_y = 60, 40
    node_dx, node_dy = 130, 56
    w = margin_x * 2 + node_dx * n
    h = margin_y * 2 + node_dy * n
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
             f'style="background:#fff;font-family:Consolas,monospace;">']

    def cx(step): return margin_x + step * node_dx
    def cy(step, j): return h / 2 - (2 * j - step) * node_dy / 2

    for step in range(n):
        for j in range(step + 1):
            x1, y1 = cx(step), cy(step, j)
            for jj in (j + 1, j):
                x2, y2 = cx(step + 1), cy(step + 1, jj)
                parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                             f'stroke="#cbd5e0" stroke-width="1"/>')

    for nd in nodes:
        x, y = cx(nd["step"]), cy(nd["step"], nd["j"])
        called = nd["called"]
        fill = "#fee2e2" if called else "#ebf8ff"
        stroke = "#e53e3e" if called else "#3182ce"
        bw, bh = 100, 42
        parts.append(f'<rect x="{x - bw // 2}" y="{y - bh // 2}" width="{bw}" height="{bh}" '
                     f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        parts.append(f'<text x="{x}" y="{y - 5}" text-anchor="middle" font-size="10" '
                     f'fill="#718096">r={nd["rate"]:.1f}%</text>')
        label = "CALLED" if called else f'{nd["callable"]:,.1f}'
        parts.append(f'<text x="{x}" y="{y + 12}" text-anchor="middle" font-size="11" '
                     f'font-weight="600" fill="{stroke}">{label}</text>')

    ly = h - 16
    parts.append(f'<rect x="10" y="{ly - 10}" width="12" height="12" rx="2" '
                 f'fill="#ebf8ff" stroke="#3182ce"/>')
    parts.append(f'<text x="26" y="{ly}" font-size="10" fill="#4a5568">Not called</text>')
    parts.append(f'<rect x="120" y="{ly - 10}" width="12" height="12" rx="2" '
                 f'fill="#fee2e2" stroke="#e53e3e"/>')
    parts.append(f'<text x="136" y="{ly}" font-size="10" fill="#4a5568">'
                 f'Called (capped at call price)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def bonds_tab():
    with st.sidebar:
        st.markdown('<p class="app-title">Bond Parameters</p>', unsafe_allow_html=True)

        bond_face = st.number_input("Face", value=1000.0, step=100.0, format="%.0f",
                                     key="b_face")
        bond_coupon = st.number_input("Coupon %", value=5.0, step=0.25, format="%.2f",
                                       key="b_cpn")
        bc1, bc2 = st.columns(2)
        bond_maturity = bc1.number_input("Mat. (y)", value=10.0, step=0.5, format="%.1f",
                                          min_value=0.5, key="b_mat")
        bond_ytm = bc2.number_input("YTM %", value=5.0, step=0.25, format="%.2f",
                                     min_value=0.01, key="b_ytm")
        bc3, bc4 = st.columns(2)
        bond_settlement_days = bc3.number_input("Settl. d", value=0.0, step=1.0,
                                                 format="%.0f", min_value=0.0, key="b_sd")
        bond_freq = bc4.selectbox("Freq", [1, 2, 4], index=1, key="b_freq")

        st.caption("Risk")
        bond_notional = st.number_input("Notional", value=1_000_000.0, step=100_000.0,
                                         format="%.0f", key="b_not")
        bond_bp_shift = st.number_input("Shift (bp)", value=1.0, step=1.0,
                                         format="%.1f", key="b_bp")

        st.caption("Callable")
        bond_callable = st.toggle("Callable", value=False, key="b_call")
        if bond_callable:
            bcc1, bcc2 = st.columns(2)
            bond_call_price = bcc1.number_input("Call px", value=1000.0, step=10.0,
                                                  format="%.0f", key="b_cpx")
            bond_first_call = bcc2.number_input("1st call y", value=5.0, step=0.5,
                                                  format="%.1f", min_value=0.5, key="b_fc")
            bond_rate_vol = st.number_input("Rate vol %", value=20.0, step=1.0,
                                              format="%.1f", min_value=0.0, key="b_rv")
        else:
            bond_call_price = bond_face
            bond_first_call = bond_maturity / 2
            bond_rate_vol = 20.0

    # ── Compute ──
    effective_mat = max(0.01, bond_maturity - bond_settlement_days / 365.0)
    try:
        res = bond_price_from_ytm(bond_face, bond_coupon, effective_mat, bond_ytm, bond_freq)
    except Exception:
        res = {"dirty_price": 0, "clean_price": 0, "accrued_interest": 0,
               "macaulay_duration": 0, "modified_duration": 0, "convexity": 0, "cashflows": []}

    dv01 = res["modified_duration"] * res["dirty_price"] * 0.0001
    pv01 = dv01 * (bond_notional / bond_face)
    dy = bond_bp_shift / 10000
    n_bonds = bond_notional / bond_face
    pnl = (-res["modified_duration"] * dy + 0.5 * res["convexity"] * dy ** 2) * res["dirty_price"] * n_bonds

    # ── Pricing ──
    _section("Bond Pricing")
    st.markdown('<div class="hero-metric">', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("Dirty Price", f"{res['dirty_price']:.2f}")
    p2.metric("Clean Price", f"{res['clean_price']:.2f}")
    p3.metric("Accrued Interest", f"{res['accrued_interest']:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    _section("Duration & Convexity")
    d1, d2, d3 = st.columns(3)
    d1.metric("Macaulay Duration", f"{res['macaulay_duration']:.4f} y")
    d2.metric("Modified Duration", f"{res['modified_duration']:.4f} y")
    d3.metric("Convexity", f"{res['convexity']:.4f}")

    _section("Risk Sensitivity")
    r1, r2, r3 = st.columns(3)
    r1.metric("DV01 (per bond)", f"{dv01:.4f}")
    r2.metric("PV01 (notional)", f"{pv01:,.2f}")
    r3.metric(f"PnL ({bond_bp_shift:+.0f} bp)", f"{pnl:,.2f}")

    st.markdown("")

    # ── Callable ──
    if bond_callable:
        _section("Callable Bond Analysis")
        first_call_adj = min(bond_first_call, effective_mat)
        try:
            call_res = callable_bond_tree(
                bond_face, bond_coupon, effective_mat, bond_ytm, bond_freq,
                bond_call_price, first_call_adj, bond_rate_vol,
            )
        except Exception:
            call_res = {"straight_price": 0, "callable_price": 0, "option_value": 0}

        try:
            ytc = yield_to_call(bond_face, bond_coupon, first_call_adj,
                                bond_call_price, bond_freq, res["dirty_price"])
            ytc_str = f"{ytc:.4f}%" if ytc is not None else "—"
        except Exception:
            ytc_str = "—"
            ytc = None

        ytw = min(bond_ytm, ytc) if ytc is not None else bond_ytm
        ytw_str = f"{ytw:.4f}%"

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Callable Price", f"{call_res['callable_price']:.4f}")
        cc2.metric("Option Value", f"{call_res['option_value']:.4f}")
        cc3.metric("Yield to Call", ytc_str)
        cc4.metric("Yield to Worst", ytw_str)

        with st.expander("**Binomial Rate Tree**", expanded=False):
            tree_steps = st.select_slider("Steps", options=[3, 4, 5, 6, 7, 8],
                                           value=6, key="tree_steps")
            try:
                nodes = callable_bond_tree_detail(
                    bond_face, bond_coupon, effective_mat, bond_ytm, bond_freq,
                    bond_call_price, first_call_adj, bond_rate_vol, tree_steps,
                )
                svg = _build_tree_svg(nodes, tree_steps)
                st.html(svg)
            except Exception as e:
                st.error(f"Error building tree: {e}")

    st.markdown("")

    # ── Charts ──
    _section("Charts")
    ch1, _sp, ch2 = st.columns([5, 0.3, 5])

    with ch1:
        try:
            if bond_callable:
                yc_data = _cached_callable_yield_curve(
                    bond_face, bond_coupon, effective_mat, bond_freq,
                    bond_call_price, min(bond_first_call, effective_mat), bond_rate_vol,
                )
                lines = [("straight", "Straight", "#3182ce"),
                         ("callable", "Callable", "#e53e3e")]
            else:
                raw = _cached_bond_yield_curve(bond_face, bond_coupon, effective_mat, bond_freq)
                yc_data = [{"ytm": d["ytm"], "straight": d["price"]} for d in raw]
                lines = [("straight", "Straight", "#3182ce")]

            fig_yc = _make_line_chart(yc_data, "ytm", lines,
                                      title="Bond Price vs Yield",
                                      x_label="YTM (%)", hline=bond_face)
            st.plotly_chart(fig_yc, use_container_width=True)
        except Exception:
            st.error("Error computing yield curve")

    with ch2:
        try:
            cfs = res["cashflows"]
            pv_data = []
            for cf in cfs:
                is_last = cf["type"] == "coupon+principal"
                if is_last:
                    y_per = bond_ytm / 100 / bond_freq
                    cpv = round(cf["pv"] - bond_face / (1 + y_per) ** (cf["t"] * bond_freq), 2)
                    ppv = round(cf["pv"] - cpv, 2)
                else:
                    cpv = round(cf["pv"], 2)
                    ppv = 0.0
                pv_data.append({"t": cf["t"], "coupon_pv": cpv, "principal_pv": ppv})

            fig_pv = _make_bar_chart(
                pv_data, "t",
                [("coupon_pv", "Coupon PV", "#3182ce"),
                 ("principal_pv", "Principal PV", "#e53e3e")],
                title="Present Value of Cash Flows",
                x_label="Maturity (y)",
            )
            st.plotly_chart(fig_pv, use_container_width=True)
        except Exception:
            st.error("Error computing PV chart")

    st.markdown("")
    _render_qa("bonds")


# ═══════════════════════════════════════════════════════════════════════════
# TURBO TAB
# ═══════════════════════════════════════════════════════════════════════════

def turbo_tab():

    # ── SIDEBAR INPUTS ──
    with st.sidebar:
        st.markdown('<p class="app-title">Turbo Parameters</p>', unsafe_allow_html=True)

        turbo_type = st.radio("Type", ["Long", "Short"], horizontal=True, key="turbo_type")
        is_long = turbo_type == "Long"

        t_S = st.number_input("Underlying (S)", value=100.0, step=1.0,
                               format="%.2f", key="t_s")

        input_mode = st.radio("Define by", ["Strike / Barrier", "Target Leverage"],
                               horizontal=True, key="t_mode")

        if input_mode == "Strike / Barrier":
            t_K = st.number_input("Strike / Financing (K)", value=80.0, step=1.0,
                                   format="%.2f", key="t_k")
            default_B = 82.0 if is_long else 118.0
            t_B = st.number_input("Knock-Out Barrier (B)", value=default_B, step=1.0,
                                   format="%.2f", key="t_b")
        else:
            # Target leverage mode: user picks leverage, we compute K & B
            target_lev = st.number_input("Target Leverage", value=5.0, step=0.5,
                                          format="%.1f", min_value=1.1, key="t_lev")
            gap_pct = st.number_input("Barrier gap above K (%)", value=2.0, step=0.5,
                                       format="%.1f", min_value=0.0, key="t_gap")
            # Leverage = S / (S - K) for Long  →  K = S × (1 - 1/L)
            # Leverage = S / (K - S) for Short →  K = S × (1 + 1/L)
            if is_long:
                t_K = round(t_S * (1 - 1 / target_lev), 2)
                t_B = round(t_K * (1 + gap_pct / 100), 2)
            else:
                t_K = round(t_S * (1 + 1 / target_lev), 2)
                t_B = round(t_K * (1 - gap_pct / 100), 2)

            st.caption(f"→ K = {t_K:.2f}  |  B = {t_B:.2f}")

        tc1, tc2 = st.columns(2)
        parity = tc1.number_input("Parity", value=10.0, step=1.0,
                                   format="%.0f", min_value=1.0, key="t_par")
        r_f = tc2.number_input("Financing %", value=2.0, step=0.1,
                                format="%.2f", key="t_rf")

        st.caption("Position")
        t_lots = st.number_input("Lots (# turbos)", value=100.0, step=10.0,
                                  format="%.0f", min_value=1.0, key="t_lots")

    # ── COMPUTE ──
    if is_long:
        intrinsic = max(0.0, t_S - t_K)
    else:
        intrinsic = max(0.0, t_K - t_S)

    turbo_price = intrinsic / parity
    leverage = (t_S / (turbo_price * parity)) if turbo_price > 0.0001 else 0.0
    dist_barrier = abs(t_S - t_B) / t_S * 100 if t_S > 0 else 0.0
    daily_funding = t_K * (r_f / 100) / 360

    initial_delta_cash = t_lots * turbo_price
    leveraged_delta_cash = leverage * initial_delta_cash

    # Is the position alive?
    knocked_out = (is_long and t_S <= t_B) or (not is_long and t_S >= t_B)

    # ── PRIMARY METRICS ──
    _section("Turbo Open-End Pricing")

    if knocked_out:
        st.error("**KNOCK-OUT** — The underlying has breached the barrier. Turbo value = 0.", icon="💥")

    st.markdown('<div class="hero-metric">', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Turbo Price", f"{turbo_price:.4f}" if not knocked_out else "0.0000")
    m2.metric("Leverage", f"{leverage:.1f}x" if not knocked_out else "—")
    m3.metric("Distance to Barrier", f"{dist_barrier:.2f}%")
    m4.metric("Daily Funding Cost", f"{daily_funding:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── POSITION DELTAS ──
    _section("Position Exposure")
    pe1, pe2, pe3 = st.columns(3)
    pe1.metric("Initial Δ Cash (investment)", f"{initial_delta_cash:,.2f}" if not knocked_out else "0.00")
    pe2.metric("Leveraged Δ Cash (exposure)", f"{leveraged_delta_cash:,.2f}" if not knocked_out else "0.00")
    pe3.metric("Equivalent Underlying", f"{leveraged_delta_cash / t_S:,.1f} units" if not knocked_out and t_S > 0 else "—")

    # Barrier proximity warning
    if not knocked_out and dist_barrier < 2.0:
        st.warning("**Knock-out risk imminent** — distance to barrier < 2%. "
                    "A small adverse move will terminate the product.", icon="⚠️")
    elif not knocked_out and dist_barrier < 5.0:
        st.info(f"Barrier proximity: {dist_barrier:.2f}% — monitor closely.", icon="ℹ️")

    st.markdown("")

    # ── STRIKE DRIFT SIMULATION ──
    _section("Financing Cost Simulation — Strike Drift Over Time")

    with st.container(border=True):
        drift_days = st.slider("Holding period (days)", 0, 360, 30, step=1, key="t_drift")

        K_drifted = t_K + daily_funding * drift_days if is_long else t_K - daily_funding * drift_days

        if is_long:
            intr_drifted = max(0.0, t_S - K_drifted)
        else:
            intr_drifted = max(0.0, K_drifted - t_S)

        price_drifted = intr_drifted / parity
        value_erosion = turbo_price - price_drifted

        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("K today", f"{t_K:.2f}")
        dc2.metric(f"K after {drift_days}d", f"{K_drifted:.4f}",
                    delta=f"{K_drifted - t_K:+.4f}")
        dc3.metric("Turbo Price (drifted)", f"{price_drifted:.4f}")
        dc4.metric("Value Erosion", f"{value_erosion:.4f}",
                    delta=f"{-value_erosion:.4f}" if value_erosion > 0 else "0",
                    delta_color="inverse")

        # Drift chart: K and Turbo price over time
        drift_data = []
        for d in range(0, drift_days + 1):
            Kd = t_K + daily_funding * d if is_long else t_K - daily_funding * d
            iv = max(0.0, t_S - Kd) if is_long else max(0.0, Kd - t_S)
            drift_data.append({
                "day": d,
                "strike": round(Kd, 4),
                "turbo_price": round(iv / parity, 4),
            })

        if drift_data:
            dc_l, _sp, dc_r = st.columns([5, 0.3, 5])
            with dc_l:
                fig_k = _make_line_chart(
                    drift_data, "day",
                    [("strike", "Strike (K)", "#e53e3e")],
                    title="Strike Drift Over Time",
                    x_label="Days",
                    legend_below=True, height=300,
                )
                # Add barrier line
                fig_k.add_hline(y=t_B, line_dash="dot", line_color="#a0aec0",
                                annotation_text="Barrier",
                                annotation_font_size=10,
                                annotation_font_color="#a0aec0")
                st.plotly_chart(fig_k, use_container_width=True)
            with dc_r:
                fig_tp = _make_line_chart(
                    drift_data, "day",
                    [("turbo_price", "Turbo Price", "#3182ce")],
                    title="Turbo Price Erosion (Spot unchanged)",
                    x_label="Days",
                    legend_below=True, height=300,
                )
                st.plotly_chart(fig_tp, use_container_width=True)

    st.markdown("")

    # ── PAYOFF CHART ──
    _section("Payoff at Current Time")

    # Build payoff data across a range of spots
    spot_lo = t_K * 0.7 if is_long else t_K * 0.5
    spot_hi = t_K * 1.5 if is_long else t_K * 1.3
    spots = np.linspace(spot_lo, spot_hi, 200)
    payoff_data = []

    for s in spots:
        s = float(s)
        # Check knock-out
        if is_long:
            ko = s <= t_B
            iv = max(0.0, s - t_K) / parity if not ko else 0.0
        else:
            ko = s >= t_B
            iv = max(0.0, t_K - s) / parity if not ko else 0.0

        payoff_data.append({
            "spot": round(s, 2),
            "payoff": round(iv, 4),
        })

    fig_payoff = _make_line_chart(
        payoff_data, "spot",
        [("payoff", f"Turbo {turbo_type}", "#3182ce")],
        title=f"Turbo {turbo_type} Payoff",
        x_label="Underlying Spot",
        height=400,
    )
    # Add barrier line
    fig_payoff.add_vline(x=t_B, line_dash="dash", line_color="#e53e3e", line_width=1.5,
                          annotation_text=f"Barrier ({t_B})",
                          annotation_font_size=10,
                          annotation_font_color="#e53e3e")
    # Add strike line
    fig_payoff.add_vline(x=t_K, line_dash="dot", line_color="#a0aec0", line_width=1,
                          annotation_text=f"Strike ({t_K})",
                          annotation_font_size=10,
                          annotation_font_color="#718096",
                          annotation_position="bottom right")
    # Add current spot marker
    if not knocked_out:
        fig_payoff.add_vline(x=t_S, line_dash="dot", line_color="#38a169", line_width=1,
                              annotation_text=f"Spot ({t_S})",
                              annotation_font_size=10,
                              annotation_font_color="#38a169",
                              annotation_position="top left")

    st.plotly_chart(fig_payoff, use_container_width=True)

    # ── SENSITIVITY TABLE ──
    with st.expander("**Turbo Sensitivity Table — Spot Scenarios**", expanded=False):
        scenarios = np.linspace(t_B * 0.98, spot_hi, 25) if is_long else np.linspace(spot_lo, t_B * 1.02, 25)
        rows = []
        for s in scenarios:
            s = float(s)
            if is_long:
                ko = s <= t_B
                iv = max(0.0, s - t_K) / parity if not ko else 0.0
                lev = (s / (iv * parity)) if iv > 0.0001 else 0.0
            else:
                ko = s >= t_B
                iv = max(0.0, t_K - s) / parity if not ko else 0.0
                lev = (s / (iv * parity)) if iv > 0.0001 else 0.0
            dist = abs(s - t_B) / s * 100 if s > 0 else 0
            pnl = (iv - turbo_price) if not knocked_out else -turbo_price
            rows.append({
                "Spot": f"{s:.2f}",
                "Turbo Price": f"{iv:.4f}" if not ko else "KO",
                "Leverage": f"{lev:.1f}x" if not ko else "—",
                "Dist. Barrier": f"{dist:.1f}%",
                "P&L / unit": f"{pnl:+.4f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# DISCOUNT CERTIFICATES TAB
# ═══════════════════════════════════════════════════════════════════════════

def discount_tab():

    with st.sidebar:
        st.markdown('<p class="app-title">Discount Certificate</p>', unsafe_allow_html=True)

        dc_S = st.number_input("Underlying (S)", value=100.0, step=1.0,
                                format="%.2f", key="dc_s")
        dc_cap = st.number_input("Cap", value=110.0, step=1.0,
                                  format="%.2f", key="dc_cap")

        dc1, dc2 = st.columns(2)
        dc_T = dc1.number_input("Maturity (y)", value=1.0, step=0.25,
                                 format="%.2f", min_value=0.01, key="dc_t")
        dc_sigma = max(0.001, dc2.number_input("Vol %", value=20.0, step=0.5,
                                                 format="%.1f", key="dc_sig") / 100)

        dc3, dc4 = st.columns(2)
        dc_r = dc3.number_input("r %", value=5.0, step=0.1,
                                 format="%.2f", key="dc_r") / 100
        dc_q = dc4.number_input("q %", value=2.0, step=0.1,
                                 format="%.2f", key="dc_q") / 100

        dc_parity = st.number_input("Parity", value=1.0, step=1.0,
                                     format="%.0f", min_value=1.0, key="dc_par")

    # ── Compute ──
    try:
        res = discount_certificate_price(dc_S, dc_cap, dc_T, dc_r, dc_q,
                                          dc_sigma, dc_parity)
    except Exception:
        res = {"dc_price": 0, "call_price": 0, "underlying_pv": 0,
               "discount_pct": 0, "max_payoff": 0, "max_return_pct": 0,
               "breakeven": 0, "sideways_return": 0}

    # ── Replication Breakdown ──
    _section("Replication — Long Underlying + Short Call(K=Cap)")

    st.markdown(
        f"**DC Price** = PV(Underlying) − Call(K=Cap) = "
        f"`{res['underlying_pv']:.4f} − {res['call_price']:.4f}` = "
        f"**`{res['dc_price']:.4f}`**  per certificate"
    )

    st.markdown("")

    # ── Hero Metrics ──
    _section("Key Metrics")
    st.markdown('<div class="hero-metric">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Certificate Price", f"{res['dc_price']:.4f}")
    m2.metric("Discount", f"{res['discount_pct']:.2f}%")
    m3.metric("Max Return", f"{res['max_return_pct']:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    m4, m5, m6 = st.columns(3)
    m4.metric("Max Payoff", f"{res['max_payoff']:.2f}")
    m5.metric("Breakeven", f"{res['breakeven']:.2f}")
    m6.metric("Sideways Return", f"{res['sideways_return']:.2f}%")

    st.markdown("")

    # ── Decomposition Table ──
    with st.expander("**Pricing Decomposition**", expanded=False):
        decomp = pd.DataFrame({
            "Component": [
                "Underlying PV  (S × e^{-qT})",
                "Short Call  BS(S, K=Cap)",
                "= Certificate Price  (per unit)",
                f"Spot / Parity  (reference)",
            ],
            "Value": [
                f"{res['underlying_pv']:.4f}",
                f"−{res['call_price']:.4f}",
                f"**{res['dc_price']:.4f}**",
                f"{dc_S / dc_parity:.4f}",
            ],
        })
        st.dataframe(decomp, hide_index=True, use_container_width=True)

    st.markdown("")

    # ── Payoff Chart ──
    _section("Payoff at Maturity — Stock vs Discount Certificate")

    try:
        payoff = discount_payoff_data(dc_S, dc_cap, res["dc_price"], dc_parity)
    except Exception:
        payoff = []

    if payoff:
        ch1, _sp, ch2 = st.columns([5, 0.3, 5])

        with ch1:
            fig_payoff = _make_line_chart(
                payoff, "spot",
                [("stock_payoff", "Stock Payoff", "#a0aec0"),
                 ("dc_payoff", "Certificate Payoff", "#3182ce")],
                title="Payoff at Maturity",
                x_label="Underlying at Expiry",
            )
            # Cap line
            fig_payoff.add_hline(y=dc_cap, line_dash="dot", line_color="#e53e3e",
                                 line_width=1,
                                 annotation_text=f"Cap ({dc_cap})",
                                 annotation_font_size=10,
                                 annotation_font_color="#e53e3e")
            st.plotly_chart(fig_payoff, use_container_width=True)

        with ch2:
            fig_pnl = _make_line_chart(
                payoff, "spot",
                [("stock_pnl", "Stock P&L", "#a0aec0"),
                 ("dc_pnl", "Certificate P&L", "#3182ce")],
                title="Profit & Loss at Maturity",
                x_label="Underlying at Expiry",
            )
            fig_pnl.add_hline(y=0, line_dash="solid", line_color="#e2e8f0",
                               line_width=1)
            # Breakeven
            fig_pnl.add_vline(x=res["breakeven"], line_dash="dot",
                               line_color="#38a169", line_width=1,
                               annotation_text=f"BE ({res['breakeven']:.1f})",
                               annotation_font_size=10,
                               annotation_font_color="#38a169")
            st.plotly_chart(fig_pnl, use_container_width=True)

    st.markdown("")

    # ── Sensitivity Charts ──
    _section("Sensitivity — Impact of Volatility & Cap Level")

    sc1, _sp2, sc2 = st.columns([5, 0.3, 5])

    with sc1:
        try:
            vol_data = _cached_dc_vols(dc_S, dc_cap, dc_T, dc_r, dc_q,
                                             dc_parity)
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=[d["vol"] for d in vol_data],
                y=[d["dc_price"] for d in vol_data],
                mode="lines", name="DC Price",
                line=dict(color="#3182ce", width=2.2),
                yaxis="y1",
            ))
            fig_vol.add_trace(go.Scatter(
                x=[d["vol"] for d in vol_data],
                y=[d["discount_pct"] for d in vol_data],
                mode="lines", name="Discount %",
                line=dict(color="#e53e3e", width=2.2, dash="dot"),
                yaxis="y2",
            ))
            fig_vol.update_layout(
                title=dict(text="DC Price & Discount vs Volatility",
                           font=dict(size=13, color="#2d3748"), x=0),
                xaxis_title="Volatility (%)",
                yaxis=dict(title="DC Price", gridcolor="#f0f0f0",
                           linecolor="#ddd"),
                yaxis2=dict(title="Discount %", overlaying="y", side="right",
                            gridcolor="#f0f0f0", linecolor="#ddd"),
                height=370,
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, system-ui, sans-serif", size=12,
                          color="#4a5568"),
                margin=dict(l=50, r=50, t=40, b=42),
                legend=dict(orientation="h", yanchor="top", y=-0.2,
                            xanchor="center", x=0.5, font=dict(size=11)),
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        except Exception:
            st.error("Error computing vol sensitivity")

    with sc2:
        try:
            cap_data = _cached_dc_caps(dc_S, dc_T, dc_r, dc_q,
                                             dc_sigma, dc_parity)
            fig_cap = go.Figure()
            fig_cap.add_trace(go.Scatter(
                x=[d["cap"] for d in cap_data],
                y=[d["dc_price"] for d in cap_data],
                mode="lines", name="DC Price",
                line=dict(color="#3182ce", width=2.2),
                yaxis="y1",
            ))
            fig_cap.add_trace(go.Scatter(
                x=[d["cap"] for d in cap_data],
                y=[d["max_return_pct"] for d in cap_data],
                mode="lines", name="Max Return %",
                line=dict(color="#38a169", width=2.2, dash="dot"),
                yaxis="y2",
            ))
            fig_cap.update_layout(
                title=dict(text="DC Price & Max Return vs Cap Level",
                           font=dict(size=13, color="#2d3748"), x=0),
                xaxis_title="Cap Level",
                yaxis=dict(title="DC Price", gridcolor="#f0f0f0",
                           linecolor="#ddd"),
                yaxis2=dict(title="Max Return %", overlaying="y", side="right",
                            gridcolor="#f0f0f0", linecolor="#ddd"),
                height=370,
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, system-ui, sans-serif", size=12,
                          color="#4a5568"),
                margin=dict(l=50, r=50, t=40, b=42),
                legend=dict(orientation="h", yanchor="top", y=-0.2,
                            xanchor="center", x=0.5, font=dict(size=11)),
            )
            # Current cap marker
            fig_cap.add_vline(x=dc_cap, line_dash="dot", line_color="#a0aec0",
                               line_width=1)
            st.plotly_chart(fig_cap, use_container_width=True)
        except Exception:
            st.error("Error computing cap sensitivity")

    st.markdown("")
    _render_qa("discount_cert")


# ═══════════════════════════════════════════════════════════════════════════
# BONUS CERTIFICATE TAB
# ═══════════════════════════════════════════════════════════════════════════

def bonus_tab():

    with st.sidebar:
        st.markdown('<p class="app-title">Bonus Certificate</p>', unsafe_allow_html=True)

        bc_S = st.number_input("Underlying (S)", value=100.0, step=1.0,
                                format="%.2f", key="bc_s")

        st.caption("Levels (absolute or % of Spot)")
        bc_mode = st.radio("Input as", ["Absolute", "% of Spot"],
                            horizontal=True, key="bc_mode")

        if bc_mode == "Absolute":
            bc_bonus = st.number_input("Bonus level", value=120.0, step=1.0,
                                        format="%.2f", key="bc_bon")
            bc_barrier = st.number_input("Barrier", value=70.0, step=1.0,
                                          format="%.2f", key="bc_bar")
            bc_cap_on = st.toggle("Cap", value=True, key="bc_cap_on")
            bc_cap = st.number_input("Cap level", value=140.0, step=1.0,
                                      format="%.2f", key="bc_cap") if bc_cap_on else None
        else:
            bc_bon_pct = st.number_input("Bonus %", value=120.0, step=5.0,
                                          format="%.0f", key="bc_bon_pct")
            bc_bar_pct = st.number_input("Barrier %", value=70.0, step=5.0,
                                          format="%.0f", key="bc_bar_pct")
            bc_bonus = bc_S * bc_bon_pct / 100
            bc_barrier = bc_S * bc_bar_pct / 100
            bc_cap_on = st.toggle("Cap", value=True, key="bc_cap_on2")
            if bc_cap_on:
                bc_cap_pct = st.number_input("Cap %", value=140.0, step=5.0,
                                              format="%.0f", key="bc_cap_pct")
                bc_cap = bc_S * bc_cap_pct / 100
            else:
                bc_cap = None
            st.caption(f"→ Bonus={bc_bonus:.2f}  Barrier={bc_barrier:.2f}"
                       + (f"  Cap={bc_cap:.2f}" if bc_cap else ""))

        bc_T = st.number_input("Mat. (y)", value=1.0, step=0.25,
                                format="%.2f", min_value=0.01, key="bc_t")

        st.caption("Volatility (skew)")
        bv1, bv2 = st.columns(2)
        bc_sigma_put = max(0.001, bv1.number_input("Put vol %", value=22.0, step=0.5,
                                                     format="%.1f", key="bc_sig_p") / 100)
        bc_sigma_call = max(0.001, bv2.number_input("Call vol %", value=18.0, step=0.5,
                                                      format="%.1f", key="bc_sig_c") / 100)

        b3, b4 = st.columns(2)
        bc_r = b3.number_input("r %", value=5.0, step=0.1,
                                format="%.2f", key="bc_r") / 100
        bc_q = b4.number_input("q %", value=2.0, step=0.1,
                                format="%.2f", key="bc_q") / 100

        bc_parity = st.number_input("Parity", value=1.0, step=1.0,
                                     format="%.0f", min_value=1.0, key="bc_par")

    # ── Compute ──
    try:
        res = bonus_certificate_price(bc_S, bc_bonus, bc_barrier, bc_T,
                                       bc_r, bc_q, bc_sigma_put, bc_cap, bc_parity,
                                       sigma_call=bc_sigma_call)
    except Exception:
        res = {"bc_price": 0, "put_do_price": 0, "underlying_pv": 0,
               "call_cap_price": 0, "discount_pct": 0, "bonus_return_pct": 0,
               "max_payoff": 0, "max_return_pct": 0, "breakeven": 0,
               "barrier_distance_pct": 0, "has_cap": False}

    # ── Replication ──
    _section("Replication — Long Underlying + Long Put Down-and-Out(K=Bonus, B=Barrier)"
             + (" − Call(K=Cap)" if res["has_cap"] else ""))

    cap_part = f" − {res['call_cap_price']:.4f}" if res["has_cap"] else ""
    formula = (f"**BC** = PV(S) + Put_DO"
               f"{' − Call(Cap)' if res['has_cap'] else ''}"
               f" = `{res['underlying_pv']:.4f} + {res['put_do_price']:.4f}"
               f"{cap_part}`"
               f" = **`{res['bc_price']:.4f}`**")
    st.markdown(formula)

    st.markdown("")

    # ── Hero Metrics ──
    _section("Key Metrics")
    st.markdown('<div class="hero-metric">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Certificate Price", f"{res['bc_price']:.4f}")
    m2.metric("Barrier Distance", f"{res['barrier_distance_pct']:.2f}%")
    m3.metric("Bonus Return", f"{res['bonus_return_pct']:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    m4, m5, m6 = st.columns(3)
    m4.metric("Breakeven", f"{res['breakeven']:.2f}")
    m5.metric("Put D&O Value", f"{res['put_do_price']:.4f}")
    if res["has_cap"]:
        m6.metric("Max Return (cap)", f"{res['max_return_pct']:.2f}%")
    else:
        m6.metric("Upside", "Unlimited")

    if res["barrier_distance_pct"] < 10:
        st.warning(f"Barrier proximity: {res['barrier_distance_pct']:.1f}% — "
                    "the down-and-out put has limited value.", icon="⚠️")

    st.markdown("")

    # ── Decomposition ──
    with st.expander("**Pricing Decomposition**", expanded=False):
        rows = [
            ("Underlying PV  (S × e^{-qT})", f"{res['underlying_pv']:.4f}"),
            ("+ Put Down-and-Out (K=Bonus, B=Barrier)", f"+{res['put_do_price']:.4f}"),
        ]
        if res["has_cap"]:
            rows.append(("− Call (K=Cap)", f"−{res['call_cap_price']:.4f}"))
        rows.append(("= **Certificate Price**", f"**{res['bc_price']:.4f}**"))
        st.dataframe(pd.DataFrame(rows, columns=["Component", "Value"]),
                      hide_index=True, use_container_width=True)

    with st.expander("**How It Works — Bonus vs Cap**", expanded=False):
        st.markdown("""
**Example:** S = 100, Barrier = 70, Bonus = 120, Cap = 140. If the barrier is **never** breached:

| S_T at Maturity | Payoff | Why |
|---|---|---|
| 60 | 60 | Barrier was hit → no protection, you hold the stock |
| 80 | **120** | S_T < Bonus → the bonus **floor** kicks in |
| 110 | **120** | Still below Bonus → payoff = Bonus = 120 |
| 130 | **130** | Bonus < S_T < Cap → you follow the stock upside |
| 150 | **140** | S_T > Cap → **capped** at 140 |

**Bonus = floor.** As long as the barrier is intact, you receive at least the Bonus level, even if the stock ends below it.

**Cap = ceiling.** Even if the stock rallies to 200, your payoff is limited to the Cap.

**Between Bonus and Cap:** you participate 1:1 in the upside.

The Cap exists because you are short a Call(K=Cap) in the replication — this is what partially funds the
Down-and-Out Put that gives you the Bonus protection. The lower the Cap (closer to Bonus), the cheaper the
certificate, but the more limited your upside.
""")

    st.markdown("")

    # ── Payoff Charts ──
    _section("Payoff at Maturity")

    try:
        payoff = bonus_payoff_data(bc_S, bc_bonus, bc_barrier, res["bc_price"],
                                    bc_cap, bc_parity)
    except Exception:
        payoff = {"below": [], "above": []}

    below = payoff.get("below", [])
    above = payoff.get("above", [])

    if below or above:
        ch1, _sp, ch2 = st.columns([5, 0.3, 5])

        # -- Helper: build the gap-style payoff figure --
        def _build_bc_figure(y_key: str, title: str, y_label: str = ""):
            fig = go.Figure()

            # Stock reference (dashed, full range)
            all_pts = below + above
            fig.add_trace(go.Scatter(
                x=[d["spot"] for d in all_pts],
                y=[d["stock" if y_key == "bc_payoff" else "stock_pnl"] for d in all_pts],
                mode="lines", name="Stock",
                line=dict(color="#bbb", width=1.5, dash="dash"),
            ))

            # Below barrier: breach zone (solid red)
            if below:
                fig.add_trace(go.Scatter(
                    x=[d["spot"] for d in below],
                    y=[d[y_key] for d in below],
                    mode="lines", name="BC (barrier breached)",
                    line=dict(color="#e53e3e", width=2.5),
                ))

            # Vertical gap at barrier (thin dotted red connector)
            if below and above:
                gap_x = bc_barrier
                gap_lo = below[-1][y_key]
                gap_hi = above[0][y_key]
                fig.add_trace(go.Scatter(
                    x=[gap_x, gap_x], y=[gap_lo, gap_hi],
                    mode="lines", name="Gap risk",
                    line=dict(color="#e53e3e", width=1.5, dash="dot"),
                    showlegend=False,
                ))

            # Above barrier: intact zone (solid red)
            if above:
                fig.add_trace(go.Scatter(
                    x=[d["spot"] for d in above],
                    y=[d[y_key] for d in above],
                    mode="lines", name="BC (barrier intact)",
                    line=dict(color="#e53e3e", width=2.5),
                    showlegend=False,
                ))

            # Annotations
            fig.add_vline(x=bc_barrier, line_dash="dash", line_color="#a0aec0",
                           line_width=1,
                           annotation_text="B", annotation_font_size=12,
                           annotation_font_color="#718096",
                           annotation_position="bottom")
            fig.add_vline(x=bc_S, line_dash="dot", line_color="#a0aec0",
                           line_width=1,
                           annotation_text="S₀", annotation_font_size=12,
                           annotation_font_color="#718096",
                           annotation_position="bottom")

            if y_key == "bc_payoff":
                cap_level = bc_cap if bc_cap is not None else bc_bonus
                fig.add_hline(y=cap_level, line_dash="dot", line_color="#a0aec0",
                               line_width=1,
                               annotation_text="N+BA" if bc_cap else "Bonus",
                               annotation_font_size=11,
                               annotation_font_color="#718096")
            else:
                fig.add_hline(y=0, line_dash="solid", line_color="#e2e8f0",
                               line_width=1)
                fig.add_vline(x=res["breakeven"], line_dash="dot",
                               line_color="#38a169", line_width=1,
                               annotation_text=f"BE",
                               annotation_font_size=11,
                               annotation_font_color="#38a169")

            fig.update_layout(
                title=dict(text=title, font=dict(size=13, color="#2d3748"),
                           x=0, xanchor="left"),
                xaxis_title="S_T",
                yaxis_title=y_label,
                height=400,
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, system-ui, sans-serif", size=12,
                          color="#4a5568"),
                margin=dict(l=50, r=15, t=40, b=50),
                legend=dict(orientation="h", yanchor="top", y=-0.15,
                            xanchor="center", x=0.5, font=dict(size=11)),
                xaxis=dict(gridcolor="#f0f0f0", linecolor="#ddd", zeroline=False),
                yaxis=dict(gridcolor="#f0f0f0", linecolor="#ddd", zeroline=False),
            )
            return fig

        with ch1:
            fig_pay = _build_bc_figure("bc_payoff", "Payoff at Maturity", "X_T")
            st.plotly_chart(fig_pay, use_container_width=True)

        with ch2:
            fig_pnl = _build_bc_figure("bc_pnl", "Profit & Loss", "P&L")
            st.plotly_chart(fig_pnl, use_container_width=True)

    st.markdown("")

    # ── Sensitivity Charts ──
    _section("Sensitivity — Volatility & Time to Maturity")

    sc1, _sp2, sc2 = st.columns([5, 0.3, 5])

    with sc1:
        try:
            vol_data = _cached_bc_vols(bc_S, bc_bonus, bc_barrier, bc_T,
                                             bc_r, bc_q, bc_cap, bc_parity,
                                             bc_sigma_call)
            fig_v = go.Figure()
            fig_v.add_trace(go.Scatter(
                x=[d["vol"] for d in vol_data],
                y=[d["bc_price"] for d in vol_data],
                mode="lines", name="BC Price",
                line=dict(color="#3182ce", width=2.2), yaxis="y1",
            ))
            fig_v.add_trace(go.Scatter(
                x=[d["vol"] for d in vol_data],
                y=[d["put_do"] for d in vol_data],
                mode="lines", name="Put D&O",
                line=dict(color="#e53e3e", width=2, dash="dot"), yaxis="y1",
            ))
            fig_v.update_layout(
                title=dict(text="BC Price & Put D&O vs Volatility",
                           font=dict(size=13, color="#2d3748"), x=0),
                xaxis_title="Volatility (%)",
                yaxis=dict(title="Price", gridcolor="#f0f0f0", linecolor="#ddd"),
                height=370, plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, system-ui, sans-serif", size=12, color="#4a5568"),
                margin=dict(l=50, r=15, t=40, b=42),
                legend=dict(orientation="h", yanchor="top", y=-0.2,
                            xanchor="center", x=0.5, font=dict(size=11)),
            )
            # Current vol marker
            fig_v.add_vline(x=bc_sigma * 100, line_dash="dot", line_color="#a0aec0",
                             line_width=1)
            st.plotly_chart(fig_v, use_container_width=True)
        except Exception:
            st.error("Error computing vol sensitivity")

    with sc2:
        try:
            time_data = _cached_bc_time(bc_S, bc_bonus, bc_barrier,
                                              bc_r, bc_q, bc_sigma_put, bc_cap, bc_parity,
                                              bc_sigma_call)
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=[d["time"] for d in time_data],
                y=[d["bc_price"] for d in time_data],
                mode="lines", name="BC Price",
                line=dict(color="#3182ce", width=2.2), yaxis="y1",
            ))
            fig_t.add_trace(go.Scatter(
                x=[d["time"] for d in time_data],
                y=[d["put_do"] for d in time_data],
                mode="lines", name="Put D&O",
                line=dict(color="#e53e3e", width=2, dash="dot"), yaxis="y1",
            ))
            fig_t.update_layout(
                title=dict(text="BC Price & Put D&O vs Time to Maturity",
                           font=dict(size=13, color="#2d3748"), x=0),
                xaxis_title="Time (y)",
                yaxis=dict(title="Price", gridcolor="#f0f0f0", linecolor="#ddd"),
                height=370, plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter, system-ui, sans-serif", size=12, color="#4a5568"),
                margin=dict(l=50, r=15, t=40, b=42),
                legend=dict(orientation="h", yanchor="top", y=-0.2,
                            xanchor="center", x=0.5, font=dict(size=11)),
            )
            fig_t.add_vline(x=bc_T, line_dash="dot", line_color="#a0aec0",
                             line_width=1)
            st.plotly_chart(fig_t, use_container_width=True)
        except Exception:
            st.error("Error computing time sensitivity")

    st.markdown("")
    _render_qa("bonus_cert")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

# Navigation in sidebar — only the active tab renders its inputs
with st.sidebar:
    st.markdown(
        '<span class="app-title">Skema Pricer</span>'
        '<span class="app-badge">Derivatives Pricer</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    active_tab = st.radio(
        "Navigation",
        ["Options", "Bonds", "Turbo", "Discount Cert.", "Bonus Cert."],
        horizontal=True, key="nav", label_visibility="collapsed",
    )
    st.markdown("---")

if active_tab == "Options":
    options_tab()
elif active_tab == "Bonds":
    bonds_tab()
elif active_tab == "Turbo":
    turbo_tab()
elif active_tab == "Discount Cert.":
    discount_tab()
else:
    bonus_tab()
