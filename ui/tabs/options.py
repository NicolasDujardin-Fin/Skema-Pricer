"""Options tab — Black-Scholes pricing, Greeks, sensitivity charts."""

import datetime
import math

import pandas as pd
import streamlit as st

from engines.bs import bs_greeks, bs_price
from engines.numerical import american_binomial_tree
from engines.rates import price_forward
from ui.components.shared import section, make_line_chart, render_qa
from ui.components.cache import (
    compute_greeks_across_vols,
    compute_greeks_across_mats,
    raw_greeks_spot_vol,
    raw_greeks_time,
    compute_spot_ladder,
)


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
    section("Pricing")
    st.markdown('<div class="hero-metric">', unsafe_allow_html=True)
    h1, h2, h3, _sp, h4, h5 = st.columns([1, 1, 1, 0.3, 1, 1])
    h1.metric("BS Price", f"{p:.4f}")
    h2.metric("Forward", f"{fwd:.4f}")
    h3.metric("Cash Delta", f"{cd:,.2f}")
    h4.metric("Delta Hedge", f"{n_shares:,.1f} shrs")
    h5.metric("Δ t+1d", f"{delta_tx:,.2f} shrs/day")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECONDARY GREEKS (compact grid) ──
    section("Cash Greeks")
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
    section("Unit Greeks")
    greeks_df = pd.DataFrame({
        "Greek": ["Delta", "Gamma", "Vega /1%", "Theta /day", "Rho /1%"],
        "Value": [f"{g['delta']:.6f}", f"{g['gamma']:.6f}", f"{g['vega']:.6f}",
                  f"{g['theta']:.6f}", f"{rho_unit:.6f}"],
    })
    st.dataframe(greeks_df, hide_index=True, width=360)

    st.markdown("")  # breathing

    # ── CHARTS ──
    section("Charts")

    # Compute spot ladder
    center = K
    half = max(0.30 * center, abs(S - K) * 1.3)
    S_min, S_max = round(center - half, 4), round(center + half, 4)

    try:
        spot_data = compute_spot_ladder(S_min, S_max, K, T, r, q, repo, sigma,
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
            fig1 = make_line_chart(spot_data, "spot", _CHART_MAP[sel_chart1],
                                   title=f"{sel_chart1} vs Spot",
                                   x_label="Spot", vline=K)
            st.plotly_chart(fig1, use_container_width=True)

    with ch2:
        sc2a, sc2b = st.columns(2)
        sel_chart2 = sc2a.selectbox("Chart", list(_CHART2_MAP.keys()), key="ch2")
        sel_axis2 = sc2b.selectbox("vs", ["Vol", "Maturity"], key="ch2ax")

        try:
            if sel_axis2 == "Vol":
                vol_data = compute_greeks_across_vols(
                    S, K, T, r, q, repo, n_lots, multiplier, option_type)
                fig2 = make_line_chart(vol_data, "vol", _CHART2_MAP[sel_chart2],
                                       title=f"{sel_chart2} vs Vol", x_label="Vol (%)")
            else:
                mat_data = compute_greeks_across_mats(
                    S, K, r, q, repo, sigma, n_lots, multiplier, option_type)
                fig2 = make_line_chart(mat_data, "maturity", _CHART2_MAP[sel_chart2],
                                       title=f"{sel_chart2} vs Maturity",
                                       x_label="Maturity (y)")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.error("Error computing chart data")

    # ── RAW GREEKS: Impact of Vol & Time on Delta, Gamma, Vega ──
    st.markdown("")
    section("Unit Greeks Sensitivity — Impact of Volatility & Time to Maturity")

    try:
        raw_sv_data, vols_used = raw_greeks_spot_vol(S, K, T, r, q, repo, option_type, S_min, S_max)
        raw_time_data, moneyness_used = raw_greeks_time(S, K, T, r, q, repo, sigma, option_type)
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
            fig = make_line_chart(raw_sv_data, "spot", lines,
                                  title="Delta vs Spot — by Volatility",
                                  x_label="Spot", vline=K, height=340,
                                  legend_below=True)
            st.plotly_chart(fig, use_container_width=True)
    if raw_time_data:
        with d_c2:
            lines = [(f"delta_{tag}", f"{tag} (S={s:.0f})", _MON_COLORS[tag])
                     for s, tag in moneyness_used]
            fig = make_line_chart(raw_time_data, "time", lines,
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
            fig = make_line_chart(raw_sv_data, "spot", lines,
                                  title="Gamma vs Spot — by Volatility",
                                  x_label="Spot", vline=K, height=340,
                                  legend_below=True)
            st.plotly_chart(fig, use_container_width=True)
    if raw_time_data:
        with g_c2:
            lines = [(f"gamma_{tag}", f"{tag} (S={s:.0f})", _MON_COLORS[tag])
                     for s, tag in moneyness_used]
            fig = make_line_chart(raw_time_data, "time", lines,
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
            fig = make_line_chart(raw_sv_data, "spot", lines,
                                  title="Vega vs Spot — by Volatility",
                                  x_label="Spot", vline=K, height=340,
                                  legend_below=True)
            st.plotly_chart(fig, use_container_width=True)
    if raw_time_data:
        with v_c2:
            lines = [(f"vega_{tag}", f"{tag} (S={s:.0f})", _MON_COLORS[tag])
                     for s, tag in moneyness_used]
            fig = make_line_chart(raw_time_data, "time", lines,
                                  title="Vega vs Time — by Moneyness",
                                  x_label="Time to Maturity (y)", height=340,
                                  legend_below=True)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("")
    render_qa("options")
