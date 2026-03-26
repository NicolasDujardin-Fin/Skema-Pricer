"""Turbo tab — Turbo Open-End Long/Short certificates."""

import numpy as np
import pandas as pd
import streamlit as st

from ui.components.shared import section, make_line_chart, render_qa


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
            target_lev = st.number_input("Target Leverage", value=5.0, step=0.5,
                                          format="%.1f", min_value=1.1, key="t_lev")
            gap_pct = st.number_input("Barrier gap above K (%)", value=2.0, step=0.5,
                                       format="%.1f", min_value=0.0, key="t_gap")
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
    section("Turbo Open-End Pricing")

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
    section("Position Exposure")
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
    section("Financing Cost Simulation — Strike Drift Over Time")

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
                fig_k = make_line_chart(
                    drift_data, "day",
                    [("strike", "Strike (K)", "#e53e3e")],
                    title="Strike Drift Over Time",
                    x_label="Days",
                    legend_below=True, height=300,
                )
                fig_k.add_hline(y=t_B, line_dash="dot", line_color="#a0aec0",
                                annotation_text="Barrier",
                                annotation_font_size=10,
                                annotation_font_color="#a0aec0")
                st.plotly_chart(fig_k, use_container_width=True)
            with dc_r:
                fig_tp = make_line_chart(
                    drift_data, "day",
                    [("turbo_price", "Turbo Price", "#3182ce")],
                    title="Turbo Price Erosion (Spot unchanged)",
                    x_label="Days",
                    legend_below=True, height=300,
                )
                st.plotly_chart(fig_tp, use_container_width=True)

    st.markdown("")

    # ── PAYOFF CHART ──
    section("Payoff at Current Time")

    spot_lo = t_K * 0.7 if is_long else t_K * 0.5
    spot_hi = t_K * 1.5 if is_long else t_K * 1.3
    spots = np.linspace(spot_lo, spot_hi, 200)
    payoff_data = []

    for s in spots:
        s = float(s)
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

    fig_payoff = make_line_chart(
        payoff_data, "spot",
        [("payoff", f"Turbo {turbo_type}", "#3182ce")],
        title=f"Turbo {turbo_type} Payoff",
        x_label="Underlying Spot",
        height=400,
    )
    fig_payoff.add_vline(x=t_B, line_dash="dash", line_color="#e53e3e", line_width=1.5,
                          annotation_text=f"Barrier ({t_B})",
                          annotation_font_size=10,
                          annotation_font_color="#e53e3e")
    fig_payoff.add_vline(x=t_K, line_dash="dot", line_color="#a0aec0", line_width=1,
                          annotation_text=f"Strike ({t_K})",
                          annotation_font_size=10,
                          annotation_font_color="#718096",
                          annotation_position="bottom right")
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

    st.markdown("")
    render_qa("turbo")
