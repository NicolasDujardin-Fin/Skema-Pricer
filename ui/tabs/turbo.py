"""Turbo tab — Turbo Open-End Long/Short certificates."""

import pandas as pd
import streamlit as st

from engines.turbo import (
    turbo_price as compute_turbo,
    is_knocked_out,
    barrier_distance,
    daily_funding_cost,
    strike_after_drift,
    drift_series,
    payoff_series,
    sensitivity_table,
)
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
    res = compute_turbo(t_S, t_K, parity, is_long)
    tp = res["price"]
    leverage = res["leverage"]
    dist = barrier_distance(t_S, t_B)
    daily_fund = daily_funding_cost(t_K, r_f)
    knocked_out = is_knocked_out(t_S, t_B, is_long)

    initial_delta_cash = t_lots * tp
    leveraged_delta_cash = leverage * initial_delta_cash

    # ── PRIMARY METRICS ──
    section("Turbo Open-End Pricing")

    if knocked_out:
        st.error("**KNOCK-OUT** — The underlying has breached the barrier. Turbo value = 0.", icon="💥")

    st.markdown('<div class="hero-metric">', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Turbo Price", f"{tp:.4f}" if not knocked_out else "0.0000")
    m2.metric("Leverage", f"{leverage:.1f}x" if not knocked_out else "—")
    m3.metric("Distance to Barrier", f"{dist:.2f}%")
    m4.metric("Daily Funding Cost", f"{daily_fund:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── POSITION DELTAS ──
    section("Position Exposure")
    pe1, pe2, pe3 = st.columns(3)
    pe1.metric("Initial Δ Cash (investment)", f"{initial_delta_cash:,.2f}" if not knocked_out else "0.00")
    pe2.metric("Leveraged Δ Cash (exposure)", f"{leveraged_delta_cash:,.2f}" if not knocked_out else "0.00")
    pe3.metric("Equivalent Underlying", f"{leveraged_delta_cash / t_S:,.1f} units" if not knocked_out and t_S > 0 else "—")

    # Barrier proximity warning
    if not knocked_out and dist < 2.0:
        st.warning("**Knock-out risk imminent** — distance to barrier < 2%. "
                    "A small adverse move will terminate the product.", icon="⚠️")
    elif not knocked_out and dist < 5.0:
        st.info(f"Barrier proximity: {dist:.2f}% — monitor closely.", icon="ℹ️")

    st.markdown("")

    # ── STRIKE DRIFT SIMULATION ──
    section("Financing Cost Simulation — Strike Drift Over Time")

    with st.container(border=True):
        drift_days = st.slider("Holding period (days)", 0, 360, 30, step=1, key="t_drift")

        K_drifted = strike_after_drift(t_K, daily_fund, drift_days, is_long)
        res_drifted = compute_turbo(t_S, K_drifted, parity, is_long)
        price_drifted = res_drifted["price"]
        value_erosion = tp - price_drifted

        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("K today", f"{t_K:.2f}")
        dc2.metric(f"K after {drift_days}d", f"{K_drifted:.4f}",
                    delta=f"{K_drifted - t_K:+.4f}")
        dc3.metric("Turbo Price (drifted)", f"{price_drifted:.4f}")
        dc4.metric("Value Erosion", f"{value_erosion:.4f}",
                    delta=f"{-value_erosion:.4f}" if value_erosion > 0 else "0",
                    delta_color="inverse")

        # Drift chart
        drift_data = drift_series(t_S, t_K, t_B, parity, r_f, drift_days, is_long)

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

    payoff_data = payoff_series(t_K, t_B, parity, is_long)

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
        rows = sensitivity_table(t_S, t_K, t_B, parity, tp, is_long)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("")
    render_qa("turbo")
