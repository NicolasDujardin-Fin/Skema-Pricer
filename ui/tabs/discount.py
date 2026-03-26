"""Discount Certificate tab — Long S − Call(K=Cap)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engines.discount import discount_certificate_price, discount_payoff_data
from ui.components.shared import section, make_line_chart, render_qa
from ui.components.cache import cached_dc_vols, cached_dc_caps


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
    section("Replication — Long Underlying + Short Call(K=Cap)")

    st.markdown(
        f"**DC Price** = PV(Underlying) − Call(K=Cap) = "
        f"`{res['underlying_pv']:.4f} − {res['call_price']:.4f}` = "
        f"**`{res['dc_price']:.4f}`**  per certificate"
    )

    st.markdown("")

    # ── Hero Metrics ──
    section("Key Metrics")
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
    section("Payoff at Maturity — Stock vs Discount Certificate")

    try:
        payoff = discount_payoff_data(dc_S, dc_cap, res["dc_price"], dc_parity)
    except Exception:
        payoff = []

    if payoff:
        ch1, _sp, ch2 = st.columns([5, 0.3, 5])

        with ch1:
            fig_payoff = make_line_chart(
                payoff, "spot",
                [("stock_payoff", "Stock Payoff", "#a0aec0"),
                 ("dc_payoff", "Certificate Payoff", "#3182ce")],
                title="Payoff at Maturity",
                x_label="Underlying at Expiry",
            )
            fig_payoff.add_hline(y=dc_cap, line_dash="dot", line_color="#e53e3e",
                                 line_width=1,
                                 annotation_text=f"Cap ({dc_cap})",
                                 annotation_font_size=10,
                                 annotation_font_color="#e53e3e")
            st.plotly_chart(fig_payoff, use_container_width=True)

        with ch2:
            fig_pnl = make_line_chart(
                payoff, "spot",
                [("stock_pnl", "Stock P&L", "#a0aec0"),
                 ("dc_pnl", "Certificate P&L", "#3182ce")],
                title="Profit & Loss at Maturity",
                x_label="Underlying at Expiry",
            )
            fig_pnl.add_hline(y=0, line_dash="solid", line_color="#e2e8f0",
                               line_width=1)
            fig_pnl.add_vline(x=res["breakeven"], line_dash="dot",
                               line_color="#38a169", line_width=1,
                               annotation_text=f"BE ({res['breakeven']:.1f})",
                               annotation_font_size=10,
                               annotation_font_color="#38a169")
            st.plotly_chart(fig_pnl, use_container_width=True)

    st.markdown("")

    # ── Sensitivity Charts ──
    section("Sensitivity — Impact of Volatility & Cap Level")

    sc1, _sp2, sc2 = st.columns([5, 0.3, 5])

    with sc1:
        try:
            vol_data = cached_dc_vols(dc_S, dc_cap, dc_T, dc_r, dc_q, dc_parity)
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
            cap_data = cached_dc_caps(dc_S, dc_T, dc_r, dc_q, dc_sigma, dc_parity)
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
            fig_cap.add_vline(x=dc_cap, line_dash="dot", line_color="#a0aec0",
                               line_width=1)
            st.plotly_chart(fig_cap, use_container_width=True)
        except Exception:
            st.error("Error computing cap sensitivity")

    st.markdown("")
    render_qa("discount_cert")
