"""Bonus Certificate tab — Long S + Put D&O − Call(K=Cap)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engines.bonus import bonus_certificate_price, bonus_payoff_data
from ui.components.shared import section, render_qa
from ui.components.cache import cached_bc_vols, cached_bc_time


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
    section("Replication — Long Underlying + Long Put Down-and-Out(K=Bonus, B=Barrier)"
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
    section("Key Metrics")
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
    section("Payoff at Maturity")

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
    section("Sensitivity — Volatility & Time to Maturity")

    sc1, _sp2, sc2 = st.columns([5, 0.3, 5])

    with sc1:
        try:
            vol_data = cached_bc_vols(bc_S, bc_bonus, bc_barrier, bc_T,
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
            fig_v.add_vline(x=bc_sigma_put * 100, line_dash="dot", line_color="#a0aec0",
                             line_width=1)
            st.plotly_chart(fig_v, use_container_width=True)
        except Exception:
            st.error("Error computing vol sensitivity")

    with sc2:
        try:
            time_data = cached_bc_time(bc_S, bc_bonus, bc_barrier,
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
    render_qa("bonus_cert")
