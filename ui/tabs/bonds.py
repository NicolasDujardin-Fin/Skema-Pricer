"""Bonds tab — fixed-income pricing, duration, callable bonds."""

import pandas as pd
import streamlit as st

from engines.bond import (
    bond_price_from_ytm,
    callable_bond_tree,
    callable_bond_tree_detail,
    yield_to_call,
)
from ui.components.shared import section, make_line_chart, make_bar_chart, render_qa
from ui.components.cache import cached_callable_yield_curve, cached_bond_yield_curve


def _build_tree_svg(nodes: list, n: int) -> str:
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
    section("Bond Pricing")
    st.markdown('<div class="hero-metric">', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("Dirty Price", f"{res['dirty_price']:.2f}")
    p2.metric("Clean Price", f"{res['clean_price']:.2f}")
    p3.metric("Accrued Interest", f"{res['accrued_interest']:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    section("Duration & Convexity")
    d1, d2, d3 = st.columns(3)
    d1.metric("Macaulay Duration", f"{res['macaulay_duration']:.4f} y")
    d2.metric("Modified Duration", f"{res['modified_duration']:.4f} y")
    d3.metric("Convexity", f"{res['convexity']:.4f}")

    section("Risk Sensitivity")
    r1, r2, r3 = st.columns(3)
    r1.metric("DV01 (per bond)", f"{dv01:.4f}")
    r2.metric("PV01 (notional)", f"{pv01:,.2f}")
    r3.metric(f"PnL ({bond_bp_shift:+.0f} bp)", f"{pnl:,.2f}")

    st.markdown("")

    # ── Callable ──
    if bond_callable:
        section("Callable Bond Analysis")
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
    section("Charts")
    ch1, _sp, ch2 = st.columns([5, 0.3, 5])

    with ch1:
        try:
            if bond_callable:
                yc_data = cached_callable_yield_curve(
                    bond_face, bond_coupon, effective_mat, bond_freq,
                    bond_call_price, min(bond_first_call, effective_mat), bond_rate_vol,
                )
                lines = [("straight", "Straight", "#3182ce"),
                         ("callable", "Callable", "#e53e3e")]
            else:
                raw = cached_bond_yield_curve(bond_face, bond_coupon, effective_mat, bond_freq)
                yc_data = [{"ytm": d["ytm"], "straight": d["price"]} for d in raw]
                lines = [("straight", "Straight", "#3182ce")]

            fig_yc = make_line_chart(yc_data, "ytm", lines,
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

            fig_pv = make_bar_chart(
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
    render_qa("bonds")
