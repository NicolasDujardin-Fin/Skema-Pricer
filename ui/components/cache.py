"""
Cached compute wrappers — all @st.cache_data functions live here.
"""

import numpy as np
import streamlit as st

from engines.bs import bs_greeks, bs_price, cash_delta_spot_ladder
from engines.bond import bond_price_yield_curve, callable_bond_yield_curve
from engines.discount import dc_price_across_vols, dc_price_across_caps
from engines.bonus import bc_price_across_vols, bc_price_across_time


# ── Options ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_greeks_across_vols(S, K, T, r, q, repo, n_lots, multiplier, option_type):
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
def compute_greeks_across_mats(S, K, r, q, repo, sigma, n_lots, multiplier, option_type):
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
def raw_greeks_spot_vol(S, K, T, r, q, repo, option_type, S_min, S_max):
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
def raw_greeks_time(S, K, T_max, r, q, repo, sigma, option_type):
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
def compute_spot_ladder(S_min, S_max, K, T, r, q, repo, sigma, n_lots, multiplier, option_type):
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


# ── Bonds ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_callable_yield_curve(face, coupon, mat, freq, call_price, first_call, rate_vol):
    return callable_bond_yield_curve(face, coupon, mat, freq, call_price, first_call, rate_vol)


@st.cache_data(show_spinner=False)
def cached_bond_yield_curve(face, coupon, mat, freq):
    return bond_price_yield_curve(face, coupon, mat, freq)


# ── Discount Certificate ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_dc_vols(S, cap, T, r, q, parity):
    return dc_price_across_vols(S, cap, T, r, q, parity)


@st.cache_data(show_spinner=False)
def cached_dc_caps(S, T, r, q, sigma, parity):
    return dc_price_across_caps(S, T, r, q, sigma, parity)


# ── Bonus Certificate ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cached_bc_vols(S, bonus, barrier, T, r, q, cap, parity, sigma_call):
    return bc_price_across_vols(S, bonus, barrier, T, r, q, cap, parity, sigma_call=sigma_call)


@st.cache_data(show_spinner=False)
def cached_bc_time(S, bonus, barrier, r, q, sigma_put, cap, parity, sigma_call):
    return bc_price_across_time(S, bonus, barrier, r, q, sigma_put, cap, parity, sigma_call=sigma_call)
