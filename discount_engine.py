"""
discount_engine.py — Discount Certificate analytics.

A Discount Certificate replicates as:
    Long  1 × Underlying  (zero-strike call ≈ S)
  + Short 1 × European Call (strike = Cap)

The investor buys the underlying at a discount in exchange for
capping upside at the Cap level.

Price_DC = S × e^(-q×T) − Call_BS(S, K=Cap, T, r, q, σ)

Key metrics:
  - Discount (%)     = (S − Price_DC) / S
  - Max Payoff        = Cap / Parity
  - Max Return (%)   = (Cap − Price_DC × Parity) / (Price_DC × Parity)
  - Breakeven         = Price_DC × Parity  (underlying level at which P&L = 0)
  - Sideways Return   = (min(S, Cap) − Price_DC × Parity) / (Price_DC × Parity)
  - Outperformance    = Price_DC < S / Parity  (always true when vol > 0)
"""

import numpy as np
from scipy.stats import norm


# ── Black-Scholes European Call ──────────────────────────────────────────

def _bs_call(S: float, K: float, T: float, r: float, q: float,
             sigma: float) -> float:
    """Black-Scholes price for a European call with continuous dividend q."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ── Discount Certificate Pricing ────────────────────────────────────────

def discount_certificate_price(
    S: float,
    cap: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    parity: float = 1.0,
) -> dict:
    """Price a Discount Certificate and return key analytics.

    Parameters
    ----------
    S : float       Current spot price of the underlying.
    cap : float     Cap level (= strike of the short call).
    T : float       Time to maturity in years.
    r : float       Risk-free rate (decimal, e.g. 0.05).
    q : float       Continuous dividend yield (decimal).
    sigma : float   Implied volatility (decimal, e.g. 0.20).
    parity : float  Number of certificates per 1 unit of underlying.

    Returns
    -------
    dict with keys:
        dc_price        – fair value of one certificate
        call_price      – BS price of the short call (strike = cap)
        underlying_pv   – PV of the underlying (S × e^{-qT})
        discount_pct    – discount vs buying the stock outright (%)
        max_payoff      – maximum payout per certificate at maturity
        max_return_pct  – maximum return (%)
        breakeven       – underlying level at which P&L = 0
        sideways_return – return if underlying stays at S (%)
    """
    call_price = _bs_call(S, cap, T, r, q, sigma)
    underlying_pv = S * np.exp(-q * T)

    # Replication: DC = (underlying_pv − call) / parity
    dc_price = (underlying_pv - call_price) / parity

    # Discount vs spot
    spot_per_cert = S / parity
    discount_pct = (spot_per_cert - dc_price) / spot_per_cert * 100

    # Max payoff = Cap / parity (capped at maturity)
    max_payoff = cap / parity
    max_return_pct = (max_payoff / dc_price - 1) * 100 if dc_price > 0 else 0

    # Breakeven: underlying level where payoff = purchase price
    breakeven = dc_price * parity  # payoff = S_T when S_T < cap

    # Sideways return: what you earn if underlying stays flat
    sideways_payoff = min(S, cap) / parity
    sideways_return = (sideways_payoff / dc_price - 1) * 100 if dc_price > 0 else 0

    return {
        "dc_price": round(dc_price, 6),
        "call_price": round(call_price, 6),
        "underlying_pv": round(underlying_pv, 6),
        "discount_pct": round(discount_pct, 4),
        "max_payoff": round(max_payoff, 6),
        "max_return_pct": round(max_return_pct, 4),
        "breakeven": round(breakeven, 4),
        "sideways_return": round(sideways_return, 4),
    }


# ── Payoff Data (for charts) ────────────────────────────────────────────

def discount_payoff_data(
    S: float,
    cap: float,
    dc_price: float,
    parity: float = 1.0,
    s_min: float | None = None,
    s_max: float | None = None,
    n_points: int = 200,
) -> list[dict]:
    """Payoff at maturity for the DC vs holding the stock.

    Returns list of dicts with keys:
        spot, stock_pnl, dc_pnl, stock_payoff, dc_payoff
    """
    if s_min is None:
        s_min = S * 0.5
    if s_max is None:
        s_max = S * 1.5

    cost_stock = S  # buy 1 share at current price
    cost_dc = dc_price * parity  # cost of parity certificates = 1 share equiv

    spots = np.linspace(s_min, s_max, n_points)
    result = []
    for st in spots:
        st = float(st)
        # Stock: payoff = S_T, P&L = S_T − S
        stock_payoff = st
        stock_pnl = st - cost_stock

        # DC: payoff = min(S_T, Cap), P&L = min(S_T, Cap) − cost
        dc_payoff = min(st, cap)
        dc_pnl = dc_payoff - cost_dc

        result.append({
            "spot": round(st, 2),
            "stock_payoff": round(stock_payoff, 2),
            "dc_payoff": round(dc_payoff, 2),
            "stock_pnl": round(stock_pnl, 2),
            "dc_pnl": round(dc_pnl, 2),
        })
    return result


# ── Price across vol / cap (for sensitivity charts) ─────────────────────

def dc_price_across_vols(
    S: float, cap: float, T: float, r: float, q: float, parity: float,
    vol_min: float = 0.05, vol_max: float = 0.60, n: int = 50,
) -> list[dict]:
    """DC price and discount across a range of vols."""
    result = []
    for v in np.linspace(vol_min, vol_max, n):
        v = float(v)
        res = discount_certificate_price(S, cap, T, r, q, v, parity)
        result.append({
            "vol": round(v * 100, 1),
            "dc_price": round(res["dc_price"], 4),
            "discount_pct": round(res["discount_pct"], 2),
            "max_return_pct": round(res["max_return_pct"], 2),
        })
    return result


def dc_price_across_caps(
    S: float, T: float, r: float, q: float, sigma: float, parity: float,
    cap_min_pct: float = 0.70, cap_max_pct: float = 1.30, n: int = 50,
) -> list[dict]:
    """DC price and discount across a range of cap levels."""
    result = []
    for pct in np.linspace(cap_min_pct, cap_max_pct, n):
        cap = float(S * pct)
        res = discount_certificate_price(S, cap, T, r, q, sigma, parity)
        result.append({
            "cap": round(cap, 2),
            "cap_pct": round(float(pct) * 100, 1),
            "dc_price": round(res["dc_price"], 4),
            "discount_pct": round(res["discount_pct"], 2),
            "max_return_pct": round(res["max_return_pct"], 2),
        })
    return result
