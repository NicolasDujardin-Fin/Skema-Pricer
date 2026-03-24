"""
bonus_engine.py — Bonus Certificate (Cap) analytics.

Replication:
    Bonus Certificate = Long Underlying + Long Put Down-and-Out(K=Bonus, B=Barrier)

    - If the barrier is NEVER breached during the life:
        Payoff = max(S_T, Bonus)   (floored at Bonus, capped at Cap if any)
    - If the barrier IS breached at any point:
        Payoff = S_T               (replicates the stock)

The Down-and-Out Put is priced with the Reiner-Rubinstein closed-form
model for continuous barrier monitoring.

References:
    Reiner, E. & Rubinstein, M. (1991) — "Breaking Down Barriers"
    Risk Magazine, 4(8), pp. 28-35.
"""

import numpy as np
from scipy.stats import norm


# ═══════════════════════════════════════════════════════════════════════════
# Reiner-Rubinstein: Down-and-Out Put
# ═══════════════════════════════════════════════════════════════════════════

def _down_and_out_put(
    S: float,
    K: float,
    B: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """Price a European Down-and-Out Put (continuous barrier).

    Parameters
    ----------
    S : float   Spot price.
    K : float   Strike (= Bonus level for the certificate).
    B : float   Barrier level (B < S, B < K for a meaningful product).
    T : float   Time to maturity in years.
    r : float   Risk-free rate (decimal).
    q : float   Continuous dividend yield (decimal).
    sigma : float  Volatility (decimal).

    Returns
    -------
    float — price of the down-and-out put.

    Notes
    -----
    If S <= B the option is already knocked out → value = 0.
    Uses the Reiner-Rubinstein decomposition:
        P_do = P_vanilla − P_di
    where P_di (down-and-in put) is computed analytically.
    """
    if S <= B or T <= 0 or sigma <= 0:
        return 0.0

    # Vanilla European put (BS)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_vanilla = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    # Down-and-in put (Reiner-Rubinstein)
    lam = (r - q + 0.5 * sigma**2) / (sigma**2)
    y = np.log(B**2 / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
    x1 = np.log(S / B) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
    y1 = np.log(B / S) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)

    # Down-and-in put for B < K (which is our case: barrier < bonus)
    if B < K:
        put_di = (
            -S * np.exp(-q * T) * norm.cdf(-x1)
            + K * np.exp(-r * T) * norm.cdf(-x1 + sigma * np.sqrt(T))
            + S * np.exp(-q * T) * (B / S) ** (2 * lam) * norm.cdf(y)
            - K * np.exp(-r * T) * (B / S) ** (2 * lam - 2) * norm.cdf(y - sigma * np.sqrt(T))
        )
    else:
        # B >= K: simpler formula
        put_di = (
            S * np.exp(-q * T) * (B / S) ** (2 * lam) * norm.cdf(y1)
            - K * np.exp(-r * T) * (B / S) ** (2 * lam - 2) * norm.cdf(y1 - sigma * np.sqrt(T))
        )

    put_di = max(0.0, put_di)

    # Down-and-out = Vanilla − Down-and-in
    put_do = max(0.0, put_vanilla - put_di)
    return put_do


# ═══════════════════════════════════════════════════════════════════════════
# Bonus Certificate Pricing
# ═══════════════════════════════════════════════════════════════════════════

def bonus_certificate_price(
    S: float,
    bonus: float,
    barrier: float,
    T: float,
    r: float,
    q: float,
    sigma_put: float,
    cap: float = None,
    parity: float = 1.0,
    sigma_call: float = None,
) -> dict:
    """Price a Bonus Certificate (with optional Cap).

    Replication:
        BC = S × e^{-qT} + Put_DO(K=Bonus, B=Barrier) [− Call(K=Cap) if capped]

    Parameters
    ----------
    S : float        Current spot price.
    bonus : float    Bonus level (guaranteed minimum payoff if barrier not hit).
    barrier : float  Knock-out barrier (lower barrier, B < S).
    T : float        Time to maturity (years).
    r : float        Risk-free rate (decimal).
    q : float        Dividend yield (decimal).
    sigma_put : float  Volatility for the Put D&O (at strike = Bonus).
    cap : float      Optional cap level (None = no cap).
    parity : float   Certificates per unit of underlying.
    sigma_call : float  Volatility for the Call (at strike = Cap).
                        If None, defaults to sigma_put.

    Returns
    -------
    dict with keys:
        bc_price, put_do_price, underlying_pv, call_cap_price,
        discount_pct, bonus_return_pct, max_payoff, breakeven,
        barrier_distance_pct
    """
    if sigma_call is None:
        sigma_call = sigma_put

    underlying_pv = S * np.exp(-q * T)
    put_do = _down_and_out_put(S, bonus, barrier, T, r, q, sigma_put)

    # Optional short call for the cap
    call_cap = 0.0
    if cap is not None and cap > 0:
        from discount_engine import _bs_call
        call_cap = _bs_call(S, cap, T, r, q, sigma_call)

    bc_total = underlying_pv + put_do - call_cap
    bc_price = bc_total / parity

    spot_per_cert = S / parity
    discount_pct = (spot_per_cert - bc_price) / spot_per_cert * 100

    # Bonus return: if barrier never hit, minimum payoff = bonus/parity
    bonus_payoff = bonus / parity
    if cap is not None:
        bonus_payoff = min(bonus_payoff, cap / parity)
    bonus_return_pct = (bonus_payoff / bc_price - 1) * 100 if bc_price > 0 else 0

    # Max payoff
    if cap is not None:
        max_payoff = cap / parity
    else:
        max_payoff = bonus / parity  # minimum; actual upside unlimited

    max_return_pct = (max_payoff / bc_price - 1) * 100 if bc_price > 0 else 0

    # Breakeven (barrier breached scenario → payoff = S_T)
    breakeven = bc_price * parity

    barrier_dist = (S - barrier) / S * 100 if S > 0 else 0

    return {
        "bc_price": round(bc_price, 6),
        "put_do_price": round(put_do, 6),
        "underlying_pv": round(underlying_pv, 6),
        "call_cap_price": round(call_cap, 6),
        "discount_pct": round(discount_pct, 4),
        "bonus_return_pct": round(bonus_return_pct, 4),
        "max_payoff": round(max_payoff, 6),
        "max_return_pct": round(max_return_pct, 4),
        "breakeven": round(breakeven, 4),
        "barrier_distance_pct": round(barrier_dist, 4),
        "has_cap": cap is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Payoff Data
# ═══════════════════════════════════════════════════════════════════════════

def bonus_payoff_data(
    S: float,
    bonus: float,
    barrier: float,
    bc_price: float,
    cap: float = None,
    parity: float = 1.0,
    n_points: int = 200,
) -> dict:
    """Payoff at maturity for Bonus Certificate vs Stock.

    Returns a dict with two lists:
      "below" — S_T from 0 to B (barrier breached, payoff = S_T)
      "above" — S_T from B to max (barrier intact, payoff = min(max(S_T, Bonus), Cap))
    Each list contains dicts with keys: spot, stock, bc_payoff, bc_pnl, stock_pnl

    The gap between below[-1] and above[0] at S_T = B is the visual risk.
    """
    s_min = 0.0
    s_max = S * 1.5
    if cap is not None:
        s_max = max(s_max, cap * 1.3)

    cost = bc_price * parity

    # ── Below barrier: breach scenario, payoff = S_T ──
    spots_lo = np.linspace(s_min, barrier, n_points // 2)
    below = []
    for st in spots_lo:
        st = float(st)
        below.append({
            "spot": round(st, 2),
            "stock": round(st, 2),
            "bc_payoff": round(st, 2),
            "bc_pnl": round(st / parity - bc_price, 4),
            "stock_pnl": round(st - S, 2),
        })

    # ── Above barrier: intact scenario, payoff = min(max(S_T, Bonus), Cap) ──
    spots_hi = np.linspace(barrier, s_max, n_points // 2)
    above = []
    for st in spots_hi:
        st = float(st)
        payoff = max(st, bonus)
        if cap is not None:
            payoff = min(payoff, cap)
        above.append({
            "spot": round(st, 2),
            "stock": round(st, 2),
            "bc_payoff": round(payoff, 2),
            "bc_pnl": round(payoff / parity - bc_price, 4),
            "stock_pnl": round(st - S, 2),
        })

    return {"below": below, "above": above}


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def bc_price_across_vols(
    S: float, bonus: float, barrier: float, T: float,
    r: float, q: float, cap: float, parity: float,
    sigma_call: float = None,
    vol_min: float = 0.05, vol_max: float = 0.60, n: int = 50,
) -> list[dict]:
    """Sweep put vol; call vol stays fixed at sigma_call (or tracks put vol if None)."""
    result = []
    for v in np.linspace(vol_min, vol_max, n):
        v = float(v)
        sc = sigma_call if sigma_call is not None else v
        res = bonus_certificate_price(S, bonus, barrier, T, r, q, v, cap, parity,
                                       sigma_call=sc)
        result.append({
            "vol": round(v * 100, 1),
            "bc_price": round(res["bc_price"], 4),
            "put_do": round(res["put_do_price"], 4),
            "bonus_return_pct": round(res["bonus_return_pct"], 2),
        })
    return result


def bc_price_across_time(
    S: float, bonus: float, barrier: float,
    r: float, q: float, sigma_put: float, cap: float, parity: float,
    sigma_call: float = None,
    t_min: float = 0.05, t_max: float = 3.0, n: int = 50,
) -> list[dict]:
    result = []
    sc = sigma_call if sigma_call is not None else sigma_put
    for t in np.linspace(t_min, t_max, n):
        t = max(0.01, float(t))
        res = bonus_certificate_price(S, bonus, barrier, t, r, q, sigma_put, cap, parity,
                                       sigma_call=sc)
        result.append({
            "time": round(t, 2),
            "bc_price": round(res["bc_price"], 4),
            "put_do": round(res["put_do_price"], 4),
            "bonus_return_pct": round(res["bonus_return_pct"], 2),
        })
    return result
