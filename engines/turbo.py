"""
Turbo Open-End certificate pricing — pure computation, no UI.
"""


def turbo_price(S, K, parity, is_long=True):
    """Compute turbo price from spot, strike, and parity.

    Returns dict with price, intrinsic, leverage, and knock-out flag.
    """
    intrinsic = max(0.0, S - K) if is_long else max(0.0, K - S)
    price = intrinsic / parity
    leverage = (S / (price * parity)) if price > 0.0001 else 0.0
    return {
        "price": price,
        "intrinsic": intrinsic,
        "leverage": leverage,
    }


def is_knocked_out(S, B, is_long=True):
    """Check if barrier has been breached."""
    return (is_long and S <= B) or (not is_long and S >= B)


def barrier_distance(S, B):
    """Distance to barrier as a percentage of spot."""
    return abs(S - B) / S * 100 if S > 0 else 0.0


def daily_funding_cost(K, r_pct):
    """Daily financing cost on strike K at rate r (in %)."""
    return K * (r_pct / 100) / 360


def strike_after_drift(K, daily_cost, days, is_long=True):
    """Strike after N days of financing drift."""
    return K + daily_cost * days if is_long else K - daily_cost * days


def drift_series(S, K, B, parity, r_pct, days, is_long=True):
    """Generate strike and turbo price over a holding period.

    Returns list of dicts with day, strike, turbo_price.
    """
    dc = daily_funding_cost(K, r_pct)
    data = []
    for d in range(days + 1):
        Kd = strike_after_drift(K, dc, d, is_long)
        iv = max(0.0, S - Kd) if is_long else max(0.0, Kd - S)
        data.append({
            "day": d,
            "strike": round(Kd, 4),
            "turbo_price": round(iv / parity, 4),
        })
    return data


def payoff_series(K, B, parity, is_long=True, spot_min=None, spot_max=None, n=200):
    """Generate payoff data across a range of spot prices.

    Returns list of dicts with spot and payoff.
    """
    import numpy as np

    if spot_min is None:
        spot_min = K * 0.7 if is_long else K * 0.5
    if spot_max is None:
        spot_max = K * 1.5 if is_long else K * 1.3

    data = []
    for s in np.linspace(spot_min, spot_max, n):
        s = float(s)
        ko = (is_long and s <= B) or (not is_long and s >= B)
        iv = 0.0
        if not ko:
            iv = max(0.0, s - K) / parity if is_long else max(0.0, K - s) / parity
        data.append({"spot": round(s, 2), "payoff": round(iv, 4)})
    return data


def sensitivity_table(S, K, B, parity, current_price, is_long=True,
                      spot_min=None, spot_max=None, n=25):
    """Generate sensitivity table across spot scenarios.

    Returns list of dicts with Spot, Turbo Price, Leverage, Dist. Barrier, P&L.
    """
    import numpy as np

    if spot_min is None:
        spot_min = B * 0.98 if is_long else K * 0.5
    if spot_max is None:
        spot_max = K * 1.5 if is_long else B * 1.02

    rows = []
    for s in np.linspace(spot_min, spot_max, n):
        s = float(s)
        ko = (is_long and s <= B) or (not is_long and s >= B)
        if not ko:
            iv = max(0.0, s - K) / parity if is_long else max(0.0, K - s) / parity
            lev = (s / (iv * parity)) if iv > 0.0001 else 0.0
        else:
            iv = 0.0
            lev = 0.0
        dist = abs(s - B) / s * 100 if s > 0 else 0
        pnl = iv - current_price
        rows.append({
            "Spot": f"{s:.2f}",
            "Turbo Price": f"{iv:.4f}" if not ko else "KO",
            "Leverage": f"{lev:.1f}x" if not ko else "—",
            "Dist. Barrier": f"{dist:.1f}%",
            "P&L / unit": f"{pnl:+.4f}",
        })
    return rows
