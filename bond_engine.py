"""
bond_engine.py — Fixed-income bond analytics.

Prices a plain-vanilla fixed-rate bond and computes:
  - Dirty price (sum of discounted cash flows)
  - Macaulay duration
  - Modified duration
  - Convexity
  - Cash-flow schedule with present values (for waterfall chart)
  - Price-yield curve (for relationship chart)

All yields and coupons are expressed as annual percentages (e.g. 5 means 5%).
Discounting uses compound interest at the given frequency.
"""

import numpy as np


def bond_cashflows(
    face: float,
    coupon_rate: float,
    maturity: float,
    freq: int = 2,
) -> list[dict]:
    """Generate the cash-flow schedule for a fixed-rate bond.

    Parameters
    ----------
    face : float
        Face (par) value of the bond.
    coupon_rate : float
        Annual coupon rate in percent (e.g. 5 for 5%).
    maturity : float
        Time to maturity in years.
    freq : int
        Coupon frequency per year (1=annual, 2=semi-annual, 4=quarterly).

    Returns
    -------
    list[dict] with keys:
        - "t"    : float — time in years of the cash flow
        - "cf"   : float — cash flow amount
        - "type" : str   — "coupon" or "coupon+principal"
    """
    coupon = face * (coupon_rate / 100) / freq
    dt = 1.0 / freq

    # Build coupon dates backwards from maturity to get the stub right
    dates = []
    t = maturity
    while t > 1e-9:
        dates.append(round(t, 6))
        t -= dt
    dates.sort()

    flows = []
    for i, t in enumerate(dates):
        if i < len(dates) - 1:
            flows.append({"t": t, "cf": coupon, "type": "coupon"})
        else:
            flows.append({"t": t, "cf": coupon + face, "type": "coupon+principal"})
    return flows


def bond_price_from_ytm(
    face: float,
    coupon_rate: float,
    maturity: float,
    ytm: float,
    freq: int = 2,
) -> dict:
    """Price a bond given its yield to maturity and compute duration & convexity.

    Parameters
    ----------
    face : float
        Face value.
    coupon_rate : float
        Annual coupon rate in percent.
    maturity : float
        Years to maturity.
    ytm : float
        Yield to maturity in percent (e.g. 5 for 5%).
    freq : int
        Coupon frequency per year.

    Returns
    -------
    dict with keys:
        - "price"             : float — dirty price
        - "macaulay_duration" : float — in years
        - "modified_duration" : float — in years
        - "convexity"         : float — in years²
        - "cashflows"         : list[dict] — each with t, cf, pv, type
    """
    flows = bond_cashflows(face, coupon_rate, maturity, freq)
    y_per = ytm / 100 / freq  # yield per period

    price = 0.0
    weighted_t = 0.0       # for Macaulay duration
    convexity_sum = 0.0    # for convexity

    enriched = []
    for flow in flows:
        t = flow["t"]
        cf = flow["cf"]
        n_per = t * freq  # number of periods to this cash flow
        disc = (1 + y_per) ** n_per
        pv = cf / disc

        price += pv
        weighted_t += t * pv
        convexity_sum += t * (t + 1.0 / freq) * pv

        enriched.append({
            "t": round(t, 4),
            "cf": round(cf, 4),
            "pv": round(pv, 4),
            "type": flow["type"],
        })

    mac_dur = weighted_t / price if price > 0 else 0.0
    mod_dur = mac_dur / (1 + y_per) if (1 + y_per) != 0 else 0.0
    convexity = convexity_sum / (price * (1 + y_per) ** 2) if price > 0 else 0.0

    # Accrued interest: coupon × (elapsed fraction of current period)
    # Time to next coupon = first cashflow time; period = 1/freq
    period = 1.0 / freq
    coupon_per_period = face * (coupon_rate / 100) / freq
    time_to_next = flows[0]["t"] if flows else period
    elapsed = period - time_to_next
    accrued = coupon_per_period * (elapsed / period) if period > 0 else 0.0
    clean = price - accrued

    return {
        "dirty_price": round(price, 6),
        "clean_price": round(clean, 6),
        "accrued_interest": round(accrued, 6),
        "macaulay_duration": round(mac_dur, 6),
        "modified_duration": round(mod_dur, 6),
        "convexity": round(convexity, 6),
        "cashflows": enriched,
    }


def bond_price_yield_curve(
    face: float,
    coupon_rate: float,
    maturity: float,
    freq: int = 2,
    ytm_min: float = 0.5,
    ytm_max: float = 15.0,
    n_points: int = 60,
) -> list[dict]:
    """Compute bond price across a range of yields.

    Parameters
    ----------
    face : float
        Face value.
    coupon_rate : float
        Annual coupon rate in percent.
    maturity : float
        Years to maturity.
    freq : int
        Coupon frequency per year.
    ytm_min, ytm_max : float
        Yield range in percent.
    n_points : int
        Number of points on the curve.

    Returns
    -------
    list[dict] with keys: "ytm" (%), "price"
    """
    ytms = np.linspace(ytm_min, ytm_max, n_points)
    result = []
    for y in ytms:
        res = bond_price_from_ytm(face, coupon_rate, maturity, float(y), freq)
        result.append({
            "ytm": round(float(y), 2),
            "price": round(res["dirty_price"], 2),
        })
    return result
