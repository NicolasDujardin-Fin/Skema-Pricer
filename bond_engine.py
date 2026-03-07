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
from scipy.optimize import brentq


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


# ---------------------------------------------------------------------------
# Callable bond pricing
# ---------------------------------------------------------------------------

def callable_bond_tree(
    face: float,
    coupon_rate: float,
    maturity: float,
    ytm: float,
    freq: int,
    call_price: float,
    first_call: float,
    rate_vol: float,
) -> dict:
    """Price a callable bond using a binomial interest rate tree.

    The short rate follows a lognormal recombining tree calibrated so that
    the median rate equals the input YTM.  At each coupon date on or after
    ``first_call``, the issuer calls if the continuation value exceeds
    ``call_price``.

    Both the straight (non-callable) and callable prices are computed on the
    same tree so that the option value is always >= 0.

    Parameters
    ----------
    face : float
        Face value.
    coupon_rate : float
        Annual coupon rate in percent.
    maturity : float
        Years to maturity.
    ytm : float
        Yield to maturity in percent (flat curve assumption).
    freq : int
        Coupon frequency per year.
    call_price : float
        Price at which the issuer can redeem the bond.
    first_call : float
        First call date in years from today.
    rate_vol : float
        Annualised lognormal volatility of the short rate in percent
        (e.g. 20 means 20 %).

    Returns
    -------
    dict with keys:
        - "straight_price" : float — non-callable price from the tree
        - "callable_price"  : float — callable bond price
        - "option_value"    : float — straight − callable (>= 0)
    """
    n = int(round(maturity * freq))
    if n < 1:
        return {"straight_price": face, "callable_price": face, "option_value": 0.0}

    dt = 1.0 / freq
    coupon = face * (coupon_rate / 100) / freq
    r0 = ytm / 100          # annual rate (decimal)
    sigma = rate_vol / 100   # annual rate vol (decimal)

    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = 0.5

    call_step = max(1, int(np.floor(first_call * freq)))

    # Terminal values at step n  (n+1 nodes)
    V_straight = np.full(n + 1, face + coupon, dtype=float)
    V_callable = np.full(n + 1, face + coupon, dtype=float)

    # Backward induction
    for step in range(n - 1, -1, -1):
        V_s_new = np.zeros(step + 1)
        V_c_new = np.zeros(step + 1)

        for j in range(step + 1):
            # Lognormal short rate at node (step, j)
            r_annual = r0 * (u ** (2 * j - step))
            r_period = r_annual / freq
            disc = 1.0 / (1.0 + r_period)

            cont_s = disc * (p * V_straight[j + 1] + (1 - p) * V_straight[j])
            cont_c = disc * (p * V_callable[j + 1] + (1 - p) * V_callable[j])

            if step > 0:
                # Coupon received at this node
                V_s_new[j] = coupon + cont_s
                if step >= call_step:
                    V_c_new[j] = coupon + min(cont_c, call_price)
                else:
                    V_c_new[j] = coupon + cont_c
            else:
                # Step 0: price only (no coupon today)
                V_s_new[j] = cont_s
                V_c_new[j] = cont_c

        V_straight = V_s_new
        V_callable = V_c_new

    straight = float(V_straight[0])
    callable_p = float(V_callable[0])

    return {
        "straight_price": round(straight, 6),
        "callable_price": round(callable_p, 6),
        "option_value": round(max(0.0, straight - callable_p), 6),
    }


def yield_to_call(
    face: float,
    coupon_rate: float,
    call_date: float,
    call_price: float,
    freq: int,
    dirty_price: float,
) -> float | None:
    """Compute the yield to call (YTC).

    Treats the bond as if it matures at ``call_date`` with redemption value
    ``call_price``.  Solves for the yield that equates the discounted cash
    flows to ``dirty_price``.

    Returns None if no solution is found.
    """
    coupon = face * (coupon_rate / 100) / freq
    n = int(round(call_date * freq))
    if n < 1:
        return None

    def _price_at(y: float) -> float:
        y_per = y / 100 / freq
        pv = 0.0
        for i in range(1, n + 1):
            cf = coupon + (call_price if i == n else 0.0)
            pv += cf / (1 + y_per) ** i
        return pv

    try:
        return float(brentq(lambda y: _price_at(y) - dirty_price, 0.01, 200.0))
    except (ValueError, RuntimeError):
        return None


def callable_bond_yield_curve(
    face: float,
    coupon_rate: float,
    maturity: float,
    freq: int,
    call_price: float,
    first_call: float,
    rate_vol: float,
    ytm_min: float = 0.5,
    ytm_max: float = 15.0,
    n_points: int = 60,
) -> list[dict]:
    """Price-yield curves for both straight and callable bonds."""
    ytms = np.linspace(ytm_min, ytm_max, n_points)
    result = []
    for y in ytms:
        yf = float(y)
        res_s = bond_price_from_ytm(face, coupon_rate, maturity, yf, freq)
        res_c = callable_bond_tree(face, coupon_rate, maturity, yf, freq,
                                   call_price, first_call, rate_vol)
        result.append({
            "ytm": round(yf, 2),
            "straight": round(res_s["dirty_price"], 2),
            "callable": round(res_c["callable_price"], 2),
        })
    return result


def callable_bond_tree_detail(
    face: float,
    coupon_rate: float,
    maturity: float,
    ytm: float,
    freq: int,
    call_price: float,
    first_call: float,
    rate_vol: float,
    n_display: int = 6,
) -> list[dict]:
    """Build a small binomial rate tree and return every node for visualisation.

    Uses the same lognormal model as :func:`callable_bond_tree` but with only
    ``n_display`` steps so the tree fits on screen.

    Returns
    -------
    list[dict], one entry per node with keys:
        step, j, rate (%), straight, callable, called (bool)
    """
    n = max(2, n_display)
    dt = maturity / n
    coupon = face * (coupon_rate / 100) * dt
    r0 = ytm / 100
    sigma = rate_vol / 100

    u = np.exp(sigma * np.sqrt(dt))
    p = 0.5
    call_step = max(1, int(round(first_call / dt)))

    # Build rate grid
    rates = []
    for step in range(n + 1):
        rates.append([r0 * (u ** (2 * j - step)) for j in range(step + 1)])

    # Terminal values
    V_s = [face + coupon] * (n + 1)
    V_c = [face + coupon] * (n + 1)

    sv = [None] * (n + 1)
    cv = [None] * (n + 1)
    cf = [None] * (n + 1)

    sv[n] = list(V_s)
    cv[n] = list(V_c)
    cf[n] = [False] * (n + 1)

    for step in range(n - 1, -1, -1):
        V_s_new, V_c_new, called = [], [], []
        for j in range(step + 1):
            r_ann = rates[step][j]
            disc = 1.0 / (1.0 + r_ann * dt)

            cont_s = disc * (p * V_s[j + 1] + (1 - p) * V_s[j])
            cont_c = disc * (p * V_c[j + 1] + (1 - p) * V_c[j])

            if step > 0:
                vs = coupon + cont_s
                if step >= call_step:
                    is_called = cont_c > call_price
                    vc = coupon + min(cont_c, call_price)
                else:
                    is_called = False
                    vc = coupon + cont_c
            else:
                vs = cont_s
                vc = cont_c
                is_called = False

            V_s_new.append(vs)
            V_c_new.append(vc)
            called.append(is_called)

        V_s = V_s_new
        V_c = V_c_new
        sv[step] = list(V_s_new)
        cv[step] = list(V_c_new)
        cf[step] = list(called)

    nodes = []
    for step in range(n + 1):
        for j in range(step + 1):
            nodes.append({
                "step": step,
                "j": j,
                "rate": round(float(rates[step][j]) * 100, 2),
                "straight": round(float(sv[step][j]), 2),
                "callable": round(float(cv[step][j]), 2),
                "called": bool(cf[step][j]),
            })
    return nodes
