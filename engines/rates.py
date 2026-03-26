"""
rates_engine.py — Spot rate curve, discount factors, and forward rates.
All rates are continuous (continuously compounded).
"""

import numpy as np
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def spot_rate_curve(maturities: list[float], rates: list[float]) -> dict:
    """Build a spot rate curve with associated discount factors.

    Parameters
    ----------
    maturities : list[float]
        Maturities in years, e.g. [0.5, 1, 2, 3, 5, 7, 10].
    rates : list[float]
        Continuously compounded spot rates corresponding to each maturity.

    Returns
    -------
    dict with keys:
        - "maturities"       : np.ndarray of maturities
        - "spot_rates"       : np.ndarray of spot rates
        - "discount_factors" : np.ndarray, exp(-r_i * T_i)
    """
    T = np.array(maturities, dtype=float)
    r = np.array(rates, dtype=float)
    df = np.exp(-r * T)
    return {
        "maturities": T,
        "spot_rates": r,
        "discount_factors": df,
    }


def forward_rate(spot_rates: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    """Compute instantaneous forward rates between consecutive maturities.

    The continuously compounded forward rate between T1 and T2 is:
        f(T1, T2) = (r2 * T2 - r1 * T1) / (T2 - T1)

    Parameters
    ----------
    spot_rates : np.ndarray
        Continuously compounded spot rates.
    maturities : np.ndarray
        Maturities in years, same length as spot_rates.

    Returns
    -------
    np.ndarray of length len(maturities) - 1.
    Each value corresponds to the midpoint of the interval [T1, T2].
    """
    r = np.asarray(spot_rates, dtype=float)
    T = np.asarray(maturities, dtype=float)
    fwd = (r[1:] * T[1:] - r[:-1] * T[:-1]) / (T[1:] - T[:-1])
    return fwd


def interpolate_spot_curve(
    maturities: list[float],
    rates: list[float],
    n_points: int = 100,
) -> dict:
    """Interpolate a spot rate curve using a cubic spline and derive forwards.

    Parameters
    ----------
    maturities : list[float]
        Pillar maturities in years.
    rates : list[float]
        Continuously compounded spot rates at each pillar.
    n_points : int, optional
        Number of points on the fine grid (default 100).

    Returns
    -------
    dict with keys:
        - "maturities_fine"   : np.ndarray of interpolated maturities
        - "spot_rates_fine"   : np.ndarray of interpolated spot rates
        - "forward_rates_fine": np.ndarray of forward rates on the fine grid
                                (length n_points - 1)
    """
    T = np.array(maturities, dtype=float)
    r = np.array(rates, dtype=float)

    cs = CubicSpline(T, r)
    T_fine = np.linspace(T.min(), T.max(), n_points)
    r_fine = cs(T_fine)
    fwd_fine = forward_rate(r_fine, T_fine)

    return {
        "maturities_fine": T_fine,
        "spot_rates_fine": r_fine,
        "forward_rates_fine": fwd_fine,
    }


# ---------------------------------------------------------------------------
# Forward pricing functions
# ---------------------------------------------------------------------------

def price_forward(S: float, r: float, q: float, repo: float, T: float) -> float:
    """Compute the forward price of an asset using continuous rates.

    Formula: F = S * exp((r - q - repo) * T)

    Parameters
    ----------
    S : float
        Spot price of the asset.
    r : float
        Continuously compounded risk-free rate (annualised).
    q : float
        Continuous dividend yield (annualised). Use 0 if no dividend.
    repo : float
        Continuous repo rate (annualised). Represents the borrowing cost of
        the security for the title lender. Use 0 if no repo.
    T : float
        Maturity in years.

    Returns
    -------
    float
        Forward price F.
    """
    return S * np.exp((r - q - repo) * T)


def forward_ladder(
    S: float,
    r: float,
    q: float,
    repo: float,
    maturities: list[float],
) -> list[dict]:
    """Compute forward prices for a series of maturities using a flat rate.

    Parameters
    ----------
    S : float
        Spot price of the asset.
    r : float
        Flat continuously compounded risk-free rate.
    q : float
        Continuous dividend yield.
    repo : float
        Continuous repo rate.
    maturities : list[float]
        List of maturities in years.

    Returns
    -------
    list[dict], one entry per maturity, each with keys:
        - "maturity"   : float
        - "forward"    : float
        - "basis"      : float  (F - S)
        - "basis_pct"  : float  ((F - S) / S * 100)
    """
    results = []
    for T in maturities:
        F = price_forward(S, r, q, repo, T)
        basis = F - S
        basis_pct = basis / S * 100
        results.append({"maturity": T, "forward": F, "basis": basis, "basis_pct": basis_pct})
    return results


def forward_ladder_with_curve(
    S: float,
    spot_maturities: list[float],
    spot_rates: list[float],
    q: float,
    repo: float,
    target_maturities: list[float],
) -> list[dict]:
    """Compute forward prices by interpolating the risk-free rate from a spot curve.

    The rate at each target maturity is obtained via linear interpolation
    (numpy.interp) of the provided spot curve.

    Parameters
    ----------
    S : float
        Spot price of the asset.
    spot_maturities : list[float]
        Pillar maturities of the spot rate curve.
    spot_rates : list[float]
        Continuously compounded spot rates at each pillar.
    q : float
        Continuous dividend yield.
    repo : float
        Continuous repo rate.
    target_maturities : list[float]
        Maturities at which to compute forward prices.

    Returns
    -------
    list[dict], one entry per target maturity, each with keys:
        - "maturity"   : float
        - "rate_used"  : float  (interpolated spot rate)
        - "forward"    : float
        - "basis"      : float  (F - S)
        - "basis_pct"  : float  ((F - S) / S * 100)
    """
    xp = np.array(spot_maturities, dtype=float)
    fp = np.array(spot_rates, dtype=float)
    results = []
    for T in target_maturities:
        r = float(np.interp(T, xp, fp))
        F = price_forward(S, r, q, repo, T)
        basis = F - S
        basis_pct = basis / S * 100
        results.append({
            "maturity": T,
            "rate_used": r,
            "forward": F,
            "basis": basis,
            "basis_pct": basis_pct,
        })
    return results


# ---------------------------------------------------------------------------
# Helpers for terminal output
# ---------------------------------------------------------------------------

def _format_table(curve: dict, label: str) -> None:
    """Print a formatted table for a rate curve scenario.

    Parameters
    ----------
    curve : dict
        Output of spot_rate_curve().
    label : str
        Scenario name shown in the header.
    """
    T = curve["maturities"]
    r = curve["spot_rates"]
    df = curve["discount_factors"]
    fwd = forward_rate(r, T)

    col_w = 16
    header = (
        f"{'Maturity':>{col_w}}"
        f"{'Spot Rate':>{col_w}}"
        f"{'Discount Factor':>{col_w}}"
        f"{'Forward Rate':>{col_w}}"
        f"  {'Comment'}"
    )
    sep = "-" * (col_w * 4 + 20)

    print(f"\n{'='*len(sep)}")
    print(f"  {label}")
    print(f"{'='*len(sep)}")
    print(header)
    print(sep)

    for i, (Ti, ri, dfi) in enumerate(zip(T, r, df)):
        if i < len(fwd):
            fi = fwd[i]
            comment = "fwd > spot" if fi > ri else "fwd < spot"
            fwd_str = f"{fi:.4%}"
        else:
            fi = float("nan")
            comment = "-"
            fwd_str = "    -    "

        print(
            f"{Ti:>{col_w}.2f}"
            f"{ri:>{col_w}.4%}"
            f"{dfi:>{col_w}.6f}"
            f"{fwd_str:>{col_w}}"
            f"  {comment}"
        )

    print(sep)
    # Summary: compare average forward vs average spot
    avg_spot = r.mean()
    avg_fwd = fwd.mean()
    verdict = "forwards are above spots on average" if avg_fwd > avg_spot else "forwards are below spots on average"
    print(f"  Avg spot: {avg_spot:.4%}  |  Avg forward: {avg_fwd:.4%}  =>  {verdict}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenarios = [
        (
            "Scenario 1 - Upward sloping",
            [0.5, 1, 2, 3, 5, 7, 10],
            [0.02, 0.025, 0.03, 0.035, 0.04, 0.042, 0.045],
        ),
        (
            "Scenario 2 - Downward sloping (inverted)",
            [0.5, 1, 2, 3, 5, 7, 10],
            [0.05, 0.048, 0.044, 0.04, 0.035, 0.032, 0.03],
        ),
        (
            "Scenario 3 - Humped",
            [0.5, 1, 2, 3, 5, 7, 10],
            [0.02, 0.03, 0.038, 0.04, 0.038, 0.035, 0.032],
        ),
    ]

    for label, mats, rates in scenarios:
        curve = spot_rate_curve(mats, rates)
        _format_table(curve, label)

    # -----------------------------------------------------------------------
    # Forward pricing tests
    # -----------------------------------------------------------------------
    sep = "=" * 84
    print(f"\n{sep}")
    print("  Forward Pricing")
    print(sep)

    # Test 1 - Simple forward, flat rate
    F1 = price_forward(S=100, r=0.05, q=0.02, repo=0.0, T=1)
    basis1 = F1 - 100
    basis_pct1 = basis1 / 100 * 100
    print(f"\nTest 1 - Simple forward (flat rate, no repo):")
    print(f"  S=100, r=5%, q=2%, repo=0%, T=1Y")
    print(f"  Forward 1Y = {F1:.4f}, Basis = {basis1:.4f} ({basis_pct1:.2f}%)")
    print(f"  Expected:  F = 100 * exp(0.03) = {100 * np.exp(0.03):.4f}")

    # Test 2 - Forward with repo
    F2 = price_forward(S=100, r=0.05, q=0.02, repo=0.01, T=1)
    print(f"\nTest 2 - Forward with repo:")
    print(f"  S=100, r=5%, q=2%, repo=1%, T=1Y")
    print(f"  Forward 1Y with repo = {F2:.4f}")
    print(f"  Expected:  F = 100 * exp(0.02) = {100 * np.exp(0.02):.4f}")
    print(f"  Note: repo reduces the forward — it is a carry cost for the title lender.")

    # Test 3 - Forward ladder, flat rate
    ladder3 = forward_ladder(S=100, r=0.04, q=0.01, repo=0.0,
                             maturities=[0.25, 0.5, 1, 2, 3, 5])
    col_w = 14
    print(f"\nTest 3 - Forward ladder (flat rate: r=4%, q=1%, repo=0%):")
    header3 = (
        f"{'Maturity':>{col_w}}"
        f"{'Rate':>{col_w}}"
        f"{'Forward':>{col_w}}"
        f"{'Basis':>{col_w}}"
        f"{'Basis%':>{col_w}}"
    )
    print(f"  {header3}")
    print(f"  {'-' * (col_w * 5)}")
    for row in ladder3:
        print(
            f"  {row['maturity']:>{col_w}.2f}"
            f"{'4.00%':>{col_w}}"
            f"{row['forward']:>{col_w}.4f}"
            f"{row['basis']:>{col_w}.4f}"
            f"{row['basis_pct']:>{col_w}.2f}"
        )

    # Test 4 - Forward ladder with upward sloping curve
    spot_mats = [0.5, 1, 2, 3, 5, 7, 10]
    spot_rts = [0.02, 0.025, 0.03, 0.035, 0.04, 0.042, 0.045]
    ladder4 = forward_ladder_with_curve(
        S=100,
        spot_maturities=spot_mats,
        spot_rates=spot_rts,
        q=0.015,
        repo=0.005,
        target_maturities=[0.5, 1, 2, 3, 5, 7, 10],
    )
    col_w2 = 16
    print(f"\nTest 4 - Forward ladder with curve (upward sloping, q=1.5%, repo=0.5%):")
    header4 = (
        f"{'Maturity':>{col_w2}}"
        f"{'Spot Rate Used':>{col_w2}}"
        f"{'Forward Price':>{col_w2}}"
        f"{'Basis':>{col_w2}}"
        f"{'Basis%':>{col_w2}}"
    )
    print(f"  {header4}")
    print(f"  {'-' * (col_w2 * 5)}")
    for row in ladder4:
        print(
            f"  {row['maturity']:>{col_w2}.2f}"
            f"  {row['rate_used']:>{col_w2 - 2}.4%}"
            f"{row['forward']:>{col_w2}.4f}"
            f"{row['basis']:>{col_w2}.4f}"
            f"{row['basis_pct']:>{col_w2}.2f}"
        )
    print(f"  Note: basis grows with maturity — carry cost compounds on a steeper curve.")

    print()
