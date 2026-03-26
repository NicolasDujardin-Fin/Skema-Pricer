"""
bs_engine.py — Black-Scholes pricer with repo in the cost of carry.

Convention (Merton / Black generalised framework):
    b = r - q - repo   (net cost of carry)

    d1 = (log(S/K) + (b + sigma**2/2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call = S * exp((b-r)*T) * N(d1)  - K * exp(-r*T) * N(d2)
    Put  = K * exp(-r*T)   * N(-d2) - S * exp((b-r)*T) * N(-d1)

Note: exp((b-r)*T) = exp(-(q+repo)*T) discounts the spot for carry leakage.
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _d1_d2(
    S: float,
    K: float,
    T: float,
    b: float,
    sigma: float,
) -> tuple[float, float]:
    """Compute d1 and d2 given the net cost of carry b."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


# ---------------------------------------------------------------------------
# Core pricing and greeks
# ---------------------------------------------------------------------------

def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    repo: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes price of a European option including repo in carry.

    Parameters
    ----------
    S : float
        Spot price of the underlying.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuous dividend yield.
    repo : float
        Continuous repo rate (borrowing cost of the security).
    sigma : float
        Annualised volatility.
    option_type : str
        "call" or "put" (case-insensitive).

    Returns
    -------
    float
        Option price.

    Raises
    ------
    ValueError
        If option_type is not "call" or "put".
    """
    b = r - q - repo
    d1, d2 = _d1_d2(S, K, T, b, sigma)
    carry_disc = np.exp((b - r) * T)   # = exp(-(q + repo) * T)
    rate_disc = np.exp(-r * T)

    ot = option_type.lower()
    if ot == "call":
        return S * carry_disc * norm.cdf(d1) - K * rate_disc * norm.cdf(d2)
    elif ot == "put":
        return K * rate_disc * norm.cdf(-d2) - S * carry_disc * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    repo: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """Closed-form Black-Scholes Greeks including repo in carry.

    Vega is expressed per 1% move in volatility (divided by 100).

    Parameters
    ----------
    S : float
        Spot price of the underlying.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuous dividend yield.
    repo : float
        Continuous repo rate.
    sigma : float
        Annualised volatility.
    option_type : str
        "call" or "put" (case-insensitive).

    Returns
    -------
    dict with keys:
        - "delta" : float
        - "gamma" : float
        - "vega"  : float  (per 1% vol move)
        - "theta" : float  (per calendar day)

    Raises
    ------
    ValueError
        If option_type is not "call" or "put".
    """
    b = r - q - repo
    d1, d2 = _d1_d2(S, K, T, b, sigma)
    sqrt_T = np.sqrt(T)
    carry_disc = np.exp((b - r) * T)
    rate_disc = np.exp(-r * T)
    n_d1 = norm.pdf(d1)   # standard normal density at d1

    # Delta
    ot = option_type.lower()
    if ot == "call":
        delta = carry_disc * norm.cdf(d1)
    elif ot == "put":
        delta = carry_disc * (norm.cdf(d1) - 1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Gamma (same for call and put)
    gamma = carry_disc * n_d1 / (S * sigma * sqrt_T)

    # Vega (same for call and put), per 1% vol move
    vega = S * carry_disc * n_d1 * sqrt_T / 100

    # Theta (per calendar day, divided by 365)
    common = -S * carry_disc * n_d1 * sigma / (2 * sqrt_T)
    if ot == "call":
        theta = (
            common
            - (b - r) * S * carry_disc * norm.cdf(d1)
            - r * K * rate_disc * norm.cdf(d2)
        ) / 365
    else:
        theta = (
            common
            + (b - r) * S * carry_disc * norm.cdf(-d1)
            + r * K * rate_disc * norm.cdf(-d2)
        ) / 365

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


# ---------------------------------------------------------------------------
# Ladder / matrix helpers
# ---------------------------------------------------------------------------

def spot_ladder(
    S_min: float,
    S_max: float,
    n_spots: int,
    K: float,
    T: float,
    r: float,
    q: float,
    repo: float,
    sigma: float,
) -> list[dict]:
    """Compute call/put prices and Greeks across a range of spot prices.

    Parameters
    ----------
    S_min, S_max : float
        Bounds of the spot range.
    n_spots : int
        Number of spot levels.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r, q, repo : float
        Risk-free rate, dividend yield, repo rate.
    sigma : float
        Annualised volatility.

    Returns
    -------
    list[dict] with keys:
        spot, call_price, put_price,
        call_delta, put_delta, gamma, vega,
        call_theta, put_theta
    """
    spots = np.linspace(S_min, S_max, n_spots)
    results = []
    for S in spots:
        call_price = bs_price(S, K, T, r, q, repo, sigma, "call")
        put_price  = bs_price(S, K, T, r, q, repo, sigma, "put")
        cg = bs_greeks(S, K, T, r, q, repo, sigma, "call")
        pg = bs_greeks(S, K, T, r, q, repo, sigma, "put")
        results.append({
            "spot":       S,
            "call_price": call_price,
            "put_price":  put_price,
            "call_delta": cg["delta"],
            "put_delta":  pg["delta"],
            "gamma":      cg["gamma"],   # same for call and put
            "vega":       cg["vega"],    # same for call and put
            "call_theta": cg["theta"],
            "put_theta":  pg["theta"],
        })
    return results


def vol_ladder(
    vol_min: float,
    vol_max: float,
    n_vols: int,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    repo: float,
) -> list[dict]:
    """Compute call/put prices and vega across a range of volatilities.

    Parameters
    ----------
    vol_min, vol_max : float
        Bounds of the volatility range (e.g. 0.05, 0.60).
    n_vols : int
        Number of vol levels.
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r, q, repo : float
        Risk-free rate, dividend yield, repo rate.

    Returns
    -------
    list[dict] with keys:
        vol, call_price, put_price, vega
    """
    vols = np.linspace(vol_min, vol_max, n_vols)
    results = []
    for sigma in vols:
        call_price = bs_price(S, K, T, r, q, repo, sigma, "call")
        put_price  = bs_price(S, K, T, r, q, repo, sigma, "put")
        vega = bs_greeks(S, K, T, r, q, repo, sigma, "call")["vega"]
        results.append({
            "vol":        sigma,
            "call_price": call_price,
            "put_price":  put_price,
            "vega":       vega,
        })
    return results


def spot_vol_matrix(
    S_min: float,
    S_max: float,
    n_spots: int,
    vol_min: float,
    vol_max: float,
    n_vols: int,
    K: float,
    T: float,
    r: float,
    q: float,
    repo: float,
    option_type: str = "call",
) -> dict:
    """Compute a 2-D option price matrix over a spot x vol grid.

    Parameters
    ----------
    S_min, S_max : float
        Spot range bounds.
    n_spots : int
        Number of spot levels.
    vol_min, vol_max : float
        Volatility range bounds.
    n_vols : int
        Number of vol levels.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r, q, repo : float
        Risk-free rate, dividend yield, repo rate.
    option_type : str
        "call" or "put".

    Returns
    -------
    dict with keys:
        - "spots"  : np.ndarray, shape (n_spots,)
        - "vols"   : np.ndarray, shape (n_vols,)
        - "prices" : np.ndarray, shape (n_vols, n_spots)
                     prices[i, j] = price at vol vols[i] and spot spots[j]
    """
    spots = np.linspace(S_min, S_max, n_spots)
    vols  = np.linspace(vol_min, vol_max, n_vols)
    prices = np.zeros((n_vols, n_spots))
    for i, sigma in enumerate(vols):
        for j, S in enumerate(spots):
            prices[i, j] = bs_price(S, K, T, r, q, repo, sigma, option_type)
    return {"spots": spots, "vols": vols, "prices": prices}


# ---------------------------------------------------------------------------
# Gamma ladders
# ---------------------------------------------------------------------------

def gamma_spot_ladder(
    S_min: float,
    S_max: float,
    n_spots: int,
    K: float,
    T: float,
    r: float,
    q: float,
    repo: float,
    sigma: float,
) -> list[dict]:
    """Gamma as a function of spot for a fixed volatility.

    Gamma peaks at-the-money and decays symmetrically on both sides.
    Its shape is a scaled normal PDF centered around the forward price.

    Parameters
    ----------
    S_min, S_max : float
        Spot range bounds.
    n_spots : int
        Number of spot levels.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r, q, repo : float
        Risk-free rate, dividend yield, repo rate.
    sigma : float
        Annualised volatility.

    Returns
    -------
    list[dict] with keys:
        spot, gamma, call_delta, put_delta, call_price, put_price
    """
    spots = np.linspace(S_min, S_max, n_spots)
    results = []
    for S in spots:
        g  = bs_greeks(S, K, T, r, q, repo, sigma, "call")
        gp = bs_greeks(S, K, T, r, q, repo, sigma, "put")
        results.append({
            "spot":       S,
            "gamma":      g["gamma"],          # identical for call and put
            "call_delta": g["delta"],
            "put_delta":  gp["delta"],
            "call_price": bs_price(S, K, T, r, q, repo, sigma, "call"),
            "put_price":  bs_price(S, K, T, r, q, repo, sigma, "put"),
        })
    return results


def gamma_vol_ladder(
    vol_min: float,
    vol_max: float,
    n_vols: int,
    K: float,
    T: float,
    r: float,
    q: float,
    repo: float,
    spots: list[float],
) -> list[dict]:
    """Gamma as a function of volatility for several spot levels (ATM, ITM, OTM).

    Key insight:
    - ATM gamma decreases monotonically as vol rises (the bell flattens).
    - ITM/OTM gamma first rises (bell widens toward these spots) then falls.

    Parameters
    ----------
    vol_min, vol_max : float
        Volatility range bounds.
    n_vols : int
        Number of vol levels.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r, q, repo : float
        Risk-free rate, dividend yield, repo rate.
    spots : list[float]
        Fixed spot prices to evaluate (e.g. [80, 100, 120]).

    Returns
    -------
    list[dict], one entry per vol level, with keys:
        vol, and one gamma_S{s} key per spot in spots.
    """
    vols = np.linspace(vol_min, vol_max, n_vols)
    results = []
    for sigma in vols:
        row: dict = {"vol": sigma}
        for S in spots:
            g = bs_greeks(S, K, T, r, q, repo, sigma, "call")
            row[f"gamma_S{int(S)}"] = g["gamma"]
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Cash delta ladders
# ---------------------------------------------------------------------------

def cash_delta_spot_ladder(
    S_min: float,
    S_max: float,
    n_spots: int,
    K: float,
    T: float,
    r: float,
    q: float,
    repo: float,
    sigma: float,
    n_lots: float = 1.0,
    multiplier: float = 100.0,
) -> list[dict]:
    """Spot ladder enriched with cash delta for call and put.

    Cash delta represents the monetary P&L sensitivity to a unit move in spot:
        cash_delta = n_lots * delta * multiplier * spot

    Parameters
    ----------
    S_min, S_max : float
        Spot range bounds.
    n_spots : int
        Number of spot levels.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r, q, repo : float
        Risk-free rate, dividend yield, repo rate.
    sigma : float
        Annualised volatility.
    n_lots : float
        Number of option contracts held (positive = long).
    multiplier : float
        Contract multiplier (e.g. 100 for standard equity options).

    Returns
    -------
    list[dict] with all keys from spot_ladder plus:
        - "call_cash_delta" : float  (cash delta of the call position)
        - "put_cash_delta"  : float  (cash delta of the put position)
    """
    base = spot_ladder(S_min, S_max, n_spots, K, T, r, q, repo, sigma)
    for row in base:
        S = row["spot"]
        row["call_cash_delta"] = n_lots * row["call_delta"] * multiplier * S
        row["put_cash_delta"]  = n_lots * row["put_delta"]  * multiplier * S
    return base


def delta_vol_ladder(
    vol_min: float,
    vol_max: float,
    n_vols: int,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    repo: float,
    n_lots: float = 1.0,
    multiplier: float = 100.0,
    option_type: str = "call",
) -> list[dict]:
    """Vol ladder focused on delta and cash delta, for a fixed spot and strike.

    Useful to demonstrate the vol-delta relationship:
    - ITM call (S > K): delta > 0.5, decreases toward 0.5 as vol rises.
    - OTM call (S < K): delta < 0.5, increases toward 0.5 as vol rises.
    Both cases converge to 0.5 as vol -> infinity (log-normal flattening).

    Parameters
    ----------
    vol_min, vol_max : float
        Volatility range bounds.
    n_vols : int
        Number of vol levels.
    S : float
        Spot price (fixed).
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r, q, repo : float
        Risk-free rate, dividend yield, repo rate.
    n_lots : float
        Number of contracts held.
    multiplier : float
        Contract multiplier.
    option_type : str
        "call" or "put".

    Returns
    -------
    list[dict] with keys:
        vol, price, delta, cash_delta
    """
    vols = np.linspace(vol_min, vol_max, n_vols)
    results = []
    for sigma in vols:
        price  = bs_price(S, K, T, r, q, repo, sigma, option_type)
        greeks = bs_greeks(S, K, T, r, q, repo, sigma, option_type)
        delta  = greeks["delta"]
        cash_d = n_lots * delta * multiplier * S
        results.append({
            "vol":        sigma,
            "price":      price,
            "delta":      delta,
            "cash_delta": cash_d,
        })
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    S, K, T = 100.0, 100.0, 1.0
    r, q, repo, sigma = 0.05, 0.02, 0.0, 0.20

    sep84 = "=" * 84
    sep_thin = "-" * 84

    # -----------------------------------------------------------------------
    # Test 1 - Base prices and Greeks
    # -----------------------------------------------------------------------
    print(f"\n{sep84}")
    print("  Test 1 - Base prices and Greeks  (S=100, K=100, T=1, r=5%, q=2%, repo=0%, vol=20%)")
    print(sep84)

    call = bs_price(S, K, T, r, q, repo, sigma, "call")
    put  = bs_price(S, K, T, r, q, repo, sigma, "put")
    cg   = bs_greeks(S, K, T, r, q, repo, sigma, "call")
    pg   = bs_greeks(S, K, T, r, q, repo, sigma, "put")

    print(f"\n  Call = {call:.4f},  Put = {put:.4f}")
    print(f"\n  {'Greek':<12} {'Call':>12} {'Put':>12}")
    print(f"  {'-'*36}")
    for greek in ("delta", "gamma", "vega", "theta"):
        print(f"  {greek.capitalize():<12} {cg[greek]:>12.6f} {pg[greek]:>12.6f}")

    # Put-call parity check: C - P = S*exp(-(q+repo)*T) - K*exp(-r*T)
    pcp_lhs = call - put
    pcp_rhs = S * np.exp(-(q + repo) * T) - K * np.exp(-r * T)
    print(f"\n  Put-call parity check:")
    print(f"    C - P         = {pcp_lhs:.6f}")
    print(f"    Fwd PV(S-K)   = {pcp_rhs:.6f}")
    print(f"    Residual      = {abs(pcp_lhs - pcp_rhs):.2e}  (should be ~0)")

    # -----------------------------------------------------------------------
    # Test 2 - Impact of repo
    # -----------------------------------------------------------------------
    repo2 = 0.01
    call2 = bs_price(S, K, T, r, q, repo2, sigma, "call")
    put2  = bs_price(S, K, T, r, q, repo2, sigma, "put")

    print(f"\n{sep84}")
    print("  Test 2 - Impact of repo  (repo=1% vs repo=0%)")
    print(sep84)
    print(f"\n  Call (no repo)   = {call:.4f}   |  Put (no repo)   = {put:.4f}")
    print(f"  Call (repo=1%)   = {call2:.4f}   |  Put (repo=1%)   = {put2:.4f}")
    print(f"  Delta call       = {call2 - call:+.4f}   |  Delta put       = {put2 - put:+.4f}")
    print(f"\n  Note: repo lowers the forward => call cheaper, put more expensive.")

    # -----------------------------------------------------------------------
    # Test 3 - Spot ladder
    # -----------------------------------------------------------------------
    ladder_s = spot_ladder(70, 130, 13, K, T, r, q, repo, sigma)

    cw = 11
    print(f"\n{sep84}")
    print("  Test 3 - Spot ladder  (K=100, T=1, r=5%, q=2%, repo=0%, vol=20%)")
    print(sep84)
    hdr = (
        f"{'Spot':>{cw}}"
        f"{'Call':>{cw}}"
        f"{'Put':>{cw}}"
        f"{'C.Delta':>{cw}}"
        f"{'P.Delta':>{cw}}"
        f"{'Gamma':>{cw}}"
        f"{'Vega':>{cw}}"
    )
    print(f"\n{hdr}")
    print(sep_thin)
    for row in ladder_s:
        print(
            f"{row['spot']:>{cw}.1f}"
            f"{row['call_price']:>{cw}.4f}"
            f"{row['put_price']:>{cw}.4f}"
            f"{row['call_delta']:>{cw}.4f}"
            f"{row['put_delta']:>{cw}.4f}"
            f"{row['gamma']:>{cw}.6f}"
            f"{row['vega']:>{cw}.4f}"
        )

    # -----------------------------------------------------------------------
    # Test 4 - Vol ladder
    # -----------------------------------------------------------------------
    ladder_v = vol_ladder(0.05, 0.60, 12, S, K, T, r, q, repo)

    print(f"\n{sep84}")
    print("  Test 4 - Vol ladder  (S=100, K=100, T=1, r=5%, q=2%, repo=0%)")
    print(sep84)
    hdr2 = (
        f"{'Vol':>{cw}}"
        f"{'Call':>{cw}}"
        f"{'Put':>{cw}}"
        f"{'Vega':>{cw}}"
    )
    print(f"\n{hdr2}")
    print("-" * (cw * 4))
    for row in ladder_v:
        print(
            f"{row['vol']:>{cw}.2%}"
            f"{row['call_price']:>{cw}.4f}"
            f"{row['put_price']:>{cw}.4f}"
            f"{row['vega']:>{cw}.4f}"
        )

    # -----------------------------------------------------------------------
    # Test 5 - Cash delta spot ladder
    # -----------------------------------------------------------------------
    n_lots, multiplier = 10, 100   # 10 contracts, 100 shares each

    ladder_cd = cash_delta_spot_ladder(70, 130, 13, K, T, r, q, repo, sigma,
                                       n_lots=n_lots, multiplier=multiplier)

    cw2 = 14
    print(f"\n{sep84}")
    print(f"  Test 5 - Cash delta spot ladder")
    print(f"  K=100, T=1, r=5%, q=2%, repo=0%, vol=20%  |  {n_lots} lots x {multiplier} multiplier")
    print(sep84)
    print(f"  cash_delta = n_lots * delta * multiplier * spot")
    hdr5 = (
        f"{'Spot':>{cw2}}"
        f"{'C.Delta':>{cw2}}"
        f"{'P.Delta':>{cw2}}"
        f"{'C.CashDelta':>{cw2}}"
        f"{'P.CashDelta':>{cw2}}"
    )
    print(f"\n{hdr5}")
    print("-" * (cw2 * 5))
    for row in ladder_cd:
        print(
            f"{row['spot']:>{cw2}.1f}"
            f"{row['call_delta']:>{cw2}.4f}"
            f"{row['put_delta']:>{cw2}.4f}"
            f"{row['call_cash_delta']:>{cw2}.1f}"
            f"{row['put_cash_delta']:>{cw2}.1f}"
        )
    print(f"  Note: call cash delta is always positive (long exposure),")
    print(f"  put cash delta always negative. Both peak/trough around ATM.")

    # -----------------------------------------------------------------------
    # Test 6 - Delta vol ladder: ITM vs OTM convergence
    # -----------------------------------------------------------------------
    print(f"\n{sep84}")
    print(f"  Test 6 - Delta vs Vol  |  K=100, T=1, r=5%, q=2%, repo=0%  |  {n_lots} lots x {multiplier}")
    print(sep84)
    print(f"  Calls: ITM (S=120) delta starts HIGH and falls toward 0.5 as vol rises.")
    print(f"         OTM (S= 80) delta starts LOW  and rises toward 0.5 as vol rises.")

    itm_ladder = delta_vol_ladder(0.05, 0.80, 16, S=120, K=100,
                                   T=T, r=r, q=q, repo=repo,
                                   n_lots=n_lots, multiplier=multiplier,
                                   option_type="call")
    otm_ladder = delta_vol_ladder(0.05, 0.80, 16, S=80,  K=100,
                                   T=T, r=r, q=q, repo=repo,
                                   n_lots=n_lots, multiplier=multiplier,
                                   option_type="call")

    cw3 = 13
    hdr6 = (
        f"{'Vol':>{cw3}}"
        f"{'ITM delta':>{cw3}}"
        f"{'ITM $delta':>{cw3}}"
        f"{'OTM delta':>{cw3}}"
        f"{'OTM $delta':>{cw3}}"
        f"  {'Direction'}"
    )
    print(f"\n{hdr6}")
    print("-" * (cw3 * 5 + 14))
    for itm, otm in zip(itm_ladder, otm_ladder):
        itm_dir = "v" if itm_ladder.index(itm) > 0 and itm["delta"] < itm_ladder[itm_ladder.index(itm)-1]["delta"] else " "
        otm_dir = "^" if otm_ladder.index(otm) > 0 and otm["delta"] > otm_ladder[otm_ladder.index(otm)-1]["delta"] else " "
        print(
            f"{itm['vol']:>{cw3}.2%}"
            f"{itm['delta']:>{cw3}.4f}"
            f"{itm['cash_delta']:>{cw3}.1f}"
            f"{otm['delta']:>{cw3}.4f}"
            f"{otm['cash_delta']:>{cw3}.1f}"
            f"  ITM:{itm_dir}  OTM:{otm_dir}"
        )
    delta_inf = np.exp(-(q + repo) * T)
    print(f"\n  Asymptotic limit as vol -> inf: delta -> exp(-(q+repo)*T) = {delta_inf:.4f} for BOTH.")
    print(f"  ITM starts near {delta_inf:.2f} (vol~0 limit), dips, then climbs back.")
    print(f"  OTM starts near 0 (vol~0 limit), climbs steadily toward the same limit.")

    # -----------------------------------------------------------------------
    # Test 7 - Gamma spot ladder (fixed vol)
    # -----------------------------------------------------------------------
    print(f"\n{sep84}")
    print(f"  Test 7 - Gamma vs Spot  |  K=100, T=1, r=5%, q=2%, repo=0%, vol=20%")
    print(f"  Gamma = exp(-(q+repo)*T) * n(d1) / (S * sigma * sqrt(T))")
    print(f"  Peaks ATM, decays on both sides (bell shape centred near forward).")
    print(sep84)

    gsl = gamma_spot_ladder(70, 130, 13, K, T, r, q, repo, sigma)

    cw_g = 12
    hdr7 = (
        f"{'Spot':>{cw_g}}"
        f"{'Gamma':>{cw_g}}"
        f"{'C.Delta':>{cw_g}}"
        f"{'P.Delta':>{cw_g}}"
        f"{'Call':>{cw_g}}"
        f"{'Put':>{cw_g}}"
    )
    print(f"\n{hdr7}")
    print("-" * (cw_g * 6))
    for row in gsl:
        # bar: visual indicator proportional to gamma (max 20 chars)
        max_g = max(r["gamma"] for r in gsl)
        bar = "#" * int(row["gamma"] / max_g * 20)
        print(
            f"{row['spot']:>{cw_g}.1f}"
            f"{row['gamma']:>{cw_g}.6f}"
            f"{row['call_delta']:>{cw_g}.4f}"
            f"{row['put_delta']:>{cw_g}.4f}"
            f"{row['call_price']:>{cw_g}.4f}"
            f"{row['put_price']:>{cw_g}.4f}"
            f"  {bar}"
        )

    # -----------------------------------------------------------------------
    # Test 8 - Gamma vol ladder (ATM / ITM / OTM)
    # -----------------------------------------------------------------------
    spot_levels = [80, 100, 120]   # OTM, ATM, ITM
    gvl = gamma_vol_ladder(0.05, 0.80, 16, K, T, r, q, repo, spots=spot_levels)

    print(f"\n{sep84}")
    print(f"  Test 8 - Gamma vs Vol  |  K=100, T=1, r=5%, q=2%, repo=0%")
    print(f"  ATM  (S=100): gamma falls monotonically as vol rises (bell flattens).")
    print(f"  OTM  (S= 80): gamma rises first (bell widens to reach S=80) then falls.")
    print(f"  ITM  (S=120): same hump behaviour as OTM by symmetry.")
    print(sep84)

    cw_v = 14
    hdr8 = (
        f"{'Vol':>{cw_v}}"
        + "".join(f"{'Gamma S='+str(s):>{cw_v}}" for s in spot_levels)
    )
    print(f"\n{hdr8}")
    print("-" * (cw_v * (1 + len(spot_levels))))
    prev = None
    for row in gvl:
        parts = f"{row['vol']:>{cw_v}.2%}"
        for S in spot_levels:
            key = f"gamma_S{int(S)}"
            g = row[key]
            if prev is not None:
                arrow = "v" if g < prev[key] else "^"
            else:
                arrow = " "
            parts += f"{g:>{cw_v - 1}.6f}{arrow}"
        print(parts)
        prev = row

    print(f"\n  Rule: ATM gamma always ^ when vol falls (=> long gamma = long vol ATM).")
    print(f"  OTM/ITM gamma has a hump: it peaks at the vol where the bell covers that spot.")

    print()
