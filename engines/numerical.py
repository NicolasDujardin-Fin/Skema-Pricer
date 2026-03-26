"""
numerical_engine.py — American option pricing via CRR binomial tree.

Uses the Cox-Ross-Rubinstein (1979) parameterisation with cost of carry b
so that dividends and repo are handled identically to the BS engine:

    b = r - q - repo

The tree discounts at rate r but the up/down probabilities use the
carry-adjusted growth rate b (same convention as Merton/Black).
"""

import numpy as np


def american_binomial_tree(
    S: float,
    K: float,
    T: float,
    r: float,
    b: float,
    sigma: float,
    n: int,
    option_type: str = "call",
) -> dict:
    """Price an American option using a CRR binomial tree (vectorised).

    Both the American and European prices are computed on the same tree
    so that the early-exercise premium is always >= 0.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free rate.
    b : float
        Net cost of carry  (b = r - q - repo).
    sigma : float
        Annualised volatility.
    n : int
        Number of time steps in the tree (higher = more accurate).
    option_type : str
        "call" or "put".

    Returns
    -------
    dict with keys:
        - "price"    : float — American option price
        - "eu_price" : float — European option price (same tree, no early exercise)
        - "delta"    : float — hedge ratio from the first step of the tree
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    # Risk-neutral probability using cost of carry b
    p = (np.exp(b * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    ot = option_type.lower()
    if ot not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    is_call = ot == "call"

    # Terminal spot prices at step n: S * u^j * d^(n-j) for j = 0..n
    j = np.arange(n + 1)
    S_T = S * u ** j * d ** (n - j)

    # Terminal payoff (same for EU and AM)
    if is_call:
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    am_values = payoff.copy()
    eu_values = payoff.copy()

    # Backward induction
    for step in range(n - 1, -1, -1):
        # Continuation value (same formula for both)
        am_cont = disc * (p * am_values[1:step + 2] + (1 - p) * am_values[:step + 1])
        eu_cont = disc * (p * eu_values[1:step + 2] + (1 - p) * eu_values[:step + 1])

        # European: just continuation
        eu_values = eu_cont

        # American: max(continuation, intrinsic)
        j_step = np.arange(step + 1)
        S_node = S * u ** j_step * d ** (step - j_step)
        if is_call:
            intrinsic = np.maximum(S_node - K, 0.0)
        else:
            intrinsic = np.maximum(K - S_node, 0.0)
        am_values = np.maximum(am_cont, intrinsic)

        # Capture values at step 1 for delta calculation
        if step == 1:
            am_up = float(am_values[1])
            am_dn = float(am_values[0])

    price = float(am_values[0])
    eu_price = float(eu_values[0])

    # Delta from step 1
    S_up = S * u
    S_dn = S * d
    delta = (am_up - am_dn) / (S_up - S_dn)

    return {"price": price, "eu_price": eu_price, "delta": delta}
