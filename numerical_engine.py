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
        - "price"  : float — American option price
        - "delta"  : float — hedge ratio from the first step of the tree
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

    # Terminal spot prices at step n: S * u^j * d^(n-j) for j = 0..n
    j = np.arange(n + 1)
    S_T = S * u ** j * d ** (n - j)

    # Terminal payoff
    if ot == "call":
        values = np.maximum(S_T - K, 0.0)
    else:
        values = np.maximum(K - S_T, 0.0)

    # Backward induction with early-exercise check
    for step in range(n - 1, -1, -1):
        j = np.arange(step + 1)
        S_node = S * u ** j * d ** (step - j)
        continuation = disc * (p * values[1:step + 2] + (1 - p) * values[:step + 1])
        if ot == "call":
            intrinsic = np.maximum(S_node - K, 0.0)
        else:
            intrinsic = np.maximum(K - S_node, 0.0)
        values = np.maximum(continuation, intrinsic)

    price = float(values[0])

    # Delta from the first step
    S_up = S * u
    S_dn = S * d
    if ot == "call":
        v_up = float(disc * (p * np.maximum(S * u * u - K, 0.0) + (1 - p) * np.maximum(S - K, 0.0)))
        v_dn = float(disc * (p * np.maximum(S - K, 0.0) + (1 - p) * np.maximum(S * d * d - K, 0.0)))
    else:
        v_up = float(disc * (p * np.maximum(K - S * u * u, 0.0) + (1 - p) * np.maximum(K - S, 0.0)))
        v_dn = float(disc * (p * np.maximum(K - S, 0.0) + (1 - p) * np.maximum(K - S * d * d, 0.0)))

    # Recompute v_up and v_dn properly from the full tree for accuracy
    # Use the actual tree values at step 1
    # Re-run a mini backward induction for step-1 values
    j1 = np.arange(n + 1)
    S_T1 = S * u * u ** j1 * d ** (n - j1)  # paths from S_up
    if ot == "call":
        vals_up = np.maximum(S_T1 - K, 0.0)
    else:
        vals_up = np.maximum(K - S_T1, 0.0)
    for step in range(n - 1, 0, -1):
        j = np.arange(step + 1)
        S_node = S * u * u ** j * d ** (step - j)
        cont = disc * (p * vals_up[1:step + 2] + (1 - p) * vals_up[:step + 1])
        if ot == "call":
            intr = np.maximum(S_node - K, 0.0)
        else:
            intr = np.maximum(K - S_node, 0.0)
        vals_up = np.maximum(cont, intr)
    f_up = float(vals_up[0])

    j1 = np.arange(n + 1)
    S_T1 = S * d * u ** j1 * d ** (n - j1)  # paths from S_dn
    if ot == "call":
        vals_dn = np.maximum(S_T1 - K, 0.0)
    else:
        vals_dn = np.maximum(K - S_T1, 0.0)
    for step in range(n - 1, 0, -1):
        j = np.arange(step + 1)
        S_node = S * d * u ** j * d ** (step - j)
        cont = disc * (p * vals_dn[1:step + 2] + (1 - p) * vals_dn[:step + 1])
        if ot == "call":
            intr = np.maximum(S_node - K, 0.0)
        else:
            intr = np.maximum(K - S_node, 0.0)
        vals_dn = np.maximum(cont, intr)
    f_dn = float(vals_dn[0])

    delta = (f_up - f_dn) / (S_up - S_dn)

    return {"price": price, "delta": delta}
