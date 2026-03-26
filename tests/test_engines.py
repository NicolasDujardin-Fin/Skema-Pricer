"""Smoke tests for pricing engines."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engines.bs import bs_price, bs_greeks
from engines.bond import bond_price_from_ytm
from engines.discount import discount_certificate_price
from engines.bonus import bonus_certificate_price
from engines.numerical import american_binomial_tree
from engines.rates import price_forward


def test_bs_call_put_parity():
    S, K, T, r, q, repo, sigma = 100, 100, 1.0, 0.05, 0.02, 0.0, 0.20
    call = bs_price(S, K, T, r, q, repo, sigma, "call")
    put = bs_price(S, K, T, r, q, repo, sigma, "put")
    fwd = price_forward(S, r, q, repo, T)
    import math
    parity_diff = call - put - (fwd - K) * math.exp(-r * T)
    assert abs(parity_diff) < 0.01, f"Put-call parity violated: {parity_diff}"


def test_bs_greeks_delta_range():
    g = bs_greeks(100, 100, 1.0, 0.05, 0.02, 0.0, 0.20, "call")
    assert 0 < g["delta"] < 1, f"Call delta out of range: {g['delta']}"
    assert g["gamma"] > 0, f"Gamma should be positive: {g['gamma']}"


def test_bond_par():
    res = bond_price_from_ytm(1000, 5.0, 10, 5.0, 2)
    assert abs(res["dirty_price"] - 1000) < 0.01, f"Par bond not at par: {res['dirty_price']}"


def test_discount_cert():
    res = discount_certificate_price(100, 110, 1.0, 0.05, 0.02, 0.20, 1.0)
    assert res["dc_price"] < 100, f"DC should be cheaper than spot: {res['dc_price']}"
    assert res["discount_pct"] > 0, f"Discount should be positive: {res['discount_pct']}"


def test_bonus_cert():
    res = bonus_certificate_price(100, 120, 70, 1.0, 0.05, 0.02, 0.22, 140, 1.0, sigma_call=0.18)
    assert res["bc_price"] > 0, f"BC price should be positive: {res['bc_price']}"
    assert res["put_do_price"] > 0, f"Put D&O should be positive: {res['put_do_price']}"


def test_american_premium():
    am = american_binomial_tree(100, 100, 1.0, 0.05, 0.03, 0.30, 100, "put")
    assert am["price"] >= am["eu_price"], "American put should be >= European"


if __name__ == "__main__":
    test_bs_call_put_parity()
    test_bs_greeks_delta_range()
    test_bond_par()
    test_discount_cert()
    test_bonus_cert()
    test_american_premium()
    print("All tests passed")
