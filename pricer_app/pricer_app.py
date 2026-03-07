"""
pricer_app.py — Reflex GUI for the Skema Options Pricer.

2 tabs:
  1. Pricer  — BS price, Greeks, cash delta/gamma + 3 charts vs spot
  2. Forward — Forward pricing, basis, maturity ladder
"""

import datetime
import math
import os
import sys

# Make bs_engine and rates_engine importable from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Any

# Default maturity = today + 1 year
_DEFAULT_MATURITY = (datetime.date.today() + datetime.timedelta(days=365)).strftime("%Y-%m-%d")

import reflex as rx

from bs_engine import (
    bs_greeks,
    bs_price,
    cash_delta_spot_ladder,
    delta_vol_ladder,
    gamma_spot_ladder,
    gamma_vol_ladder,
)
from bond_engine import bond_price_from_ytm, bond_price_yield_curve
from numerical_engine import american_binomial_tree
from rates_engine import forward_ladder, price_forward


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(rx.State):
    # ------------------------------------------------------------------
    # Base input parameters
    # ------------------------------------------------------------------
    S: float = 100.0
    K: float = 100.0
    T: float = 1.0
    maturity_date: str = _DEFAULT_MATURITY
    r: float = 0.05
    q: float = 0.02
    repo: float = 0.0
    sigma: float = 0.20
    n_lots: float = 1.0
    multiplier: float = 100.0
    option_type: str = "call"
    selected_chart: str = "Price"
    selected_chart2: str = "Price"
    selected_axis2: str = "Vol"
    spot_move_pct: float = 1.0
    iv_for_move: float = 20.0
    tick_size: float = 0.01
    qa_gamma: float = 20000000.0
    qa_gamma_str: str = "20,000,000"
    qa_vol: float = 22.0
    bin_steps: int = 100

    # ------------------------------------------------------------------
    # Bond pricer inputs
    # ------------------------------------------------------------------
    bond_face: float = 1000.0
    bond_coupon: float = 5.0
    bond_maturity: float = 10.0
    bond_ytm: float = 5.0
    bond_freq: int = 2

    # ------------------------------------------------------------------
    # Ladder range controls
    # ------------------------------------------------------------------
    spot_range_pct: float = 0.30
    n_spots: int = 21
    vol_min: float = 0.05
    vol_max: float = 0.60
    n_vols: int = 21
    fwd_maturities_str: str = "0.25,0.5,1,2,3,5"

    # ------------------------------------------------------------------
    # Computed vars — derived automatically from base vars
    # ------------------------------------------------------------------

    @rx.var(cache=True)
    def K_ref(self) -> int:
        """K as int for recharts reference_line (accepts str|int only)."""
        return int(round(self.K))

    @rx.var(cache=True)
    def S_min(self) -> float:
        center = self.K
        half = max(self.spot_range_pct * center, abs(self.S - self.K) * 1.3)
        return round(center - half, 4)

    @rx.var(cache=True)
    def S_max(self) -> float:
        center = self.K
        half = max(self.spot_range_pct * center, abs(self.S - self.K) * 1.3)
        return round(center + half, 4)

    @rx.var(cache=True)
    def T_years_str(self) -> str:
        return f"{self.T:.4f} y"

    # Display vars for % inputs
    @rx.var(cache=True)
    def r_pct(self) -> float:
        return round(self.r * 100, 4)

    @rx.var(cache=True)
    def q_pct(self) -> float:
        return round(self.q * 100, 4)

    @rx.var(cache=True)
    def repo_pct(self) -> float:
        return round(self.repo * 100, 4)

    @rx.var(cache=True)
    def sigma_pct(self) -> float:
        return round(self.sigma * 100, 4)

    @rx.var(cache=True)
    def forward_price(self) -> str:
        try:
            f = price_forward(self.S, self.r, self.q, self.repo, self.T)
            return f"{f:.4f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def bs_price_display(self) -> str:
        try:
            p = bs_price(self.S, self.K, self.T, self.r,
                         self.q, self.repo, self.sigma, self.option_type)
            return f"{p:.4f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def delta_display(self) -> str:
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            return f"{g['delta']:.6f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def gamma_display(self) -> str:
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            return f"{g['gamma']:.6f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def vega_display(self) -> str:
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            return f"{g['vega']:.6f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def theta_display(self) -> str:
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            return f"{g['theta']:.6f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def cash_delta_display(self) -> str:
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            cd = self.n_lots * g["delta"] * self.multiplier * self.S
            return f"{cd:,.2f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def cash_gamma_display(self) -> str:
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            cg = self.n_lots * g["gamma"] * self.multiplier * self.S ** 2 / 100
            return f"{cg:,.2f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def cash_theta_display(self) -> str:
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            ct = self.n_lots * g["theta"] * self.multiplier
            return f"{ct:,.2f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def cash_vega_display(self) -> str:
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            cv = self.n_lots * g["vega"] * self.multiplier
            return f"{cv:,.2f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def cash_charm_display(self) -> str:
        """Cash charm = change in cash delta after 1 calendar day."""
        try:
            dt = 1.0 / 365.0
            T1 = max(0.001, self.T - dt)
            d_now = bs_greeks(self.S, self.K, self.T, self.r,
                              self.q, self.repo, self.sigma, self.option_type)["delta"]
            d_tm1 = bs_greeks(self.S, self.K, T1, self.r,
                              self.q, self.repo, self.sigma, self.option_type)["delta"]
            charm = (d_tm1 - d_now) * self.n_lots * self.multiplier * self.S
            return f"{charm:,.2f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def n_stocks_display(self) -> str:
        """Number of shares equivalent to the current cash delta hedge."""
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            n = self.n_lots * g["delta"] * self.multiplier
            return f"{n:,.1f} shares"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def cash_vanna_display(self) -> str:
        """Cash vanna per 1% vol = (dDelta/dSigma) * n_lots * multiplier * S / 100."""
        try:
            ds = 0.001
            d_up = bs_greeks(self.S, self.K, self.T, self.r, self.q, self.repo,
                             self.sigma + ds, self.option_type)["delta"]
            d_dn = bs_greeks(self.S, self.K, self.T, self.r, self.q, self.repo,
                             max(0.001, self.sigma - ds), self.option_type)["delta"]
            vanna = (d_up - d_dn) / (2 * ds)
            cv = vanna * self.n_lots * self.multiplier * self.S / 100
            return f"{cv:,.2f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def rho_display(self) -> str:
        """Rho per 1% rate (unit, not cash)."""
        try:
            dr = 0.001
            p_up = bs_price(self.S, self.K, self.T, self.r + dr,
                            self.q, self.repo, self.sigma, self.option_type)
            p_dn = bs_price(self.S, self.K, self.T, self.r - dr,
                            self.q, self.repo, self.sigma, self.option_type)
            rho = (p_up - p_dn) / (2 * dr) / 100
            return f"{rho:.6f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def cash_rho_display(self) -> str:
        """Cash rho per 1% rate = (dPrice/dr) * n_lots * multiplier / 100."""
        try:
            dr = 0.001
            p_up = bs_price(self.S, self.K, self.T, self.r + dr,
                            self.q, self.repo, self.sigma, self.option_type)
            p_dn = bs_price(self.S, self.K, self.T, self.r - dr,
                            self.q, self.repo, self.sigma, self.option_type)
            rho = (p_up - p_dn) / (2 * dr)
            cr = rho * self.n_lots * self.multiplier / 100
            return f"{cr:,.2f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def delta_tx_display(self) -> str:
        """Change in share hedge per day = cash_charm / S = charm * n_lots * multiplier."""
        try:
            dt = 1.0 / 365.0
            T1 = max(0.001, self.T - dt)
            d_now = bs_greeks(self.S, self.K, self.T, self.r,
                              self.q, self.repo, self.sigma, self.option_type)["delta"]
            d_tm1 = bs_greeks(self.S, self.K, T1, self.r,
                              self.q, self.repo, self.sigma, self.option_type)["delta"]
            delta_tx = (d_tm1 - d_now) * self.n_lots * self.multiplier
            return f"{delta_tx:,.2f} shares/day"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def new_cash_delta_display(self) -> str:
        """New cash delta after spot move = cash_gamma * move_pct."""
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            cash_gamma = self.n_lots * g["gamma"] * self.multiplier * self.S ** 2 / 100
            new_cd = cash_gamma * self.spot_move_pct
            return f"{new_cd:,.0f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def gamma_pnl_display(self) -> str:
        """Gamma PnL = 1/2 × new_cash_delta × move%.
        Triangle: base = move%, height = new_cash_delta."""
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            cash_gamma = self.n_lots * g["gamma"] * self.multiplier * self.S ** 2 / 100
            new_cd = cash_gamma * self.spot_move_pct
            pnl = 0.5 * new_cd * (self.spot_move_pct / 100)
            return f"{pnl:,.2f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def daily_move_display(self) -> str:
        """Daily spot move proxy = IV / sqrt(252), in %."""
        try:
            dm = self.iv_for_move / math.sqrt(252)
            return f"{dm:.2f}%"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def breakeven_spot_display(self) -> str:
        """Breakeven spot at expiry: call → K + premium, put → K - premium."""
        try:
            p = bs_price(self.S, self.K, self.T, self.r,
                         self.q, self.repo, self.sigma, self.option_type)
            if self.option_type == "call":
                be = self.K + p
            else:
                be = self.K - p
            pct = (be / self.S - 1) * 100
            return f"{be:.2f}  ({pct:+.2f}%)"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def breakeven_vol_display(self) -> str:
        """Breakeven realized vol + daily spot range.
        daily_be_move = sqrt(2 * |theta| / gamma), annualize × sqrt(252).
        Then show the spot range [S - move, S + move]."""
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            gamma = g["gamma"]
            theta = g["theta"]
            if gamma <= 0:
                return "—"
            daily_move = math.sqrt(2 * abs(theta) / gamma)
            be_vol = (daily_move / self.S) * math.sqrt(252) * 100
            s_lo = self.S - daily_move
            s_hi = self.S + daily_move
            ticks = daily_move / self.tick_size if self.tick_size > 0 else 0
            return f"{be_vol:.2f}%  →  [{s_lo:.2f} , {s_hi:.2f}]  ({ticks:.0f} ticks)"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def gamma_theta_ratio_display(self) -> str:
        """Gamma/Theta ratio = cash_gamma / |cash_theta|."""
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            cash_gamma = self.n_lots * g["gamma"] * self.multiplier * self.S ** 2 / 100
            cash_theta = self.n_lots * g["theta"] * self.multiplier
            if abs(cash_theta) < 1e-10:
                return "—"
            ratio = cash_gamma / abs(cash_theta)
            return f"{ratio:.2f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def theta_earn_move_display(self) -> str:
        """Spot move (%) needed for gamma PnL to cover theta.
        solve: 1/2 * cash_gamma * m^2 / 100 = |cash_theta|
        → m = sqrt(200 * |cash_theta| / cash_gamma)."""
        try:
            g = bs_greeks(self.S, self.K, self.T, self.r,
                          self.q, self.repo, self.sigma, self.option_type)
            cash_gamma = self.n_lots * g["gamma"] * self.multiplier * self.S ** 2 / 100
            cash_theta = self.n_lots * g["theta"] * self.multiplier
            if cash_gamma <= 0:
                return "—"
            m = math.sqrt(200 * abs(cash_theta) / cash_gamma)
            return f"{m:.2f}%"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def qa_theta_bill_display(self) -> str:
        """Quick calc: Theta/day = Cash_Gamma(M) × σ² / 50,400 × 1,000,000."""
        try:
            theta = self.qa_gamma * self.qa_vol ** 2 / 50_400
            return f"${theta:,.0f} /day"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def qa_daily_move_display(self) -> str:
        """Daily move at the given vol = vol / sqrt(252)."""
        try:
            dm = self.qa_vol / math.sqrt(252)
            return f"{dm:.2f}%"
        except Exception:
            return "—"

    # ------------------------------------------------------------------
    # American pricing (binomial tree)
    # Both EU and AM prices come from the SAME tree so premium >= 0.
    # ------------------------------------------------------------------

    @rx.var(cache=True)
    def _american_result(self) -> dict[str, float]:
        try:
            b = self.r - self.q - self.repo
            return american_binomial_tree(
                self.S, self.K, self.T, self.r, b, self.sigma,
                self.bin_steps, self.option_type,
            )
        except Exception:
            return {"price": 0.0, "eu_price": 0.0, "delta": 0.0}

    @rx.var(cache=True)
    def american_price_display(self) -> str:
        try:
            return f"{self._american_result['price']:.4f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def american_eu_price_display(self) -> str:
        try:
            return f"{self._american_result['eu_price']:.4f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def american_delta_display(self) -> str:
        try:
            return f"{self._american_result['delta']:.6f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def exercise_premium_display(self) -> str:
        try:
            res = self._american_result
            prem = res["price"] - res["eu_price"]
            return f"{prem:.4f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def exercise_premium_pct_display(self) -> str:
        try:
            res = self._american_result
            eu = res["eu_price"]
            if eu < 0.0001:
                return "—"
            prem = res["price"] - eu
            return f"{prem / eu * 100:.2f}%"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def exercise_premium_positive(self) -> bool:
        try:
            res = self._american_result
            return (res["price"] - res["eu_price"]) > 0.0001
        except Exception:
            return False

    @rx.var(cache=True)
    def early_exercise_reason(self) -> str:
        ot = self.option_type.lower()
        if ot == "put":
            return "Deep ITM put: remaining time value is less than interest earned by exercising now and collecting K (K x r x dt > time value)."
        else:
            return "Call with dividend: the dividend lost by not holding the underlying exceeds the remaining time value of the option."

    # ------------------------------------------------------------------
    # Bond pricer computed vars
    # ------------------------------------------------------------------

    @rx.var(cache=True)
    def bond_face_ref(self) -> int:
        return int(round(self.bond_face))

    @rx.var(cache=True)
    def bond_result(self) -> dict[str, Any]:
        try:
            return bond_price_from_ytm(
                self.bond_face, self.bond_coupon, self.bond_maturity,
                self.bond_ytm, self.bond_freq,
            )
        except Exception:
            return {"dirty_price": 0.0, "clean_price": 0.0, "accrued_interest": 0.0,
                    "macaulay_duration": 0.0, "modified_duration": 0.0,
                    "convexity": 0.0, "cashflows": []}

    @rx.var(cache=True)
    def bond_dirty_price_display(self) -> str:
        try:
            return f"{self.bond_result['dirty_price']:.2f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def bond_clean_price_display(self) -> str:
        try:
            return f"{self.bond_result['clean_price']:.2f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def bond_accrued_display(self) -> str:
        try:
            return f"{self.bond_result['accrued_interest']:.4f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def bond_mac_dur_display(self) -> str:
        try:
            return f"{self.bond_result['macaulay_duration']:.4f} y"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def bond_mod_dur_display(self) -> str:
        try:
            return f"{self.bond_result['modified_duration']:.4f} y"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def bond_freq_str(self) -> str:
        return str(self.bond_freq)

    @rx.var(cache=True)
    def bond_convexity_display(self) -> str:
        try:
            return f"{self.bond_result['convexity']:.4f}"
        except Exception:
            return "—"

    @rx.var(cache=True)
    def bond_pv_chart_data(self) -> list[dict[str, Any]]:
        """PV waterfall: coupon PV and principal PV at each date."""
        try:
            cfs = self.bond_result["cashflows"]
            result = []
            for cf in cfs:
                is_last = cf["type"] == "coupon+principal"
                coupon_pv = cf["pv"] if not is_last else round(cf["pv"] - self.bond_face / (1 + self.bond_ytm / 100 / self.bond_freq) ** (cf["t"] * self.bond_freq), 2)
                principal_pv = round(cf["pv"] - coupon_pv, 2) if is_last else 0.0
                result.append({
                    "t": cf["t"],
                    "coupon_pv": round(coupon_pv, 2) if not is_last else round(coupon_pv, 2),
                    "principal_pv": principal_pv,
                })
            return result
        except Exception:
            return []

    @rx.var(cache=True)
    def bond_yield_curve_data(self) -> list[dict[str, Any]]:
        try:
            return bond_price_yield_curve(
                self.bond_face, self.bond_coupon, self.bond_maturity,
                self.bond_freq,
            )
        except Exception:
            return []

    @rx.var(cache=True)
    def vol_greeks_data(self) -> list[dict[str, Any]]:
        """All cash Greeks across a range of vols (for chart 2)."""
        try:
            n = 31
            vols = [self.vol_min + i * (self.vol_max - self.vol_min) / (n - 1) for i in range(n)]
            result = []
            for sigma in vols:
                sigma = max(0.001, sigma)
                g = bs_greeks(self.S, self.K, self.T, self.r, self.q, self.repo, sigma, self.option_type)
                p_call = bs_price(self.S, self.K, self.T, self.r, self.q, self.repo, sigma, "call")
                p_put = bs_price(self.S, self.K, self.T, self.r, self.q, self.repo, sigma, "put")
                cd = self.n_lots * g["delta"] * self.multiplier * self.S
                cg = self.n_lots * g["gamma"] * self.multiplier * self.S ** 2 / 100
                ct = self.n_lots * g["theta"] * self.multiplier
                cv = self.n_lots * g["vega"] * self.multiplier
                # charm (numerical)
                dt = 1.0 / 365.0
                T1 = max(0.001, self.T - dt)
                d_now = g["delta"]
                d_tm1 = bs_greeks(self.S, self.K, T1, self.r, self.q, self.repo, sigma, self.option_type)["delta"]
                cc = (d_tm1 - d_now) * self.n_lots * self.multiplier * self.S
                # vanna (numerical)
                ds = 0.001
                d_up = bs_greeks(self.S, self.K, self.T, self.r, self.q, self.repo, sigma + ds, self.option_type)["delta"]
                d_dn = bs_greeks(self.S, self.K, self.T, self.r, self.q, self.repo, max(0.001, sigma - ds), self.option_type)["delta"]
                cva = (d_up - d_dn) / (2 * ds) * self.n_lots * self.multiplier * self.S / 100
                # rho (numerical)
                dr = 0.001
                p_up = bs_price(self.S, self.K, self.T, self.r + dr, self.q, self.repo, sigma, self.option_type)
                p_dn = bs_price(self.S, self.K, self.T, self.r - dr, self.q, self.repo, sigma, self.option_type)
                cr = (p_up - p_dn) / (2 * dr) * self.n_lots * self.multiplier / 100
                result.append({
                    "vol": round(float(sigma * 100), 2),
                    "call_price": round(float(p_call), 4),
                    "put_price": round(float(p_put), 4),
                    "cash_delta": round(float(cd), 2),
                    "cash_gamma": round(float(cg), 2),
                    "cash_theta": round(float(ct), 2),
                    "cash_vega": round(float(cv), 2),
                    "cash_charm": round(float(cc), 2),
                    "cash_vanna": round(float(cva), 2),
                    "cash_rho": round(float(cr), 2),
                })
            return result
        except Exception:
            return []

    @rx.var(cache=True)
    def maturity_greeks_data(self) -> list[dict[str, Any]]:
        """All cash Greeks across a range of maturities (for chart 2)."""
        try:
            mats = [i / 12.0 for i in range(1, 37)]  # 1 month to 3 years
            result = []
            for T in mats:
                T = max(0.001, T)
                g = bs_greeks(self.S, self.K, T, self.r, self.q, self.repo, self.sigma, self.option_type)
                p_call = bs_price(self.S, self.K, T, self.r, self.q, self.repo, self.sigma, "call")
                p_put = bs_price(self.S, self.K, T, self.r, self.q, self.repo, self.sigma, "put")
                cd = self.n_lots * g["delta"] * self.multiplier * self.S
                cg = self.n_lots * g["gamma"] * self.multiplier * self.S ** 2 / 100
                ct = self.n_lots * g["theta"] * self.multiplier
                cv = self.n_lots * g["vega"] * self.multiplier
                # charm
                dt = 1.0 / 365.0
                T1 = max(0.001, T - dt)
                d_now = g["delta"]
                d_tm1 = bs_greeks(self.S, self.K, T1, self.r, self.q, self.repo, self.sigma, self.option_type)["delta"]
                cc = (d_tm1 - d_now) * self.n_lots * self.multiplier * self.S
                # vanna
                ds = 0.001
                d_up = bs_greeks(self.S, self.K, T, self.r, self.q, self.repo, self.sigma + ds, self.option_type)["delta"]
                d_dn = bs_greeks(self.S, self.K, T, self.r, self.q, self.repo, max(0.001, self.sigma - ds), self.option_type)["delta"]
                cva = (d_up - d_dn) / (2 * ds) * self.n_lots * self.multiplier * self.S / 100
                # rho
                dr = 0.001
                p_up = bs_price(self.S, self.K, T, self.r + dr, self.q, self.repo, self.sigma, self.option_type)
                p_dn = bs_price(self.S, self.K, T, self.r - dr, self.q, self.repo, self.sigma, self.option_type)
                cr = (p_up - p_dn) / (2 * dr) * self.n_lots * self.multiplier / 100
                result.append({
                    "maturity": round(float(T), 2),
                    "call_price": round(float(p_call), 4),
                    "put_price": round(float(p_put), 4),
                    "cash_delta": round(float(cd), 2),
                    "cash_gamma": round(float(cg), 2),
                    "cash_theta": round(float(ct), 2),
                    "cash_vega": round(float(cv), 2),
                    "cash_charm": round(float(cc), 2),
                    "cash_vanna": round(float(cva), 2),
                    "cash_rho": round(float(cr), 2),
                })
            return result
        except Exception:
            return []

    @rx.var(cache=True)
    def spot_ladder_data(self) -> list[dict[str, Any]]:
        try:
            rows = cash_delta_spot_ladder(
                self.S_min, self.S_max, self.n_spots,
                self.K, self.T, self.r, self.q, self.repo,
                self.sigma, self.n_lots, self.multiplier,
            )
            dt = 1.0 / 365.0
            T1 = max(0.001, self.T - dt)
            result = []
            for row in rows:
                S = row["spot"]
                cash_gamma = self.n_lots * row["gamma"] * self.multiplier * S ** 2 / 100
                cash_theta = self.n_lots * row["call_theta"] * self.multiplier
                cash_vega  = self.n_lots * row["vega"] * self.multiplier
                d_now = row["call_delta"]
                d_tm1 = bs_greeks(S, self.K, T1, self.r,
                                  self.q, self.repo, self.sigma, "call")["delta"]
                cash_charm = (d_tm1 - d_now) * self.n_lots * self.multiplier * S
                ds = 0.001
                d_up = bs_greeks(S, self.K, self.T, self.r, self.q, self.repo,
                                 self.sigma + ds, "call")["delta"]
                d_dn = bs_greeks(S, self.K, self.T, self.r, self.q, self.repo,
                                 max(0.001, self.sigma - ds), "call")["delta"]
                cash_vanna = (d_up - d_dn) / (2 * ds) * self.n_lots * self.multiplier * S / 100
                dr = 0.001
                p_up = bs_price(S, self.K, self.T, self.r + dr,
                                self.q, self.repo, self.sigma, "call")
                p_dn = bs_price(S, self.K, self.T, self.r - dr,
                                self.q, self.repo, self.sigma, "call")
                cash_rho = (p_up - p_dn) / (2 * dr) * self.n_lots * self.multiplier / 100
                r2 = {k: round(float(v), 4) if isinstance(v, float) else v
                      for k, v in row.items()}
                r2["cash_gamma"] = round(float(cash_gamma), 4)
                r2["cash_theta"] = round(float(cash_theta), 4)
                r2["cash_vega"]  = round(float(cash_vega),  4)
                r2["cash_charm"] = round(float(cash_charm), 4)
                r2["cash_vanna"] = round(float(cash_vanna), 4)
                r2["cash_rho"]   = round(float(cash_rho), 4)
                result.append(r2)
            return result
        except Exception:
            return []

    @rx.var(cache=True)
    def vol_ladder_data(self) -> list[dict[str, Any]]:
        try:
            rows = delta_vol_ladder(
                self.vol_min, self.vol_max, self.n_vols,
                self.S, self.K, self.T, self.r, self.q, self.repo,
                self.n_lots, self.multiplier, self.option_type,
            )
            return [
                {k: round(float(v), 4) if isinstance(v, float) else v
                 for k, v in row.items()}
                for row in rows
            ]
        except Exception:
            return []

    @rx.var(cache=True)
    def gamma_spot_data(self) -> list[dict[str, Any]]:
        try:
            rows = gamma_spot_ladder(
                self.S_min, self.S_max, self.n_spots,
                self.K, self.T, self.r, self.q, self.repo, self.sigma,
            )
            return [
                {k: round(float(v), 6) if isinstance(v, float) else v
                 for k, v in row.items()}
                for row in rows
            ]
        except Exception:
            return []

    @rx.var(cache=True)
    def gamma_vol_data(self) -> list[dict[str, Any]]:
        """Gamma vs vol for OTM (S*0.8), ATM (S), ITM (S*1.2).
        Keys renamed to fixed 'gamma_otm/atm/itm' for recharts compatibility."""
        try:
            s_otm = round(self.S * 0.8, 2)
            s_atm = round(self.S, 2)
            s_itm = round(self.S * 1.2, 2)
            rows = gamma_vol_ladder(
                self.vol_min, self.vol_max, self.n_vols,
                self.K, self.T, self.r, self.q, self.repo,
                spots=[s_otm, s_atm, s_itm],
            )
            result = []
            for row in rows:
                result.append({
                    "vol":        round(float(row["vol"]) * 100, 2),   # display as %
                    "gamma_otm":  round(float(row[f"gamma_S{int(s_otm)}"]), 6),
                    "gamma_atm":  round(float(row[f"gamma_S{int(s_atm)}"]), 6),
                    "gamma_itm":  round(float(row[f"gamma_S{int(s_itm)}"]), 6),
                })
            return result
        except Exception:
            return []

    @rx.var(cache=True)
    def forward_ladder_data(self) -> list[dict[str, Any]]:
        try:
            mats = [float(m.strip())
                    for m in self.fwd_maturities_str.split(",")
                    if m.strip()]
            rows = forward_ladder(self.S, self.r, self.q, self.repo, mats)
            return [
                {k: round(float(v), 4) if isinstance(v, float) else v
                 for k, v in row.items()}
                for row in rows
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def set_S(self, v: str):
        try: self.S = float(v)
        except ValueError: pass

    def set_K(self, v: str):
        try: self.K = float(v)
        except ValueError: pass

    def set_T(self, v: str):
        try: self.T = max(0.001, float(v))
        except ValueError: pass

    def set_maturity_date(self, v: str):
        try:
            target = datetime.date.fromisoformat(v)
            days = (target - datetime.date.today()).days
            self.T = max(0.001, days / 365.25)
            self.maturity_date = v
        except (ValueError, TypeError):
            pass

    def set_r(self, v: str):
        try: self.r = float(v) / 100
        except ValueError: pass

    def set_q(self, v: str):
        try: self.q = float(v) / 100
        except ValueError: pass

    def set_repo(self, v: str):
        try: self.repo = float(v) / 100
        except ValueError: pass

    def set_sigma(self, v: str):
        try: self.sigma = max(0.001, float(v) / 100)
        except ValueError: pass

    def set_n_lots(self, v: str):
        try: self.n_lots = float(v)
        except ValueError: pass

    def set_multiplier(self, v: str):
        try: self.multiplier = float(v)
        except ValueError: pass

    def set_option_type(self, v: str):
        if v in ("call", "put"):
            self.option_type = v

    def set_spot_range_pct(self, v: str):
        try: self.spot_range_pct = max(0.05, min(0.95, float(v)))
        except ValueError: pass

    def set_n_spots(self, v: str):
        try: self.n_spots = max(5, min(51, int(v)))
        except ValueError: pass

    def set_vol_min(self, v: str):
        try: self.vol_min = max(0.01, float(v))
        except ValueError: pass

    def set_vol_max(self, v: str):
        try: self.vol_max = min(2.0, float(v))
        except ValueError: pass

    def set_n_vols(self, v: str):
        try: self.n_vols = max(5, min(51, int(v)))
        except ValueError: pass

    def set_fwd_maturities_str(self, v: str):
        self.fwd_maturities_str = v

    def set_selected_chart(self, v: str):
        self.selected_chart = v

    def set_selected_chart2(self, v: str):
        self.selected_chart2 = v

    def set_selected_axis2(self, v: str):
        self.selected_axis2 = v

    def set_spot_move_pct(self, v: str):
        try: self.spot_move_pct = float(v)
        except ValueError: pass

    def set_iv_for_move(self, v: str):
        try: self.iv_for_move = float(v)
        except ValueError: pass

    def set_tick_size(self, v: str):
        try: self.tick_size = max(0.0001, float(v))
        except ValueError: pass

    def set_qa_gamma(self, v: str):
        try:
            cleaned = v.replace(",", "").replace(" ", "")
            self.qa_gamma = float(cleaned)
            self.qa_gamma_str = f"{self.qa_gamma:,.0f}"
        except ValueError:
            self.qa_gamma_str = v

    def set_qa_vol(self, v: str):
        try: self.qa_vol = float(v)
        except ValueError: pass

    def set_bin_steps(self, v: list):
        try: self.bin_steps = max(10, min(500, int(v[0])))
        except (ValueError, IndexError, TypeError): pass

    def set_bond_face(self, v: str):
        try: self.bond_face = max(1, float(v))
        except ValueError: pass

    def set_bond_coupon(self, v: str):
        try: self.bond_coupon = max(0, float(v))
        except ValueError: pass

    def set_bond_maturity(self, v: str):
        try: self.bond_maturity = max(0.5, float(v))
        except ValueError: pass

    def set_bond_ytm(self, v: str):
        try: self.bond_ytm = max(0.01, float(v))
        except ValueError: pass

    def set_bond_freq(self, v: str):
        if v in ("1", "2", "4"):
            self.bond_freq = int(v)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

LABEL_W = "90px"
INPUT_W = "110px"

# Excel-like palette — all explicit, immune to OS dark mode
_BG     = "white"
_CARD   = "#f5f5f5"
_BORDER = "#d4d4d4"
_TEXT   = "#1a1a1a"
_MUTED  = "#666666"

CARD_STYLE = {
    "background": _CARD,
    "border_radius": "4px",
    "border": f"1px solid {_BORDER}",
}


def param_row(label: str, value, on_change, step: str = "0.01") -> rx.Component:
    return rx.hstack(
        rx.text(label, width=LABEL_W, font_size="2", color=_TEXT),
        rx.input(
            value=value,
            on_change=on_change,
            type="number",
            step=step,
            width=INPUT_W,
            size="2",
        ),
        align="center",
        spacing="2",
    )


def metric_card(label: str, value) -> rx.Component:
    return rx.box(
        rx.text(label, font_size="1", color=_MUTED, margin_bottom="4px"),
        rx.text(value, font_size="5", font_weight="600", color=_TEXT),
        **CARD_STYLE,
        padding="1em",
        min_width="140px",
        text_align="center",
    )


# ---------------------------------------------------------------------------
# Tab 1 — Pricer
# ---------------------------------------------------------------------------

def inputs_panel() -> rx.Component:
    return rx.vstack(
        rx.heading("Parameters", size="4", margin_bottom="0.5em", color=_TEXT),
        param_row("S (spot)",   State.S,          State.set_S,     "1"),
        param_row("K (strike)", State.K,          State.set_K,     "1"),
        rx.hstack(
            rx.text("Maturity", width=LABEL_W, font_size="2", color=_TEXT),
            rx.vstack(
                rx.input(
                    value=State.maturity_date,
                    on_change=State.set_maturity_date,
                    type="date",
                    width=INPUT_W,
                    size="2",
                ),
                rx.text(State.T_years_str, font_size="1", color=_MUTED),
                spacing="0",
            ),
            align="center",
            spacing="2",
        ),
        param_row("r (%)",      State.r_pct,      State.set_r,     "0.1"),
        param_row("q (%)",      State.q_pct,      State.set_q,     "0.1"),
        param_row("repo (%)",   State.repo_pct,   State.set_repo,  "0.1"),
        param_row("σ (%)",      State.sigma_pct,  State.set_sigma, "0.5"),
        param_row("Lots",       State.n_lots,     State.set_n_lots),
        param_row("Multiplier", State.multiplier, State.set_multiplier, "1"),
        param_row("Tick size",  State.tick_size,  State.set_tick_size, "0.01"),
        rx.hstack(
            rx.text("Type", width=LABEL_W, font_size="2", color=_TEXT),
            rx.select(
                ["call", "put"],
                value=State.option_type,
                on_change=State.set_option_type,
                size="2",
                width=INPUT_W,
            ),
            align="center",
            spacing="2",
        ),
        spacing="2",
        padding="1em",
        **CARD_STYLE,
        width="260px",
    )


_TH = {"padding": "6px 16px", "text_align": "left", "font_weight": "600",
       "color": _TEXT, "background": "#e8e8e8", "border_bottom": f"1px solid {_BORDER}"}
_TD = {"padding": "4px 16px", "color": _TEXT, "border_bottom": f"1px solid {_BORDER}"}


def greeks_table() -> rx.Component:
    rows = [
        ("Delta",          State.delta_display),
        ("Gamma",          State.gamma_display),
        ("Vega (per 1%)",  State.vega_display),
        ("Theta (per day)",State.theta_display),
        ("Rho (per 1%)",   State.rho_display),
    ]
    return rx.el.table(
        rx.el.thead(
            rx.el.tr(
                rx.el.th("Greek", style=_TH),
                rx.el.th("Value", style={**_TH, "text_align": "right"}),
            ),
        ),
        rx.el.tbody(
            *[
                rx.el.tr(
                    rx.el.td(label, style=_TD),
                    rx.el.td(val,   style={**_TD, "text_align": "right"}),
                )
                for label, val in rows
            ]
        ),
        style={"width": "360px", "border_collapse": "collapse",
               "background": _CARD, "border": f"1px solid {_BORDER}",
               "border_radius": "4px", "font_size": "14px"},
    )


def _chart(title: str, lines: list, data_key_x: str, data, h: int = 240, w: int = 700, ref_x=None) -> rx.Component:
    """Helper: titled line chart. ref_x adds a dashed vertical reference line."""
    extras = []
    x_axis_props: dict = {"data_key": data_key_x, "stroke": "#bbb", "tick_line": False}
    if ref_x is not None:
        extras.append(rx.recharts.reference_line(
            x=ref_x, stroke="#999", stroke_dasharray="4 4", label="ATM",
        ))
        x_axis_props["type_"] = "number"
        x_axis_props["domain"] = ["dataMin", "dataMax"]
    return rx.vstack(
        rx.text(title, font_size="2", font_weight="600", color=_TEXT),
        rx.recharts.line_chart(
            *lines,
            *extras,
            rx.recharts.x_axis(**x_axis_props),
            rx.recharts.y_axis(stroke="#bbb", tick_line=False),
            rx.recharts.cartesian_grid(stroke="#e5e5e5", horizontal=True, vertical=False),
            rx.recharts.legend(vertical_align="top"),
            rx.recharts.graphing_tooltip(),
            data=data,
            width=w,
            height=h,
        ),
        spacing="1",
    )


def _chart2_vol(greek: str, data_key: str, color: str) -> rx.Component:
    """Chart 2 variant: Greek vs Vol."""
    return _chart(
        f"{greek} vs Vol",
        [rx.recharts.line(data_key=data_key, stroke=color, stroke_width=2, dot=False, name=greek)],
        "vol", State.vol_greeks_data, h=360, w=500,
    )


def _chart2_mat(greek: str, data_key: str, color: str) -> rx.Component:
    """Chart 2 variant: Greek vs Maturity."""
    return _chart(
        f"{greek} vs Maturity",
        [rx.recharts.line(data_key=data_key, stroke=color, stroke_width=2, dot=False, name=greek)],
        "maturity", State.maturity_greeks_data, h=360, w=500,
    )


_CHART2_MAP = {
    "Price":      ("call_price",  "#3182ce"),
    "Cash Delta": ("cash_delta",  "#38a169"),
    "Cash Gamma": ("cash_gamma",  "#805ad5"),
    "Cash Theta": ("cash_theta",  "#c05621"),
    "Cash Vega":  ("cash_vega",   "#2b6cb0"),
    "Cash Charm": ("cash_charm",  "#276749"),
    "Cash Vanna": ("cash_vanna",  "#97266d"),
    "Cash Rho":   ("cash_rho",    "#b83280"),
}


def chart2_box() -> rx.Component:
    """Second chart: Greek vs Vol or Maturity."""
    greeks = ["Price", "Cash Delta", "Cash Gamma", "Cash Theta",
              "Cash Vega", "Cash Charm", "Cash Vanna", "Cash Rho"]

    def _vol_match():
        return rx.match(
            State.selected_chart2,
            *[
                (name, _chart2_vol(name, dk, col))
                for name, (dk, col) in _CHART2_MAP.items()
                if name != "Price"
            ],
            # Default: Price (call + put)
            _chart(
                "Price vs Vol",
                [
                    rx.recharts.line(data_key="call_price", stroke="#3182ce", stroke_width=2, dot=False, name="Call"),
                    rx.recharts.line(data_key="put_price", stroke="#e53e3e", stroke_width=2, dot=False, name="Put"),
                ],
                "vol", State.vol_greeks_data, h=360, w=500,
            ),
        )

    def _mat_match():
        return rx.match(
            State.selected_chart2,
            *[
                (name, _chart2_mat(name, dk, col))
                for name, (dk, col) in _CHART2_MAP.items()
                if name != "Price"
            ],
            # Default: Price (call + put)
            _chart(
                "Price vs Maturity",
                [
                    rx.recharts.line(data_key="call_price", stroke="#3182ce", stroke_width=2, dot=False, name="Call"),
                    rx.recharts.line(data_key="put_price", stroke="#e53e3e", stroke_width=2, dot=False, name="Put"),
                ],
                "maturity", State.maturity_greeks_data, h=360, w=500,
            ),
        )

    return rx.box(
        rx.hstack(
            rx.text("Chart:", font_size="2", font_weight="600", color=_TEXT),
            rx.select(
                greeks,
                value=State.selected_chart2,
                on_change=State.set_selected_chart2,
                size="2",
                width="150px",
            ),
            rx.text("vs", font_size="2", color=_MUTED),
            rx.select(
                ["Vol", "Maturity"],
                value=State.selected_axis2,
                on_change=State.set_selected_axis2,
                size="2",
                width="120px",
            ),
            align="center",
            spacing="2",
            margin_bottom="0.5em",
        ),
        rx.match(
            State.selected_axis2,
            ("Maturity", _mat_match()),
            _vol_match(),
        ),
        background=_BG,
        border=f"1px solid {_BORDER}",
        border_radius="4px",
        padding="1em",
    )


def pricer_tab() -> rx.Component:
    return rx.hstack(
        # Left: inputs
        inputs_panel(),
        # Right: metrics + greeks + charts
        rx.vstack(
            # Hero metrics row
            rx.hstack(
                metric_card("BS Price",      State.bs_price_display),
                metric_card("Forward",       State.forward_price),
                metric_card("Cash Delta",    State.cash_delta_display),
                metric_card("Cash Gamma/1%", State.cash_gamma_display),
                metric_card("Cash Theta/day", State.cash_theta_display),
                metric_card("Cash Vega/1%",  State.cash_vega_display),
                metric_card("Cash Charm/day",State.cash_charm_display),
                metric_card("Cash Vanna/1%", State.cash_vanna_display),
                metric_card("Cash Rho/1%",  State.cash_rho_display),
                spacing="3",
                flex_wrap="wrap",
            ),
            rx.hstack(
                rx.text("Delta hedge: ", State.n_stocks_display, font_size="2", color=_MUTED),
                rx.text("  |  Delta t+1d: ", State.delta_tx_display, font_size="2", color=_MUTED),
                margin_top="0.25em",
                spacing="1",
            ),
            # Gamma PnL calculator
            rx.hstack(
                rx.text("Gamma PnL", font_size="2", font_weight="600", color=_TEXT),
                rx.text("  move:", font_size="2", color=_MUTED),
                rx.input(
                    value=State.spot_move_pct,
                    on_change=State.set_spot_move_pct,
                    type="number",
                    step="0.5",
                    width="70px",
                    size="1",
                ),
                rx.text("%", font_size="2", color=_MUTED),
                rx.text(" → new Δ cash: ", font_size="2", color=_MUTED),
                rx.text(State.new_cash_delta_display, font_size="2", font_weight="600", color=_TEXT),
                rx.text(" → PnL = ½ × ΔCash × move% = ", font_size="2", color=_MUTED),
                rx.text(State.gamma_pnl_display, font_size="2", font_weight="600", color="#2b6cb0"),
                rx.text("  |  IV:", font_size="2", color=_MUTED),
                rx.input(
                    value=State.iv_for_move,
                    on_change=State.set_iv_for_move,
                    type="number",
                    step="1",
                    width="65px",
                    size="1",
                ),
                rx.text("% → daily move ≈ ", font_size="2", color=_MUTED),
                rx.text(State.daily_move_display, font_size="2", font_weight="600", color="#2b6cb0"),
                align="center",
                spacing="1",
                padding="6px 12px",
                **CARD_STYLE,
            ),
            # Trading shortcuts
            rx.hstack(
                rx.vstack(
                    rx.text("BE Spot", font_size="1", color=_MUTED),
                    rx.text(State.breakeven_spot_display, font_size="2", font_weight="600", color=_TEXT),
                    spacing="0",
                ),
                rx.vstack(
                    rx.text("BE Vol (realized)", font_size="1", color=_MUTED),
                    rx.text(State.breakeven_vol_display, font_size="2", font_weight="600", color=_TEXT),
                    spacing="0",
                ),
                rx.vstack(
                    rx.text("Gamma/Theta", font_size="1", color=_MUTED),
                    rx.text(State.gamma_theta_ratio_display, font_size="2", font_weight="600", color=_TEXT),
                    spacing="0",
                ),
                rx.vstack(
                    rx.text("Theta earn move", font_size="1", color=_MUTED),
                    rx.text(State.theta_earn_move_display, font_size="2", font_weight="600", color=_TEXT),
                    spacing="0",
                ),
                spacing="6",
                padding="6px 12px",
                **CARD_STYLE,
            ),
            # Quick calc: gamma → theta bill
            rx.hstack(
                rx.text("Quick calc", font_size="2", font_weight="600", color=_TEXT),
                rx.text("  Gamma $", font_size="2", color=_MUTED),
                rx.input(
                    value=State.qa_gamma_str,
                    on_change=State.set_qa_gamma,
                    width="130px",
                    size="1",
                ),
                rx.text("@", font_size="2", color=_MUTED),
                rx.input(
                    value=State.qa_vol,
                    on_change=State.set_qa_vol,
                    type="number",
                    step="1",
                    width="60px",
                    size="1",
                ),
                rx.text("% vol", font_size="2", color=_MUTED),
                rx.text("  →  θ/day = CG × σ² / 50,400 = ", font_size="2", color=_MUTED),
                rx.text(State.qa_theta_bill_display, font_size="2", font_weight="600", color="#2b6cb0"),
                rx.text("  (daily move = ", font_size="2", color=_MUTED),
                rx.text(State.qa_daily_move_display, font_size="2", font_weight="600", color=_TEXT),
                rx.text(")", font_size="2", color=_MUTED),
                align="center",
                spacing="1",
                padding="6px 12px",
                **CARD_STYLE,
            ),
            # American vs European comparison
            rx.vstack(
                rx.text("Early Exercise Analysis (American Style)",
                        font_size="2", font_weight="600", color=_TEXT),
                rx.hstack(
                    rx.vstack(
                        rx.el.table(
                            rx.el.tbody(
                                rx.el.tr(
                                    rx.el.td("European Price (tree)", style=_TD),
                                    rx.el.td(State.american_eu_price_display, style={**_TD, "text_align": "right"}),
                                ),
                                rx.el.tr(
                                    rx.el.td("American Price (CRR)", style=_TD),
                                    rx.el.td(State.american_price_display, style={**_TD, "text_align": "right", "font_weight": "600"}),
                                ),
                                rx.el.tr(
                                    rx.el.td("Early Exercise Premium", style=_TD),
                                    rx.el.td(State.exercise_premium_display, style={**_TD, "text_align": "right", "color": "#2b6cb0"}),
                                ),
                                rx.el.tr(
                                    rx.el.td("Premium (%)", style=_TD),
                                    rx.el.td(State.exercise_premium_pct_display, style={**_TD, "text_align": "right", "color": "#2b6cb0"}),
                                ),
                                rx.el.tr(
                                    rx.el.td("American Delta", style=_TD),
                                    rx.el.td(State.american_delta_display, style={**_TD, "text_align": "right"}),
                                ),
                            ),
                            style={"border_collapse": "collapse",
                                   "background": _CARD, "border": f"1px solid {_BORDER}",
                                   "border_radius": "4px", "font_size": "14px"},
                        ),
                        rx.cond(
                            State.exercise_premium_positive,
                            rx.text(
                                State.early_exercise_reason,
                                font_size="1", color="#b45309",
                                padding="6px 10px",
                                background="#fef3c7",
                                border_radius="4px",
                                max_width="400px",
                            ),
                        ),
                        spacing="2",
                    ),
                    rx.vstack(
                        rx.text("Steps: ", State.bin_steps,
                                font_size="1", color=_MUTED),
                        rx.slider(
                            default_value=[100],
                            min=10,
                            max=500,
                            step=10,
                            on_value_commit=State.set_bin_steps,
                            width="200px",
                        ),
                        rx.text("10 ← convergence → 500",
                                font_size="1", color=_MUTED),
                        spacing="1",
                    ),
                    spacing="4",
                    align="start",
                ),
                spacing="2",
                padding="6px 12px",
                **CARD_STYLE,
            ),
            # Greeks table
            greeks_table(),
            # 2 Charts side by side below
            rx.hstack(
                # Chart 1: Greek vs Spot
                rx.box(
                    rx.hstack(
                        rx.text("vs Spot:", font_size="2", font_weight="600", color=_TEXT),
                        rx.select(
                            ["Price", "Cash Delta", "Cash Gamma", "Cash Theta",
                             "Cash Vega", "Cash Charm", "Cash Vanna", "Cash Rho"],
                            value=State.selected_chart,
                            on_change=State.set_selected_chart,
                            size="2",
                            width="150px",
                        ),
                        align="center",
                        spacing="2",
                        margin_bottom="0.5em",
                    ),
                    rx.match(
                        State.selected_chart,
                        ("Cash Delta", _chart(
                            "Cash Delta vs Spot",
                            [
                                rx.recharts.line(data_key="call_cash_delta", stroke="#38a169",
                                                 stroke_width=2, dot=False, name="Call Cash$"),
                                rx.recharts.line(data_key="put_cash_delta", stroke="#dd6b20",
                                                 stroke_width=2, dot=False, name="Put Cash$"),
                            ],
                            "spot", State.spot_ladder_data, h=360, w=500, ref_x=State.K_ref,
                        )),
                        ("Cash Gamma", _chart(
                            "Cash Gamma / 1% vs Spot",
                            [rx.recharts.line(data_key="cash_gamma", stroke="#805ad5",
                                              stroke_width=2, dot=False, name="Cash Gamma")],
                            "spot", State.spot_ladder_data, h=360, w=500, ref_x=State.K_ref,
                        )),
                        ("Cash Theta", _chart(
                            "Cash Theta / day vs Spot",
                            [rx.recharts.line(data_key="cash_theta", stroke="#c05621",
                                              stroke_width=2, dot=False, name="Cash Theta")],
                            "spot", State.spot_ladder_data, h=360, w=500, ref_x=State.K_ref,
                        )),
                        ("Cash Vega", _chart(
                            "Cash Vega / 1% vs Spot",
                            [rx.recharts.line(data_key="cash_vega", stroke="#2b6cb0",
                                              stroke_width=2, dot=False, name="Cash Vega")],
                            "spot", State.spot_ladder_data, h=360, w=500, ref_x=State.K_ref,
                        )),
                        ("Cash Charm", _chart(
                            "Cash Charm / day vs Spot",
                            [rx.recharts.line(data_key="cash_charm", stroke="#276749",
                                              stroke_width=2, dot=False, name="Cash Charm")],
                            "spot", State.spot_ladder_data, h=360, w=500, ref_x=State.K_ref,
                        )),
                        ("Cash Vanna", _chart(
                            "Cash Vanna / 1% vs Spot",
                            [rx.recharts.line(data_key="cash_vanna", stroke="#97266d",
                                              stroke_width=2, dot=False, name="Cash Vanna")],
                            "spot", State.spot_ladder_data, h=360, w=500, ref_x=State.K_ref,
                        )),
                        ("Cash Rho", _chart(
                            "Cash Rho / 1% vs Spot",
                            [rx.recharts.line(data_key="cash_rho", stroke="#b83280",
                                              stroke_width=2, dot=False, name="Cash Rho")],
                            "spot", State.spot_ladder_data, h=360, w=500, ref_x=State.K_ref,
                        )),
                        # Default: Price
                        _chart(
                            "Price vs Spot",
                            [
                                rx.recharts.line(data_key="call_price", stroke="#3182ce",
                                                 stroke_width=2, dot=False, name="Call"),
                                rx.recharts.line(data_key="put_price", stroke="#e53e3e",
                                                 stroke_width=2, dot=False, name="Put"),
                            ],
                            "spot", State.spot_ladder_data, h=360, w=500, ref_x=State.K_ref,
                        ),
                    ),
                    background=_BG,
                    border=f"1px solid {_BORDER}",
                    border_radius="4px",
                    padding="1em",
                ),
                # Chart 2: Greek vs Vol or Maturity
                chart2_box(),
                align="start",
                spacing="4",
            ),
            spacing="4",
            flex="1",
            padding="1em",
        ),
        align="start",
        spacing="4",
        width="100%",
        padding="1em",
    )


# ---------------------------------------------------------------------------
# Tab 2 — Ladders
# ---------------------------------------------------------------------------

def range_controls() -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.text("Spot range ±", font_size="2", color=_MUTED),
            rx.input(value=State.spot_range_pct, on_change=State.set_spot_range_pct,
                     type="number", step="0.05", width="80px", size="1"),
        ),
        rx.box(
            rx.text("n spots", font_size="2", color=_MUTED),
            rx.input(value=State.n_spots, on_change=State.set_n_spots,
                     type="number", step="2", width="70px", size="1"),
        ),
        rx.box(
            rx.text("Vol min", font_size="2", color=_MUTED),
            rx.input(value=State.vol_min, on_change=State.set_vol_min,
                     type="number", step="0.01", width="80px", size="1"),
        ),
        rx.box(
            rx.text("Vol max", font_size="2", color=_MUTED),
            rx.input(value=State.vol_max, on_change=State.set_vol_max,
                     type="number", step="0.05", width="80px", size="1"),
        ),
        rx.box(
            rx.text("n vols", font_size="2", color=_MUTED),
            rx.input(value=State.n_vols, on_change=State.set_n_vols,
                     type="number", step="2", width="70px", size="1"),
        ),
        spacing="4",
        align="end",
        flex_wrap="wrap",
        padding="0.5em",
        margin_bottom="1em",
        **CARD_STYLE,
    )


def spot_ladder_table() -> rx.Component:
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Spot"),
                rx.table.column_header_cell("Call"),
                rx.table.column_header_cell("Put"),
                rx.table.column_header_cell("C.Delta"),
                rx.table.column_header_cell("P.Delta"),
                rx.table.column_header_cell("Gamma"),
                rx.table.column_header_cell("Vega"),
                rx.table.column_header_cell("C.Cash$"),
                rx.table.column_header_cell("P.Cash$"),
            )
        ),
        rx.table.body(
            rx.foreach(
                State.spot_ladder_data,
                lambda row: rx.table.row(
                    rx.table.cell(row["spot"]),
                    rx.table.cell(row["call_price"]),
                    rx.table.cell(row["put_price"]),
                    rx.table.cell(row["call_delta"]),
                    rx.table.cell(row["put_delta"]),
                    rx.table.cell(row["gamma"]),
                    rx.table.cell(row["vega"]),
                    rx.table.cell(row["call_cash_delta"]),
                    rx.table.cell(row["put_cash_delta"]),
                ),
            )
        ),
        size="1",
        variant="surface",
    )


def vol_ladder_table() -> rx.Component:
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Vol"),
                rx.table.column_header_cell("Price"),
                rx.table.column_header_cell("Delta"),
                rx.table.column_header_cell("Cash Delta"),
            )
        ),
        rx.table.body(
            rx.foreach(
                State.vol_ladder_data,
                lambda row: rx.table.row(
                    rx.table.cell(row["vol"]),
                    rx.table.cell(row["price"]),
                    rx.table.cell(row["delta"]),
                    rx.table.cell(row["cash_delta"]),
                ),
            )
        ),
        size="1",
        variant="surface",
    )


def ladders_tab() -> rx.Component:
    return rx.vstack(
        range_controls(),
        # Spot ladder
        rx.heading("Spot Ladder", size="3"),
        rx.hstack(
            rx.box(spot_ladder_table(), overflow_x="auto", max_width="640px"),
            rx.vstack(
                rx.recharts.line_chart(
                    rx.recharts.line(data_key="call_delta", stroke="#3182ce",
                                     stroke_width=2, dot=False, name="Call Delta"),
                    rx.recharts.line(data_key="put_delta",  stroke="#e53e3e",
                                     stroke_width=2, dot=False, name="Put Delta"),
                    rx.recharts.x_axis(data_key="spot", stroke="#bbb", tick_line=False),
                    rx.recharts.y_axis(stroke="#bbb", tick_line=False),
                    rx.recharts.cartesian_grid(stroke="#e5e5e5", horizontal=True, vertical=False),
                    rx.recharts.legend(vertical_align="top"),
                    rx.recharts.graphing_tooltip(),
                    data=State.spot_ladder_data,
                    width=480,
                    height=220,
                ),
                rx.recharts.line_chart(
                    rx.recharts.line(data_key="call_cash_delta", stroke="#38a169",
                                     stroke_width=2, dot=False, name="Call Cash$"),
                    rx.recharts.line(data_key="put_cash_delta",  stroke="#dd6b20",
                                     stroke_width=2, dot=False, name="Put Cash$"),
                    rx.recharts.x_axis(data_key="spot", stroke="#bbb", tick_line=False),
                    rx.recharts.y_axis(stroke="#bbb", tick_line=False),
                    rx.recharts.cartesian_grid(stroke="#e5e5e5", horizontal=True, vertical=False),
                    rx.recharts.legend(vertical_align="top"),
                    rx.recharts.graphing_tooltip(),
                    data=State.spot_ladder_data,
                    width=480,
                    height=220,
                ),
            ),
            align="start",
            spacing="4",
        ),
        # Vol ladder
        rx.heading("Vol Ladder", size="3", margin_top="1em"),
        rx.hstack(
            rx.box(vol_ladder_table(), overflow_x="auto", max_width="360px"),
            rx.recharts.line_chart(
                rx.recharts.line(data_key="price",      stroke="#3182ce",
                                 stroke_width=2, dot=False, name="Price"),
                rx.recharts.line(data_key="delta",      stroke="#805ad5",
                                 stroke_width=2, dot=False, name="Delta"),
                rx.recharts.x_axis(data_key="vol", stroke="#bbb", tick_line=False),
                rx.recharts.y_axis(stroke="#bbb", tick_line=False),
                rx.recharts.cartesian_grid(stroke="#e5e5e5", horizontal=True, vertical=False),
                rx.recharts.legend(vertical_align="top"),
                rx.recharts.graphing_tooltip(),
                data=State.vol_ladder_data,
                width=480,
                height=300,
            ),
            align="start",
            spacing="4",
        ),
        spacing="3",
        padding="1em",
        width="100%",
    )


# ---------------------------------------------------------------------------
# Tab 3 — Gamma
# ---------------------------------------------------------------------------

def gamma_spot_table() -> rx.Component:
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Spot"),
                rx.table.column_header_cell("Gamma"),
                rx.table.column_header_cell("Call Delta"),
                rx.table.column_header_cell("Put Delta"),
                rx.table.column_header_cell("Call"),
                rx.table.column_header_cell("Put"),
            )
        ),
        rx.table.body(
            rx.foreach(
                State.gamma_spot_data,
                lambda row: rx.table.row(
                    rx.table.cell(row["spot"]),
                    rx.table.cell(row["gamma"]),
                    rx.table.cell(row["call_delta"]),
                    rx.table.cell(row["put_delta"]),
                    rx.table.cell(row["call_price"]),
                    rx.table.cell(row["put_price"]),
                ),
            )
        ),
        size="1",
        variant="surface",
    )


def gamma_tab() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.vstack(
                rx.heading("Gamma vs Spot (bell curve)", size="3"),
                rx.recharts.line_chart(
                    rx.recharts.line(data_key="gamma", stroke="#e67e22",
                                     stroke_width=2, dot=False, name="Gamma"),
                    rx.recharts.x_axis(data_key="spot", stroke="#bbb", tick_line=False),
                    rx.recharts.y_axis(stroke="#bbb", tick_line=False),
                    rx.recharts.cartesian_grid(stroke="#e5e5e5", horizontal=True, vertical=False),
                    rx.recharts.graphing_tooltip(),
                    data=State.gamma_spot_data,
                    width=520,
                    height=300,
                ),
            ),
            rx.vstack(
                rx.heading("Gamma vs Vol — OTM / ATM / ITM", size="3"),
                rx.text(
                    "OTM = S×0.8 | ATM = S | ITM = S×1.2  (clés fixes)",
                    font_size="1",
                    color=_MUTED,
                ),
                rx.recharts.line_chart(
                    rx.recharts.line(data_key="gamma_otm", stroke="#e53e3e",
                                     stroke_width=2, dot=False, name="OTM (S×0.8)"),
                    rx.recharts.line(data_key="gamma_atm", stroke="#3182ce",
                                     stroke_width=2, dot=False, name="ATM (S)"),
                    rx.recharts.line(data_key="gamma_itm", stroke="#38a169",
                                     stroke_width=2, dot=False, name="ITM (S×1.2)"),
                    rx.recharts.x_axis(data_key="vol", stroke="#bbb", tick_line=False,
                                       label={"value": "Vol %", "position": "insideBottom", "offset": -4}),
                    rx.recharts.y_axis(stroke="#bbb", tick_line=False),
                    rx.recharts.cartesian_grid(stroke="#e5e5e5", horizontal=True, vertical=False),
                    rx.recharts.legend(vertical_align="top"),
                    rx.recharts.graphing_tooltip(),
                    data=State.gamma_vol_data,
                    width=520,
                    height=300,
                ),
            ),
            spacing="6",
            align="start",
        ),
        rx.heading("Gamma Spot Table", size="3", margin_top="1em"),
        rx.box(gamma_spot_table(), overflow_x="auto"),
        spacing="3",
        padding="1em",
        width="100%",
    )


# ---------------------------------------------------------------------------
# Tab 2 — Bond Pricer
# ---------------------------------------------------------------------------

def bond_inputs_panel() -> rx.Component:
    return rx.vstack(
        rx.heading("Bond Parameters", size="4", margin_bottom="0.5em", color=_TEXT),
        param_row("Face value", State.bond_face, State.set_bond_face, "100"),
        param_row("Coupon (%)", State.bond_coupon, State.set_bond_coupon, "0.25"),
        param_row("Maturity (y)", State.bond_maturity, State.set_bond_maturity, "1"),
        param_row("YTM (%)", State.bond_ytm, State.set_bond_ytm, "0.25"),
        rx.hstack(
            rx.text("Frequency", width=LABEL_W, font_size="2", color=_TEXT),
            rx.select(
                ["1", "2", "4"],
                value=State.bond_freq_str,
                on_change=State.set_bond_freq,
                size="2",
                width=INPUT_W,
            ),
            align="center",
            spacing="2",
        ),
        spacing="2",
        padding="1em",
        **CARD_STYLE,
        width="260px",
    )


def bond_tab() -> rx.Component:
    return rx.hstack(
        # Left: inputs
        bond_inputs_panel(),
        # Right: metrics + charts
        rx.vstack(
            # Metrics row
            rx.hstack(
                metric_card("Dirty Price", State.bond_dirty_price_display),
                metric_card("Clean Price", State.bond_clean_price_display),
                metric_card("Accrued Interest", State.bond_accrued_display),
                metric_card("Macaulay Duration", State.bond_mac_dur_display),
                metric_card("Modified Duration", State.bond_mod_dur_display),
                metric_card("Convexity", State.bond_convexity_display),
                spacing="3",
                flex_wrap="wrap",
            ),
            # Two charts side by side
            rx.hstack(
                # Chart 1: Price-Yield relationship
                rx.box(
                    rx.vstack(
                        rx.text("Bond Price vs Yield", font_size="2", font_weight="600", color=_TEXT),
                        rx.recharts.line_chart(
                            rx.recharts.line(data_key="price", stroke="#3182ce",
                                             stroke_width=2, dot=False, name="Price"),
                            rx.recharts.x_axis(data_key="ytm", stroke="#bbb", tick_line=False,
                                               label={"value": "YTM (%)", "position": "insideBottom", "offset": -4}),
                            rx.recharts.y_axis(stroke="#bbb", tick_line=False),
                            rx.recharts.cartesian_grid(stroke="#e5e5e5", horizontal=True, vertical=False),
                            rx.recharts.reference_line(y=State.bond_face_ref, stroke="#999",
                                                       stroke_dasharray="4 4", label="Par"),
                            rx.recharts.graphing_tooltip(),
                            data=State.bond_yield_curve_data,
                            width=500,
                            height=360,
                        ),
                        spacing="1",
                    ),
                    background=_BG,
                    border=f"1px solid {_BORDER}",
                    border_radius="4px",
                    padding="1em",
                ),
                # Chart 2: PV waterfall (stacked bar)
                rx.box(
                    rx.vstack(
                        rx.text("Present Value of Cash Flows", font_size="2", font_weight="600", color=_TEXT),
                        rx.recharts.bar_chart(
                            rx.recharts.bar(data_key="coupon_pv", fill="#3182ce",
                                            stack_id="pv", name="Coupon PV"),
                            rx.recharts.bar(data_key="principal_pv", fill="#e53e3e",
                                            stack_id="pv", name="Principal PV"),
                            rx.recharts.x_axis(data_key="t", stroke="#bbb", tick_line=False,
                                               label={"value": "Maturity (y)", "position": "insideBottom", "offset": -4}),
                            rx.recharts.y_axis(stroke="#bbb", tick_line=False),
                            rx.recharts.cartesian_grid(stroke="#e5e5e5", horizontal=True, vertical=False),
                            rx.recharts.legend(vertical_align="top"),
                            rx.recharts.graphing_tooltip(),
                            data=State.bond_pv_chart_data,
                            width=500,
                            height=360,
                        ),
                        spacing="1",
                    ),
                    background=_BG,
                    border=f"1px solid {_BORDER}",
                    border_radius="4px",
                    padding="1em",
                ),
                align="start",
                spacing="4",
            ),
            spacing="4",
            flex="1",
            padding="1em",
        ),
        align="start",
        spacing="4",
        width="100%",
        padding="1em",
    )


# ---------------------------------------------------------------------------
# Root page
# ---------------------------------------------------------------------------

def index() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.heading("Skema Pricer", size="6", color=_TEXT),
            rx.badge("Options & Bonds", color_scheme="blue", variant="soft"),
            align="center",
            spacing="3",
            margin_bottom="1em",
        ),
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("Options",  value="pricer",  style={"color": _TEXT}),
                rx.tabs.trigger("Bonds",    value="bonds",   style={"color": _TEXT}),
            ),
            rx.tabs.content(pricer_tab(),   value="pricer"),
            rx.tabs.content(bond_tab(),     value="bonds"),
            default_value="pricer",
            width="100%",
        ),
        padding="2em",
        max_width="1800px",
        margin="0 auto",
        background=_BG,
        color=_TEXT,
        min_height="100vh",
    )


app = rx.App(
    theme=rx.theme(appearance="light", accent_color="blue"),
    style={
        "background_color": _BG,
        "color": _TEXT,
        "font_family": "Segoe UI, Calibri, Arial, sans-serif",
    },
)
app.add_page(index, title="Skema Options Pricer")
