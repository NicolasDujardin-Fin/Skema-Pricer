"""
pricer_app.py — Reflex GUI for the Skema Options Pricer.

4 tabs:
  1. Pricer  — BS price, Greeks, cash delta/gamma, price vs spot chart
  2. Ladders — Spot ladder + Vol ladder (tables + charts)
  3. Gamma   — Gamma bell (vs spot) + Gamma vs Vol (ATM/OTM/ITM)
  4. Forward — Forward pricing, basis, maturity ladder
"""

import os
import sys

# Make bs_engine and rates_engine importable from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Any

import reflex as rx

from bs_engine import (
    bs_greeks,
    bs_price,
    cash_delta_spot_ladder,
    delta_vol_ladder,
    gamma_spot_ladder,
    gamma_vol_ladder,
)
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
    r: float = 0.05
    q: float = 0.02
    repo: float = 0.0
    sigma: float = 0.20
    n_lots: float = 1.0
    multiplier: float = 100.0
    option_type: str = "call"

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
    def S_min(self) -> float:
        return round(self.S * (1.0 - self.spot_range_pct), 4)

    @rx.var(cache=True)
    def S_max(self) -> float:
        return round(self.S * (1.0 + self.spot_range_pct), 4)

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
    def spot_ladder_data(self) -> list[dict[str, Any]]:
        try:
            rows = cash_delta_spot_ladder(
                self.S_min, self.S_max, self.n_spots,
                self.K, self.T, self.r, self.q, self.repo,
                self.sigma, self.n_lots, self.multiplier,
            )
            return [
                {k: round(float(v), 4) if isinstance(v, float) else v
                 for k, v in row.items()}
                for row in rows
            ]
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

    def set_r(self, v: str):
        try: self.r = float(v)
        except ValueError: pass

    def set_q(self, v: str):
        try: self.q = float(v)
        except ValueError: pass

    def set_repo(self, v: str):
        try: self.repo = float(v)
        except ValueError: pass

    def set_sigma(self, v: str):
        try: self.sigma = max(0.001, float(v))
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


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

LABEL_W = "90px"
INPUT_W = "110px"
CARD_STYLE = {
    "background": "var(--gray-2)",
    "border_radius": "8px",
    "border": "1px solid var(--gray-5)",
}


def param_row(label: str, value, on_change, step: str = "0.01") -> rx.Component:
    return rx.hstack(
        rx.text(label, width=LABEL_W, font_size="2", color="var(--gray-11)"),
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
        rx.text(label, font_size="1", color="var(--gray-10)", margin_bottom="4px"),
        rx.text(value, font_size="5", font_weight="600"),
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
        rx.heading("Parameters", size="4", margin_bottom="0.5em"),
        param_row("S (spot)",   State.S,          State.set_S,     "1"),
        param_row("K (strike)", State.K,          State.set_K,     "1"),
        param_row("T (years)",  State.T,          State.set_T,     "0.1"),
        param_row("r (rate)",   State.r,          State.set_r,     "0.001"),
        param_row("q (div)",    State.q,          State.set_q,     "0.001"),
        param_row("repo",       State.repo,       State.set_repo,  "0.001"),
        param_row("σ (vol)",    State.sigma,      State.set_sigma, "0.01"),
        param_row("Lots",       State.n_lots,     State.set_n_lots),
        param_row("Multiplier", State.multiplier, State.set_multiplier, "1"),
        rx.hstack(
            rx.text("Type", width=LABEL_W, font_size="2", color="var(--gray-11)"),
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


def greeks_table() -> rx.Component:
    rows = [
        ("Delta",          State.delta_display),
        ("Gamma",          State.gamma_display),
        ("Vega (per 1%)",  State.vega_display),
        ("Theta (per day)",State.theta_display),
    ]
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Greek"),
                rx.table.column_header_cell("Value"),
            )
        ),
        rx.table.body(
            *[
                rx.table.row(
                    rx.table.cell(label),
                    rx.table.cell(val),
                )
                for label, val in rows
            ]
        ),
        size="2",
        variant="surface",
        width="100%",
    )


def pricer_tab() -> rx.Component:
    return rx.hstack(
        # Left: inputs
        inputs_panel(),
        # Right: results
        rx.vstack(
            # Hero metrics
            rx.hstack(
                metric_card("BS Price",       State.bs_price_display),
                metric_card("Forward",        State.forward_price),
                metric_card("Cash Delta",     State.cash_delta_display),
                metric_card("Cash Gamma/1%",  State.cash_gamma_display),
                spacing="3",
                flex_wrap="wrap",
            ),
            # Greeks table
            greeks_table(),
            # Call + Put price vs spot chart
            rx.heading("Call & Put Price vs Spot", size="3", margin_top="1em"),
            rx.recharts.line_chart(
                rx.recharts.line(data_key="call_price", stroke="#3182ce",
                                 stroke_width=2, dot=False, name="Call"),
                rx.recharts.line(data_key="put_price",  stroke="#e53e3e",
                                 stroke_width=2, dot=False, name="Put"),
                rx.recharts.x_axis(data_key="spot", label={"value": "Spot", "position": "insideBottom", "offset": -4}),
                rx.recharts.y_axis(),
                rx.recharts.cartesian_grid(stroke_dasharray="3 3", opacity=0.4),
                rx.recharts.legend(vertical_align="top"),
                rx.recharts.graphing_tooltip(),
                data=State.spot_ladder_data,
                width=620,
                height=280,
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
            rx.text("Spot range ±", font_size="2", color="var(--gray-10)"),
            rx.input(value=State.spot_range_pct, on_change=State.set_spot_range_pct,
                     type="number", step="0.05", width="80px", size="1"),
        ),
        rx.box(
            rx.text("n spots", font_size="2", color="var(--gray-10)"),
            rx.input(value=State.n_spots, on_change=State.set_n_spots,
                     type="number", step="2", width="70px", size="1"),
        ),
        rx.box(
            rx.text("Vol min", font_size="2", color="var(--gray-10)"),
            rx.input(value=State.vol_min, on_change=State.set_vol_min,
                     type="number", step="0.01", width="80px", size="1"),
        ),
        rx.box(
            rx.text("Vol max", font_size="2", color="var(--gray-10)"),
            rx.input(value=State.vol_max, on_change=State.set_vol_max,
                     type="number", step="0.05", width="80px", size="1"),
        ),
        rx.box(
            rx.text("n vols", font_size="2", color="var(--gray-10)"),
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
                    rx.recharts.x_axis(data_key="spot"),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(stroke_dasharray="3 3", opacity=0.4),
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
                    rx.recharts.x_axis(data_key="spot"),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(stroke_dasharray="3 3", opacity=0.4),
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
                rx.recharts.x_axis(data_key="vol"),
                rx.recharts.y_axis(),
                rx.recharts.cartesian_grid(stroke_dasharray="3 3", opacity=0.4),
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
                    rx.recharts.x_axis(data_key="spot"),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(stroke_dasharray="3 3", opacity=0.4),
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
                    color="var(--gray-10)",
                ),
                rx.recharts.line_chart(
                    rx.recharts.line(data_key="gamma_otm", stroke="#e53e3e",
                                     stroke_width=2, dot=False, name="OTM (S×0.8)"),
                    rx.recharts.line(data_key="gamma_atm", stroke="#3182ce",
                                     stroke_width=2, dot=False, name="ATM (S)"),
                    rx.recharts.line(data_key="gamma_itm", stroke="#38a169",
                                     stroke_width=2, dot=False, name="ITM (S×1.2)"),
                    rx.recharts.x_axis(data_key="vol",
                                       label={"value": "Vol %", "position": "insideBottom", "offset": -4}),
                    rx.recharts.y_axis(),
                    rx.recharts.cartesian_grid(stroke_dasharray="3 3", opacity=0.4),
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
# Tab 4 — Forward
# ---------------------------------------------------------------------------

def forward_table() -> rx.Component:
    return rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Maturity (y)"),
                rx.table.column_header_cell("Forward"),
                rx.table.column_header_cell("Basis (F-S)"),
                rx.table.column_header_cell("Basis %"),
            )
        ),
        rx.table.body(
            rx.foreach(
                State.forward_ladder_data,
                lambda row: rx.table.row(
                    rx.table.cell(row["maturity"]),
                    rx.table.cell(row["forward"]),
                    rx.table.cell(row["basis"]),
                    rx.table.cell(row["basis_pct"]),
                ),
            )
        ),
        size="2",
        variant="surface",
        width="380px",
    )


def forward_tab() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.text("Maturities (comma-separated years):", font_size="2"),
            rx.input(
                value=State.fwd_maturities_str,
                on_change=State.set_fwd_maturities_str,
                width="320px",
                size="2",
            ),
            align="center",
            spacing="3",
            margin_bottom="1em",
        ),
        rx.hstack(
            forward_table(),
            rx.recharts.line_chart(
                rx.recharts.line(data_key="forward", stroke="#3182ce",
                                 stroke_width=2, dot=True, name="Forward Price"),
                rx.recharts.x_axis(data_key="maturity",
                                   label={"value": "Maturity (y)", "position": "insideBottom", "offset": -4}),
                rx.recharts.y_axis(),
                rx.recharts.cartesian_grid(stroke_dasharray="3 3", opacity=0.4),
                rx.recharts.graphing_tooltip(),
                data=State.forward_ladder_data,
                width=480,
                height=280,
            ),
            spacing="4",
            align="start",
        ),
        rx.heading("Basis % by Maturity", size="3", margin_top="1em"),
        rx.recharts.bar_chart(
            rx.recharts.bar(data_key="basis_pct", fill="#805ad5", name="Basis %"),
            rx.recharts.x_axis(data_key="maturity"),
            rx.recharts.y_axis(),
            rx.recharts.cartesian_grid(stroke_dasharray="3 3", opacity=0.4),
            rx.recharts.reference_line(y=0, stroke="#aaa"),
            rx.recharts.graphing_tooltip(),
            data=State.forward_ladder_data,
            width=620,
            height=220,
        ),
        spacing="3",
        padding="1em",
        width="100%",
    )


# ---------------------------------------------------------------------------
# Root page
# ---------------------------------------------------------------------------

def index() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.heading("Skema Options Pricer", size="6"),
            rx.badge("BS with repo", color_scheme="blue", variant="soft"),
            align="center",
            spacing="3",
            margin_bottom="1em",
        ),
        rx.tabs.root(
            rx.tabs.list(
                rx.tabs.trigger("Pricer",   value="pricer"),
                rx.tabs.trigger("Ladders",  value="ladders"),
                rx.tabs.trigger("Gamma",    value="gamma"),
                rx.tabs.trigger("Forward",  value="forward"),
            ),
            rx.tabs.content(pricer_tab(),   value="pricer"),
            rx.tabs.content(ladders_tab(),  value="ladders"),
            rx.tabs.content(gamma_tab(),    value="gamma"),
            rx.tabs.content(forward_tab(),  value="forward"),
            default_value="pricer",
            width="100%",
        ),
        padding="2em",
        max_width="1440px",
        margin="0 auto",
    )


app = rx.App(
    theme=rx.theme(appearance="light", accent_color="blue"),
)
app.add_page(index, title="Skema Options Pricer")
