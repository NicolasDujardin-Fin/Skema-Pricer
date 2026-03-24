# Skema Pricer

Interactive derivatives & fixed-income pricer built with Streamlit.

## Tabs

### Options
Black-Scholes pricing with cost-of-carry (repo). All unit and cash Greeks (delta, gamma, vega, theta, charm, vanna, rho). Gamma PnL calculator, breakeven spot/vol, theta earn move. American vs European comparison via CRR binomial tree. Six sensitivity charts showing raw delta/gamma/vega across spot (by volatility) and time (by moneyness).

### Bonds
Dirty/clean price, accrued interest, Macaulay/modified duration, convexity. DV01, PV01 on notional, PnL with convexity adjustment. Callable bond pricing via lognormal binomial rate tree — option value, YTC, YTW. Interactive SVG tree visualization. Price-yield and PV waterfall charts.

### Turbo Pricer
Turbo Open-End (Long/Short) pricing. Two input modes: manual strike/barrier or target leverage (K and B auto-computed). Payoff chart with knock-out barrier. Financing cost simulation showing strike drift over time. Sensitivity table across spot scenarios.

## Project structure

```
app.py               Streamlit UI
bs_engine.py         Black-Scholes pricing & Greeks
bond_engine.py       Bond analytics, callable pricing, rate tree
numerical_engine.py  American option pricing (CRR binomial)
rates_engine.py      Forward pricing
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key formulas

- **BS cost of carry**: `b = r - q - repo`
- **Cash delta**: `n_lots * delta * multiplier * S`
- **Cash gamma/1%**: `n_lots * gamma * multiplier * S^2 / 100`
- **DV01**: `modified_duration * dirty_price * 0.0001`
- **PnL (bond)**: `(-ModDur * dy + 0.5 * Convexity * dy^2) * Price * n`
- **Callable bond**: lognormal binomial rate tree, issuer calls when continuation > call_price
- **Turbo price**: `(S - K) / parity` (Long), leverage = `S / (price * parity)`
- **American premium**: both EU and AM prices from same CRR tree (premium >= 0)
