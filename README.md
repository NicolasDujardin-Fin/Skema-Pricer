# Skema Pricer

Derivatives & fixed-income pricer built with Streamlit and Plotly.

**Live app:** https://skema-pricer-uvvqujpqjqepnjdhfbi45c.streamlit.app/

## Tabs

**Options** — Black-Scholes with cost-of-carry (repo). Unit and cash Greeks. Gamma PnL calculator, breakeven spot/vol, theta earn move. American vs European (CRR tree). 6 sensitivity charts (delta/gamma/vega vs spot by vol, vs time by moneyness). Interview Q&A (forwards, convexity, gamma P&L).

**Bonds** — Dirty/clean price, accrued interest, duration, convexity. DV01, PV01, PnL with convexity adjustment. Callable bond pricing (lognormal binomial rate tree), YTC, YTW. SVG tree visualization. Price-yield and PV waterfall charts. Interview Q&A (swaps, duration, convexity).

**Turbo** — Turbo Open-End (Long/Short). Manual strike/barrier or target leverage mode. Payoff chart, financing cost simulation (strike drift), sensitivity table. Interview Q&A (hedge sizing, leverage vs price, barrier distance, daily funding, residual value, KO unwind).

**Discount Certificate** — Replication: Long S − Call(K=Cap). Payoff and P&L charts. Sensitivity vs vol and cap level. Trading Q&A (client vs trader perspective).

**Bonus Certificate** — Replication: Long S + Put Down-and-Out(K=Bonus, B=Barrier) − Call(K=Cap). Reiner-Rubinstein barrier option pricing. Separate put/call vol (skew). Gap-style payoff chart. Sensitivity vs vol and time. Trading Q&A (gamma/vega sign flips, PDI vs PDO hedging, barrier pin risk).

**Interview** — Standalone Q&A tab. Greeks book-reading (delta from gamma, short gamma curse, vega P&L, gamma P&L formula). Delta hedging, pin risk at expiry, early exercise for dividends, volatility drag, IV→0/∞ gamma limits, skew position, rho.

## Structure

```
app.py                Streamlit UI (all tabs)
bs_engine.py          Black-Scholes pricing & Greeks
bond_engine.py        Bond analytics, callable pricing, rate tree
numerical_engine.py   American option pricing (CRR binomial)
rates_engine.py       Forward pricing
discount_engine.py    Discount Certificate (BS call replication)
bonus_engine.py       Bonus Certificate (Reiner-Rubinstein D&O put)
turbo_engine.py       Turbo Certificate pricing & financing
questions.json        All interview Q&A content
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key formulas

- **Forward**: `F = S * exp((r - q - repo) * T)`
- **Cash delta**: `n_lots * delta * multiplier * S`
- **Cash gamma/1%**: `n_lots * gamma * multiplier * S^2 / 100`
- **DV01**: `modified_duration * dirty_price * 0.0001`
- **Bond PnL**: `(-ModDur * dy + 0.5 * Convexity * dy^2) * Price * n`
- **Callable bond**: lognormal binomial rate tree, issuer calls when continuation > call_price
- **American**: EU and AM prices from same CRR tree (premium >= 0)
- **Turbo**: `price = (S - K) / parity`, `leverage = S / (price * parity)`
- **Discount Cert**: `DC = S * exp(-qT) - Call_BS(K=Cap)`
- **Bonus Cert**: `BC = S * exp(-qT) + Put_DO(K=Bonus, B=Barrier) - Call(K=Cap)`
