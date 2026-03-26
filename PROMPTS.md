# Golden Prompts

The 10 prompts that built this project from scratch with Claude Code.

---

**1. Project setup**

> Create a Python project called "pricer" with a virtual environment (venv), a .gitignore for Python (include .venv/, \_\_pycache\_\_/, .env, *.pyc), and a requirements.txt with streamlit, plotly, numpy, scipy, pandas. Initialize a git repo.

**2. Black-Scholes engine**

> Build a bs_engine.py module with Black-Scholes pricing for European calls and puts with cost-of-carry (risk-free rate, dividend yield, repo rate). Include all Greeks: delta, gamma, vega, theta, rho, charm, vanna — both unit and cash versions. Use scipy.stats for the normal distribution.

**3. Options tab (Streamlit UI)**

> Create an app.py with Streamlit. First tab: "Options". Sidebar inputs for spot, strike, vol, rate, dividend, repo, time to maturity, lots, multiplier. Display price + all cash Greeks as metrics. Add 6 sensitivity charts with Plotly (delta/gamma/vega vs spot by vol, and vs time by moneyness). Add a Gamma PnL calculator and trading shortcuts (breakeven spot, breakeven vol, gamma/theta ratio).

**4. American options**

> Add a numerical_engine.py with a CRR binomial tree for American call and put pricing. In the Options tab, add an expander comparing European vs American prices and showing the early exercise premium.

**5. Bond engine + tab**

> Build a bond_engine.py with: dirty/clean price from YTM, accrued interest, Macaulay and modified duration, convexity, DV01, PV01. Add callable bond pricing using a lognormal binomial interest rate tree (issuer calls when continuation value > call price). Create a "Bonds" tab with price-yield curve, cash flow waterfall chart, and an SVG rate tree visualization.

**6. Turbo certificates**

> Build a turbo_engine.py for Turbo Open-End Long and Short. Price = (S - K) / parity with knock-out at barrier. Include financing cost simulation (daily strike drift). Create a "Turbo" tab with payoff chart, strike drift chart over time, and a sensitivity table showing price/leverage/barrier distance across spot scenarios.

**7. Discount certificates**

> Build a discount_engine.py. Replication: DC = S * e^(-qT) - Call_BS(K=Cap). Create a "Discount Cert." tab with payoff at maturity, P&L chart, and dual-axis sensitivity charts (price + discount% vs vol, price + max return vs cap level).

**8. Bonus certificates**

> Build a bonus_engine.py with Reiner-Rubinstein closed-form for down-and-out puts. Replication: BC = S * e^(-qT) + Put_DO(K=Bonus, B=Barrier) - Call(K=Cap). Support separate put vol and call vol for skew. Create a "Bonus Cert." tab with gap-style payoff chart, sensitivity vs vol and time, and a pricing decomposition table.

**9. Interview Q&A system**

> Create a questions.json file with sections keyed by tab (options, bonds, discount_cert, turbo, bonus_cert, interview_greeks). Each section has a title, subtitle, and array of {q, a} objects with markdown answers. In app.py, add a _render_qa() function that reads the JSON and displays each question in a collapsible st.expander. Add a standalone "Interview" tab for general trading questions.

**10. Deployment + caching**

> Add @st.cache_data decorators on all heavy compute functions (vol sweeps, time sweeps, spot ladders). Make sure requirements.txt is UTF-8 encoded and pinned. Push to GitHub and connect to Streamlit Cloud for auto-deploy on every push to main. Verify Python 3.9 compatibility (use Optional[float] instead of float | None).
