# PSX Daily Trading Platform — Analysis Engine (Step 1)

Zero-LLM, formula-only PSX analysis. Runs free on GitHub Actions + static hosting.

## Architecture
1. `backfill.py` — ONE-TIME: 5yr EOD OHLCV for KSE-100 -> data/psx.sqlite (uses `psxdata`)
2. `daily_update.py` — DAILY 5pm PKT Mon-Fri (`.github/workflows/daily.yml`):
   fetch EOD bars + macro (Brent, USD/PKR, KSE-100) -> run engine -> output/*.json
3. `engine/` — pure pandas/numpy: all indicators, pivots (project methodology),
   Fibonacci, market structure (HH/HL/LH/LL, BOS/CHOCH, OB, FVG), candlesticks,
   ZigZag Elliott estimate, M/W/D multi-timeframe, weighted verdict + R:R gate.
4. Web UI (Step 2) reads `output/summary.json` + `output/<SYMBOL>.json`.

## Verdict scoring
trend 25% | momentum 20% | structure 20% | volume 15% | volatility 10% | MTF 10%
>=7.5 Strong Buy | >=6.2 Buy | >=4.5 Hold | >=3.2 Sell | else Strong Sell.
Setups with R:R < 1:2 are flagged invalid (project rule).

## Setup
1. Push this folder to a GitHub repo.
2. Run backfill once (locally or via a manual Actions run): `pip install pandas numpy psxdata && python backfill.py`
3. Commit `data/psx.sqlite`. The daily workflow keeps it updated and commits `output/`.
4. Edit `kse100.txt` after index recompositions.

## Honest limitations (v1)
No intraday timeframes (4H/1H/15M), no fundamentals (phase 2), no news sentiment.
All output is probabilistic technical analysis, not investment advice.
# psx-platform
# psx-platform
