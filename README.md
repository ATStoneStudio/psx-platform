# PSX Web Platform — Architecture & Status

Goal: zero-LLM daily trading platform for KSE-100. Formula-only analysis, runs free.

## Stack (LIVE, all green as of 2026-08-17; v2 upgrade authored 2026-08-19, pending commit)
- Repo: https://github.com/ATStoneStudio/psx-platform (canonical source of truth)
- Site: GitHub Pages from main root → https://atstonestudio.github.io/psx-platform/ (index.html dashboard + stock.html?s=SYM detail)
- Data: `psxdata` (scrapes dps.psx.com.pk) → data/psx.sqlite (5y EOD, 94 KSE-100 symbols)
- Cron: Actions `0 12 * * 1-5` + `30 14 * * 1-5` retry (5pm & 7:30pm PKT) → daily_update.py → commits data + output/*.json (incl. price_history 120d + macro in summary)
- Macro multi-source fallback: Brent (stooq→Yahoo), USD/PKR (stooq→Yahoo), KSE-100 (PSX portal r[3]=close). v2 fetch depth: 60d Brent/USDPKR, 400d KSE-100 (for 50-EMA regime filter).
- Engine: pure pandas/numpy; math cross-validated (Wilder exact); pivots match project methodology; recompute matched committed output byte-for-byte
- Safeguards (dataio.py): date-column detection, OHLC cleaning, split back-adjust (ratio outside [0.75,1.30]), stale-symbol exclusion, CI fails loudly on empty output
- Tickers renamed in kse100.txt: ICI→LCI, ENGRO→ENGROH, EFOODS→FCEPL, LALPIR→LPL, SHEL→WAFI; PSMC/DAWH removed. Parser strips inline # comments.

## v2 upgrade (2026-08-19) — 5 features, all in `code/` snapshot, tested on repo clone + real DB
1. **Timeframe tags + position sizer**: order table tagged Short-Term (1-3d), verdict tagged Swing (2-4w), price_targets Long-Term. stock.html has interactive sizer: shares = capital×risk% ÷ (entry−SL), capped by affordability, default 2% risk.
2. **Macro scoring** (analysis.py): SECTORS map (94 symbols→buckets). Brent ±5%/20d → E&P/Refinery/OMC ±0.6, Cement/Auto/Power/Electronics ∓0.4. USDPKR ±2%/20d → exporters (Textile/Tech/E&P) vs importers (Auto/Cement/Steel/Pharma). KSE-100 < 50-EMA → −0.75 all (Risk-Off regime). Total clipped ±1.5, added to composite (clip 0-10). Output: `macro_context` + `composite_technical` vs `composite_score`.
3. **Dynamic calibration** (daily_update.py): 30d win rates per component (high-score ≥6.5 signals). If component win rate < overall −10pts (n≥8, ≥12 resolved), shift 0.05 weight to best component. Bounds [0.10, 0.35], renormalized, 7-day cooldown, full audit log in `calibration` table (keys: weights, weights_log). analyze() takes `weights=` param.
4. **Audit trail** (predictions_history table in psx.sqlite): every signal frozen at publish (INSERT OR IGNORE — retry run can't overwrite). Evaluation: limit fills if Low≤entry within 3 sessions else NOT_FILLED; after fill SL if Low≤stop (same-bar stop = loss, conservative), TP2/TP1 on touch, EXPIRED after 5 sessions at mark-to-market; corporate action in window → VOID_CA. `source` column: 'live' vs 'backfill' (simulated replay via `python daily_update.py --replay 40`, labelled SIM in UI). Output: output/track_record.json → index.html track-record section (win rates 30d/all-time/live-only, ledger, method disclosure).
5. **Time-to-target**: est days = ceil(distance ÷ 0.5×ATR14) for TP1/TP2, shown in order table with basis note.

Order of operations in run_engine: ensure_schema → evaluate_predictions → calibrate_weights → load_macro_context → analyze(sym, df, context, weights) → log_prediction → summary.json (+ regime, weights, calibration stats) → track_record.json.

Test results (clone, 2026-08-18 data): replay 40 sessions × 25 symbols = 920 signals, 850 resolved; calibration correctly declined to adjust at n=209 (no component under threshold); same-day rerun idempotent (94 live rows, not 188); both pages rendered headless with zero JS errors; sizer math verified (500k cap, 1.5% → risk Rs 7,492 ≤ 7,500).

## Operational lessons
- PSX firewall temporarily blocked GitHub Actions IPs after 3 backfills in 2 days — avoid re-running backfill; daily job is light. Blocks lift within hours.
- Workflow commit order matters: add+commit → pull --rebase → push (rebase before commit fails on unstaged changes).
- Claude sessions have no push access unless user adds repo to session sources; user commits via GitHub web UI (must select "Commit directly to main").
- Calibration without a cooldown compounds the same 30-day evidence every run — fixed with CALIB_COOLDOWN_DAYS=7.
- Macro table had only ~11 days of history at v2 time; regime filter self-disables below 50 sessions of KSE-100 and reports "regime filter inactive". Deepened fetch fills it on first Actions run.

## Deploy notes for v2 (user action)
- Commit 5 files: engine/analysis.py, engine/dataio.py, daily_update.py, index.html, stock.html (workflow unchanged).
- Optional one-time: run `python daily_update.py --replay 40` locally or via a temp workflow step to seed the ledger (rows honestly labelled SIM). Otherwise ledger populates organically from day 1.
- predictions_history/calibration tables auto-create (ensure_schema is idempotent; ALTER migration for `source` included).

## NEXT: Step 3 auth + Step 4 payments (decisions pending)
- Requires real backend: repo → private, data behind API w/ subscription check. Proposed: Supabase (auth Google/Facebook/LinkedIn — Instagram login NOT viable, Meta discontinued) + Cloudflare Pages.
- Payments: Stripe unavailable in Pakistan. Options: Paddle / LemonSqueezy (merchant-of-record; verify PK payout) or local Safepay / PayFast. Trial: 1 month free → $10/mo.
- Flag: charging for buy/sell signals strengthens SECP investment-advice licensing question — needs legal opinion before launch. The public track record (v2) also makes accuracy claims verifiable — never market a win-rate the ledger doesn't show.
- Verdict (swing view) is shown separately from pivot order table (intraday, R:R-gated, most days "no valid setup" — by design; ~6/94 pass).