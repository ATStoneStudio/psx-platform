"""DAILY job (5pm PKT Mon-Fri): fetch EOD bars + macro, run engine, write output/*.json"""
import json, sqlite3, sys, time, traceback
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from engine.analysis import analyze, WEIGHTS, normalize_weights       # noqa: E402
from engine.dataio import (normalize_frame, upsert_ohlcv, purge_bad_rows,  # noqa: E402
                           load_symbol, ensure_schema, load_macro_context,
                           get_calibration, set_calibration)

DB = ROOT / "data" / "psx.sqlite"
OUT = ROOT / "output"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) psx-platform/1.0"}

EXPIRY_DAYS = 5          # trading sessions a signal stays open before Expired
FILL_WINDOW = 3          # sessions the limit order waits for a pullback fill
CALIB_MIN_RESOLVED = 12  # minimum resolved signals before calibration may act
CALIB_MIN_COMPONENT = 8  # minimum high-score signals per component to judge it
CALIB_STEP = 0.05        # weight moved per calibration adjustment
CALIB_COOLDOWN_DAYS = 7  # min calendar days between weight adjustments

# ---- Trading costs -----------------------------------------------------
# PSX standard retail brokerage is 0.15% of value or 3 paisa per share,
# whichever is HIGHER (PSX notice N-1258), plus SST on the commission and
# CDC/NCCPL charges. Every published return is stated NET of this, because a
# gross number is not money anyone can actually keep. Raw gross returns stay
# in the database; costs are applied at presentation so the rate is auditable.
COST_PCT_PER_SIDE = 0.15      # % of trade value
COST_MIN_PAISA_PER_SHARE = 3  # absolute floor, hurts low-priced shares most
COST_TAX_ON_COMMISSION = 0.15 # SST/levies applied to the commission itself


def round_trip_cost_pct(price):
    """Round-trip cost as a % of position value for a share at `price`."""
    if not price or price <= 0:
        return 2 * COST_PCT_PER_SIDE * (1 + COST_TAX_ON_COMMISSION)
    per_side = max(COST_PCT_PER_SIDE, (COST_MIN_PAISA_PER_SHARE / 100.0) / price * 100)
    return 2 * per_side * (1 + COST_TAX_ON_COMMISSION)


def load_shariah():
    """Mirror of the PSX KMI All Share constituents (see shariah.txt header).
    We never screen stocks ourselves. Returns (set_of_symbols, as_of_date_or_None).
    Missing/unreadable file => empty set => no badge anywhere (fail safe, never
    labels a stock compliant on a guess)."""
    f = ROOT / "shariah.txt"
    if not f.exists():
        print("shariah.txt missing — Shariah badges disabled", file=sys.stderr)
        return set(), None
    syms, as_of = set(), None
    for line in f.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.upper().startswith("AS_OF"):
            as_of = line.split("=", 1)[-1].strip()
            continue
        syms.add(line.upper())
    return syms, as_of


def fetch_eod(symbols):
    import psxdata
    con = sqlite3.connect(DB)
    purge_bad_rows(con)
    start = date.today() - timedelta(days=10)  # overlap heals gaps/holidays
    ok = 0
    for sym in symbols:
        try:
            raw = psxdata.stocks(sym, start, date.today())
            if raw is None or len(raw) == 0:
                continue
            upsert_ohlcv(con, sym, normalize_frame(raw))
            ok += 1
        except Exception as e:
            print(f"{sym}: fetch failed — {e}", file=sys.stderr)
        time.sleep(0.8)
    print(f"EOD fetch: {ok}/{len(symbols)} symbols updated")


def _stooq(code, n=10):
    import io, urllib.request
    req = urllib.request.Request(f"https://stooq.com/q/d/l/?s={code}&i=d", headers=UA)
    df = pd.read_csv(io.StringIO(urllib.request.urlopen(req, timeout=30).read().decode())).tail(n)
    return [(str(r.Date), float(r.Close)) for r in df.itertuples()]


def _yahoo(code, rng="1mo", n=10):
    import urllib.request
    req = urllib.request.Request(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{code}?range={rng}&interval=1d",
        headers=UA)
    j = json.loads(urllib.request.urlopen(req, timeout=30).read())["chart"]["result"][0]
    closes = j["indicators"]["quote"][0]["close"]
    return [(str(pd.Timestamp(ts, unit="s").date()), float(c))
            for ts, c in zip(j["timestamp"], closes) if c is not None][-n:]


def _psx_index(code="KSE100", n=10):
    import urllib.request
    req = urllib.request.Request(f"https://dps.psx.com.pk/timeseries/eod/{code}", headers=UA)
    j = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return sorted((str(pd.Timestamp(r[0], unit="s").date()), float(r[3]))
                  for r in j["data"])[-n:]


def fetch_macro():
    """Each series tries multiple sources; failure of one never blocks the rest.
    Depth: 60d for Brent/USDPKR trend windows; up to 1500d KSE-100 so the 50-EMA
    regime filter works from day one AND the benchmark covers the audited record."""
    con = sqlite3.connect(DB)
    plans = {"brent": [lambda: _stooq("cb.f", 60), lambda: _yahoo("BZ=F", "3mo", 60)],
             "usdpkr": [lambda: _stooq("usdpkr", 60), lambda: _yahoo("PKR=X", "3mo", 60)],
             # deep KSE-100 history so the benchmark line spans the whole audited
             # record; if a source returns less, we simply store what we get
             "kse100": [lambda: _psx_index("KSE100", 1500), lambda: _stooq("^kse", 1500)]}
    for name, sources in plans.items():
        for fn in sources:
            try:
                rows = [(name, d, v) for d, v in fn()]
                con.executemany("INSERT OR REPLACE INTO macro VALUES (?,?,?)", rows)
                con.commit()
                print(f"macro {name}: ok (last={rows[-1][1]} {rows[-1][2]})")
                break
            except Exception as e:
                print(f"macro {name}: source failed — {e}", file=sys.stderr)


# ======================================================================
# Audit trail: log signals -> evaluate outcomes -> calibrate weights
# ======================================================================

def log_prediction(con, res, source="live"):
    """Freeze today's signal in predictions_history (audit trail).
    INSERT OR IGNORE: the 7:30pm retry run never overwrites the 5pm record."""
    o, v = res["order_table"], res["verdict"]
    con.execute(
        "INSERT OR IGNORE INTO predictions_history "
        "(symbol, signal_date, verdict, composite, confidence, horizon, entry, "
        " stop_loss, tp1, tp2, setup_valid, scores_json, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (res["symbol"], res["as_of"], v["call"], v["composite_score"], v["confidence_pct"],
         o["horizon"], o["primary_buy_limit"], o["stop_loss"], o["take_profit_1"],
         o["take_profit_2"], int(o["setup_valid_rr_rule"]), json.dumps(v["scores"]), source))


def evaluate_predictions(con):
    """Resolve open signals against subsequent RAW bars.
    Model: buy limit at `entry`; fills if a later session's Low <= entry within
    FILL_WINDOW sessions. After fill: SL if Low <= stop (checked first each bar,
    conservative), TP2 if High >= tp2, else TP1 noted if High >= tp1; open
    positions expire EXPIRY_DAYS sessions after fill at mark-to-market.
    A corporate action inside the window voids the signal (VOID_CA)."""
    open_rows = con.execute(
        "SELECT symbol, signal_date, entry, stop_loss, tp1, tp2 "
        "FROM predictions_history WHERE outcome IS NULL").fetchall()
    resolved = 0
    for sym, sdate, entry, stop, tp1, tp2 in open_rows:
        bars = con.execute(
            "SELECT date, open, high, low, close FROM ohlcv "
            "WHERE symbol=? AND date>? AND date LIKE '____-__-__' ORDER BY date",
            (sym, sdate)).fetchall()
        if not bars:
            continue
        outcome = fill_date = out_date = None
        days = ret = None
        best = None            # 'TP1' once touched, upgradeable to TP2
        prev_close = None
        fill_k = None
        for k, (d, o_, h, l, c) in enumerate(bars, start=1):
            # corporate action inside evaluation window -> raw levels invalid
            if prev_close and prev_close > 0 and not (0.75 <= c / prev_close <= 1.30):
                outcome, out_date, days = "VOID_CA", d, k
                break
            prev_close = c
            if fill_k is None:
                if l <= entry:
                    fill_k, fill_date = k, d
                    if l <= stop:            # same-bar stop: assume worst case
                        outcome, out_date, days, ret = "SL", d, 0, (stop - entry) / entry
                        break
                    if c >= tp1:             # same-bar TP only if it CLOSED there
                        best = "TP2" if c >= tp2 else "TP1"
                elif k >= FILL_WINDOW:
                    outcome, out_date, days = "NOT_FILLED", d, k
                    break
                continue
            # position open
            if l <= stop:
                if best:                     # TP1 was banked before the stop hit
                    outcome, ret = best, ((tp2 if best == "TP2" else tp1) - entry) / entry
                else:
                    outcome, ret = "SL", (stop - entry) / entry
                out_date, days = d, k - fill_k
                break
            if h >= tp2:
                outcome, out_date, days, ret = "TP2", d, k - fill_k, (tp2 - entry) / entry
                break
            if h >= tp1:
                best = "TP1"
            if k - fill_k >= EXPIRY_DAYS:
                if best:
                    outcome, ret = best, (tp1 - entry) / entry
                else:
                    outcome, ret = "EXPIRED", (c - entry) / entry
                out_date, days = d, k - fill_k
                break
        if outcome:
            con.execute(
                "UPDATE predictions_history SET outcome=?, fill_date=?, outcome_date=?, "
                "days_to_outcome=?, return_pct=? WHERE symbol=? AND signal_date=?",
                (outcome, fill_date, out_date, days,
                 round(ret * 100, 2) if ret is not None else None, sym, sdate))
            resolved += 1
    con.commit()
    print(f"audit: {resolved} signals resolved, {len(open_rows) - resolved} still open")


def calibrate_weights(con):
    """30-day rolling accuracy per component; if signals where a component
    scored high (>=6.5) win measurably less than average, shift CALIB_STEP of
    weight from it to the best-performing component. Bounded + fully logged."""
    max_d = con.execute("SELECT MAX(date) FROM ohlcv WHERE date LIKE '____-__-__'").fetchone()[0]
    rows = con.execute(
        "SELECT scores_json, outcome FROM predictions_history "
        "WHERE outcome IN ('TP1','TP2','SL') AND signal_date >= date(?, '-30 day')",
        (max_d,)).fetchall()
    current = normalize_weights(get_calibration(con, "weights", WEIGHTS))
    stats = {}
    # Cooldown: one adjustment per CALIB_COOLDOWN_DAYS. Without this, the same
    # 30-day evidence would compound a 0.05 shift on every single run.
    log_prev = get_calibration(con, "weights_log", [])
    if log_prev:
        from datetime import datetime
        last_adj = datetime.strptime(log_prev[-1]["date"], "%Y-%m-%d")
        if (datetime.strptime(max_d, "%Y-%m-%d") - last_adj).days < CALIB_COOLDOWN_DAYS:
            print(f"calibration: cooldown ({log_prev[-1]['date']} adjustment still settling)")
            return current, stats, len(rows)
    if len(rows) >= CALIB_MIN_RESOLVED:
        overall = sum(1 for _, oc in rows if oc in ("TP1", "TP2")) / len(rows)
        comp_map = {"trend": "trend", "momentum": "momentum", "volume": "volume",
                    "structure": "structure", "volatility": "volatility", "mtf": "mtf_alignment"}
        for wkey, skey in comp_map.items():
            hi = [(s, oc) for s, oc in ((json.loads(sj), oc) for sj, oc in rows)
                  if s.get(skey, 0) >= 6.5]
            if len(hi) >= CALIB_MIN_COMPONENT:
                stats[wkey] = {"n": len(hi),
                               "win_rate": round(sum(1 for _, oc in hi if oc != "SL") / len(hi), 3)}
        losers = {k: v for k, v in stats.items() if v["win_rate"] < overall - 0.10}
        if losers and stats:
            worst = min(losers, key=lambda k: stats[k]["win_rate"])
            candidates = {k: v for k, v in stats.items() if k != worst}
            if candidates:
                best = max(candidates, key=lambda k: stats[k]["win_rate"])
                adjusted = dict(current)
                adjusted[worst] = current[worst] - CALIB_STEP
                adjusted[best] = current[best] + CALIB_STEP
                new = normalize_weights(adjusted)
                log = get_calibration(con, "weights_log", [])
                log.append({"date": max_d, "moved": CALIB_STEP, "from": worst, "to": best,
                            "reason": f"{worst}-driven signals won {stats[worst]['win_rate']:.0%} "
                                      f"vs {overall:.0%} overall over 30d (n={stats[worst]['n']})",
                            "weights": new})
                set_calibration(con, "weights", new)
                set_calibration(con, "weights_log", log[-50:])
                print(f"calibration: {worst} underperforming -> weights now {new}")
                return new, stats, len(rows)
    print(f"calibration: no change ({len(rows)} resolved signals in window)")
    set_calibration(con, "weights", current)
    return current, stats, len(rows)


def simulate_portfolio(con, start_capital=100000.0, slot_frac=0.20):
    """'Growth of Rs 100,000' series: mechanically follow every Buy-rated valid
    setup from the audited ledger. Each trade takes a fixed 20% slot; its audited
    P/L compounds into equity on the trade's outcome date (step curve — marked at
    trade close, no intraday mark-to-market, no hindsight). KSE-100 closes ride
    along for the benchmark line. Purely additive output: returns None if there is
    nothing to plot, and the frontend hides the card in that case."""
    trades = con.execute(
        "SELECT outcome_date, return_pct, source, entry FROM predictions_history "
        "WHERE setup_valid=1 AND verdict IN ('Buy','Strong Buy') "
        "AND outcome IN ('TP1','TP2','SL','EXPIRED') AND return_pct IS NOT NULL "
        "AND outcome_date IS NOT NULL ORDER BY outcome_date").fetchall()
    if len(trades) < 2:
        return None
    first_signal = con.execute(
        "SELECT MIN(signal_date) FROM predictions_history "
        "WHERE setup_valid=1 AND verdict IN ('Buy','Strong Buy')").fetchone()[0]
    dates = [r[0] for r in con.execute(
        "SELECT DISTINCT date FROM ohlcv WHERE date LIKE '____-__-__' AND date >= ? "
        "ORDER BY date", (first_signal,))]
    kse = dict(con.execute("SELECT date, value FROM macro WHERE series='kse100'"))
    first_live = con.execute(
        "SELECT MIN(outcome_date) FROM predictions_history "
        "WHERE setup_valid=1 AND source='live' AND outcome_date IS NOT NULL").fetchone()[0]
    eq, i, series, last_k = start_capital, 0, [], None
    for d in dates:
        while i < len(trades) and trades[i][0] <= d:
            net = float(trades[i][1]) - round_trip_cost_pct(trades[i][3])   # NET of costs
            eq *= (1 + slot_frac * net / 100)
            i += 1
        kv = kse.get(d)
        if kv is not None:
            last_k = float(kv)
        series.append([d, round(eq, 2), round(last_k, 2) if last_k else None])
    return {"start_capital": start_capital, "slot_pct": int(slot_frac * 100),
            "net_of_costs": True,
            "cost_note": f"Every trade is charged a realistic PSX round trip "
                         f"({COST_PCT_PER_SIDE}% or {COST_MIN_PAISA_PER_SHARE} paisa/share per side, "
                         f"whichever is higher, plus {int(COST_TAX_ON_COMMISSION*100)}% tax on commission). "
                         f"Gross figures would be materially higher and are not achievable.",
            "rules": f"Every Buy/Strong Buy-rated valid setup takes a {int(slot_frac*100)}% "
                     "slot of equity; its audited P/L (incl. losses, expiries, fills) is "
                     "applied at trade close and compounds. No hindsight, no cherry-picking.",
            "first_live_close": first_live, "trades": len(trades), "series": series}


def signal_sample(con, portfolio=None):
    """Evidence the planner needs to state RANGES instead of a fake projection:
    the actual distribution of Buy-rated trade returns, how often setups appear,
    the worst peak-to-trough drop, and the best/worst calendar month.
    Everything here is measured, never modelled."""
    rows = con.execute(
        "SELECT signal_date, return_pct, entry FROM predictions_history "
        "WHERE setup_valid=1 AND verdict IN ('Buy','Strong Buy') "
        "AND outcome IN ('TP1','TP2','SL','EXPIRED') AND return_pct IS NOT NULL "
        "ORDER BY signal_date").fetchall()
    if len(rows) < 10:
        return None
    gross = [float(r[1]) for r in rows]
    rets = [round(float(r[1]) - round_trip_cost_pct(r[2]), 3) for r in rows]   # NET
    d0, d1 = rows[0][0], rows[-1][0]
    from datetime import datetime as _dt
    months = max(0.5, (_dt.strptime(d1, "%Y-%m-%d") - _dt.strptime(d0, "%Y-%m-%d")).days / 30.44)
    out = {"trades": len(rets), "returns": rets, "from": d0, "to": d1,
           "net_of_costs": True,
           "avg_gross_pct": round(sum(gross) / len(gross), 3),
           "avg_cost_pct": round(sum(gross) / len(gross) - sum(rets) / len(rets), 3),
           "months_covered": round(months, 1),
           "trades_per_month": round(len(rets) / months, 2),
           "win_rate_pct": round(100 * sum(1 for x in rets if x > 0) / len(rets), 1),
           "avg_return_pct": round(sum(rets) / len(rets), 2),
           "worst_trade_pct": min(rets), "best_trade_pct": max(rets),
           "max_drawdown_pct": None, "worst_month_pct": None, "best_month_pct": None,
           "months": []}
    if portfolio and portfolio.get("series"):
        ser = portfolio["series"]
        peak, dd = ser[0][1], 0.0
        for _, eq, _k in ser:
            peak = max(peak, eq)
            dd = min(dd, (eq / peak - 1) * 100)
        out["max_drawdown_pct"] = round(dd, 2)
        by_month = {}
        for d, eq, _k in ser:
            by_month.setdefault(d[:7], []).append(eq)
        ms = []
        for m in sorted(by_month):
            v = by_month[m]
            ms.append({"month": m, "pct": round((v[-1] / v[0] - 1) * 100, 2)})
        out["months"] = ms
        if ms:
            out["worst_month_pct"] = min(x["pct"] for x in ms)
            out["best_month_pct"] = max(x["pct"] for x in ms)
    return out


def write_track_record(con):
    """Public, transparent ledger -> output/track_record.json (index.html)."""
    def bucket(where=""):
        rows = con.execute(
            "SELECT outcome, return_pct, days_to_outcome, entry FROM predictions_history "
            f"WHERE setup_valid=1 AND outcome IS NOT NULL {where}").fetchall()
        played = [r for r in rows if r[0] in ("TP1", "TP2", "SL", "EXPIRED")]
        wins = [r for r in played if r[0] in ("TP1", "TP2")]
        # NET of trading costs — must match the equity curve and the sample,
        # otherwise the page contradicts itself.
        rets = [r[1] - round_trip_cost_pct(r[3]) for r in played if r[1] is not None]
        return {"signals": len(rows), "trades": len(played), "wins": len(wins),
                "losses": sum(1 for r in played if r[0] == "SL"),
                "expired": sum(1 for r in played if r[0] == "EXPIRED"),
                "not_filled": sum(1 for r in rows if r[0] == "NOT_FILLED"),
                "win_rate_pct": round(100 * len(wins) / len(played), 1) if played else None,
                "avg_return_pct": round(sum(rets) / len(rets), 2) if rets else None,
                "returns_net_of_costs": True,
                "avg_days_to_win": (round(sum(r[2] for r in wins if r[2] is not None)
                                          / max(1, len([r for r in wins if r[2] is not None])), 1)
                                    if wins else None)}
    max_d = con.execute("SELECT MAX(date) FROM ohlcv WHERE date LIKE '____-__-__'").fetchone()[0]
    ledger = [dict(zip(("symbol", "signal_date", "verdict", "entry", "stop_loss",
                        "tp1", "tp2", "outcome", "outcome_date", "days", "return_pct",
                        "source"), r))
              for r in con.execute(
                  "SELECT symbol, signal_date, verdict, entry, stop_loss, tp1, tp2, outcome, "
                  "outcome_date, days_to_outcome, return_pct, source FROM predictions_history "
                  "WHERE setup_valid=1 AND outcome IS NOT NULL "
                  "ORDER BY outcome_date DESC, symbol LIMIT 60").fetchall()]
    open_n = con.execute("SELECT COUNT(*) FROM predictions_history "
                         "WHERE setup_valid=1 AND outcome IS NULL").fetchone()[0]
    weights = get_calibration(con, "weights", WEIGHTS)
    # Two cuts, both published: "buy" = setups the site actually recommended
    # (verdict Buy/Strong Buy), "all setups" = every R:R-valid setup regardless
    # of verdict. Headline = buy; all-setups shown alongside for transparency.
    BUY = "AND verdict IN ('Buy','Strong Buy')"
    W30 = f"AND signal_date >= date('{max_d}', '-30 day')"
    track = {"generated": str(date.today()),
             "method": {"fill_rule": f"limit at entry, fills if Low touches within {FILL_WINDOW} sessions",
                        "expiry_days": EXPIRY_DAYS,
                        "conservative": "same-bar stop counts as a loss; targets only count on touch after fill",
                        "universe": "Buy-rated = verdict Buy/Strong Buy AND R:R-valid; "
                                    "all-setups = every R:R-valid setup regardless of verdict"},
             "last_30d_buy": bucket(f"{W30} {BUY}"), "all_time_buy": bucket(BUY),
             "last_30d": bucket(W30),
             "all_time": bucket(), "live_only": bucket("AND source='live'"),
             "backfill_note": "Rows marked 'backfill' are a simulated walk-forward replay of the "
                              "engine on historical data — not signals published in advance.",
             "open_signals": open_n,
             "portfolio": (pf := simulate_portfolio(con)),
             "sample": signal_sample(con, pf),
             "weights": weights, "weights_log": get_calibration(con, "weights_log", [])[-5:],
             "ledger": ledger}
    (OUT / "track_record.json").write_text(json.dumps(track, default=str))
    print(f"track record: buy-rated {track['all_time_buy']['trades']} trades "
          f"(30d win {track['last_30d_buy']['win_rate_pct']}%), "
          f"all setups {track['all_time']['trades']} trades "
          f"(30d win {track['last_30d']['win_rate_pct']}%)")
    return track


def replay_ledger(n_sessions=40, max_symbols=None):
    """Walk-forward replay: re-run the engine as-of each of the last n sessions,
    log signals with source='backfill', then evaluate them. Seeds the public
    track record honestly (rows are labelled simulated) and gives calibration
    a 30-day sample from day one. Run once: python daily_update.py --replay 40"""
    con = sqlite3.connect(DB)
    ensure_schema(con)
    macro_ctx = load_macro_context(con)
    dates = [r[0] for r in con.execute(
        "SELECT DISTINCT date FROM ohlcv WHERE date LIKE '____-__-__' ORDER BY date")][-n_sessions - 1:-1]
    symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM ohlcv "
                                         "WHERE date LIKE '____-__-__'")][:max_symbols]
    logged = 0
    for sym in symbols:
        df_full, _ = load_symbol(con, sym)
        for d in dates:
            df = df_full[df_full.index <= d]
            if len(df) < 60 or str(df.index[-1].date()) != d:
                continue  # symbol didn't trade that day — don't fabricate a signal
            try:
                ctx = {k: (v[v.index <= d] if v is not None and len(v) else v)
                       for k, v in macro_ctx.items()}
                res = analyze(sym, df, context=ctx)      # default weights: no lookahead
                log_prediction(con, res, source="backfill")
                logged += 1
            except Exception as e:
                print(f"replay {sym}@{d}: {e}", file=sys.stderr)
    con.commit()
    print(f"replay: logged {logged} signal-days across {len(symbols)} symbols")
    evaluate_predictions(con)
    write_track_record(con)


def run_engine():
    con = sqlite3.connect(DB)
    ensure_schema(con)
    evaluate_predictions(con)                      # resolve yesterday's signals first
    weights, calib_stats, calib_n = calibrate_weights(con)
    macro_ctx = load_macro_context(con)
    shariah, shariah_as_of = load_shariah()
    print(f"shariah list: {len(shariah)} symbols (as of {shariah_as_of})")
    symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM ohlcv WHERE date LIKE '____-__-__'")]
    OUT.mkdir(exist_ok=True)
    # exclude symbols with no fresh data (renamed/delisted/suspended) — a stale
    # quote presented as current is worse for a trader than no quote at all
    max_date = con.execute("SELECT MAX(date) FROM ohlcv WHERE date LIKE '____-__-__'").fetchone()[0]
    stale = dict(con.execute(
        "SELECT symbol, MAX(date) m FROM ohlcv GROUP BY symbol HAVING m < date(?, '-10 day')",
        (max_date,)).fetchall())
    if stale:
        print(f"excluding {len(stale)} stale symbols (last data too old): {stale}")
        symbols = [s for s in symbols if s not in stale]
        for s in stale:  # remove any previously published analysis
            (OUT / f"{s}.json").unlink(missing_ok=True)
    summary, first_error = [], True
    for sym in symbols:
        try:
            df, ca_events = load_symbol(con, sym)
            if len(df) < 60:
                print(f"{sym}: skipped ({len(df)} rows < 60)")
                continue
            res = analyze(sym, df, context=macro_ctx, weights=weights)
            res["corporate_actions_adjusted"] = ca_events
            res["shariah_compliant"] = sym in shariah
            res["shariah_as_of"] = shariah_as_of
            res["price_history"] = [[d.strftime("%Y-%m-%d"), round(float(c), 2)] for d, c in df["Close"].tail(120).items()]
            (OUT / f"{sym}.json").write_text(json.dumps(res, default=str))
            log_prediction(con, res)               # audit trail: freeze today's levels
            summary.append({"symbol": sym, "close": res["close"], "as_of": res["as_of"],
                            "verdict": res["verdict"]["call"],
                            "confidence": res["verdict"]["confidence_pct"],
                            "score": res["verdict"]["composite_score"],
                            "structure": res["market_structure"],
                            "sector": res["macro_context"]["sector"],
                            "shariah": sym in shariah,
                            "macro_adj": res["macro_context"]["adjustment"],
                            "setup_valid": res["order_table"]["setup_valid_rr_rule"],
                            "primary_buy": res["order_table"]["primary_buy_limit"],
                            "stop_loss": res["order_table"]["stop_loss"],
                            "tp1": res["order_table"]["take_profit_1"],
                            "est_days_tp1": res["order_table"]["time_to_target"]["tp1_days"]})
        except Exception as e:
            print(f"{sym}: engine failed — {e}", file=sys.stderr)
            if first_error:  # full traceback once, to make CI logs actionable
                traceback.print_exc()
                first_error = False
    con.commit()                                   # persist logged predictions
    summary.sort(key=lambda x: -x["score"])
    regime = "Unknown"
    k = macro_ctx.get("kse100")
    if k is not None and len(k.dropna()) >= 50:
        kk = k.dropna()
        regime = "Risk-On" if float(kk.iloc[-1]) >= float(kk.ewm(span=50, adjust=False).mean().iloc[-1]) else "Risk-Off"
    (OUT / "summary.json").write_text(json.dumps(
        {"generated": str(date.today()), "stocks": summary, "excluded_stale": stale,
         "regime": regime, "weights": weights,
         "shariah_as_of": shariah_as_of,
         "shariah_source": "PSX KMI All Share Index constituents (screening by Al Meezan Shariah Supervisory Board). We do not screen stocks ourselves.",
         "calibration": {"resolved_30d": calib_n, "component_stats": calib_stats},
         "macro": dict(con.execute("SELECT series, value FROM macro WHERE (series, date) IN (SELECT series, MAX(date) FROM macro GROUP BY series)"))}, default=str))
    write_track_record(con)
    print(f"Engine done: {len(summary)}/{len(symbols)} stocks analyzed.")
    if not summary:
        sys.exit(1)  # fail the workflow loudly instead of committing empty output


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--replay":
        replay_ledger(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
        sys.exit(0)
    symbols = [s.split("#")[0].strip().upper() for s in (ROOT / "kse100.txt").read_text().splitlines()
               if s.split("#")[0].strip()]
    fetch_eod(symbols)
    fetch_macro()
    run_engine()