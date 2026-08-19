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
    Depth: 60d for Brent/USDPKR trend windows; 400d KSE-100 so the 50-EMA
    regime filter works from day one instead of after 50 daily runs."""
    con = sqlite3.connect(DB)
    plans = {"brent": [lambda: _stooq("cb.f", 60), lambda: _yahoo("BZ=F", "3mo", 60)],
             "usdpkr": [lambda: _stooq("usdpkr", 60), lambda: _yahoo("PKR=X", "3mo", 60)],
             "kse100": [lambda: _psx_index("KSE100", 400), lambda: _stooq("^kse", 400)]}
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


def write_track_record(con):
    """Public, transparent ledger -> output/track_record.json (index.html)."""
    def bucket(where=""):
        rows = con.execute(
            "SELECT outcome, return_pct, days_to_outcome FROM predictions_history "
            f"WHERE setup_valid=1 AND outcome IS NOT NULL {where}").fetchall()
        played = [r for r in rows if r[0] in ("TP1", "TP2", "SL", "EXPIRED")]
        wins = [r for r in played if r[0] in ("TP1", "TP2")]
        rets = [r[1] for r in played if r[1] is not None]
        return {"signals": len(rows), "trades": len(played), "wins": len(wins),
                "losses": sum(1 for r in played if r[0] == "SL"),
                "expired": sum(1 for r in played if r[0] == "EXPIRED"),
                "not_filled": sum(1 for r in rows if r[0] == "NOT_FILLED"),
                "win_rate_pct": round(100 * len(wins) / len(played), 1) if played else None,
                "avg_return_pct": round(sum(rets) / len(rets), 2) if rets else None,
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
    track = {"generated": str(date.today()),
             "method": {"fill_rule": f"limit at entry, fills if Low touches within {FILL_WINDOW} sessions",
                        "expiry_days": EXPIRY_DAYS,
                        "conservative": "same-bar stop counts as a loss; targets only count on touch after fill",
                        "universe": "R:R-gated valid setups only (setup_valid=1)"},
             "last_30d": bucket(f"AND signal_date >= date('{max_d}', '-30 day')"),
             "all_time": bucket(), "live_only": bucket("AND source='live'"),
             "backfill_note": "Rows marked 'backfill' are a simulated walk-forward replay of the "
                              "engine on historical data — not signals published in advance.",
             "open_signals": open_n,
             "weights": weights, "weights_log": get_calibration(con, "weights_log", [])[-5:],
             "ledger": ledger}
    (OUT / "track_record.json").write_text(json.dumps(track, default=str))
    print(f"track record: {track['all_time']['trades']} trades all-time, "
          f"30d win rate {track['last_30d']['win_rate_pct']}%")
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
            res["price_history"] = [[d.strftime("%Y-%m-%d"), round(float(c), 2)] for d, c in df["Close"].tail(120).items()]
            (OUT / f"{sym}.json").write_text(json.dumps(res, default=str))
            log_prediction(con, res)               # audit trail: freeze today's levels
            summary.append({"symbol": sym, "close": res["close"], "as_of": res["as_of"],
                            "verdict": res["verdict"]["call"],
                            "confidence": res["verdict"]["confidence_pct"],
                            "score": res["verdict"]["composite_score"],
                            "structure": res["market_structure"],
                            "sector": res["macro_context"]["sector"],
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