"""DAILY job (5pm PKT Mon-Fri): fetch EOD bars + macro, run engine, write output/*.json"""
import json, sqlite3, sys, time, traceback
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from engine.analysis import analyze                                   # noqa: E402
from engine.dataio import normalize_frame, upsert_ohlcv, purge_bad_rows, load_symbol  # noqa: E402

DB = ROOT / "data" / "psx.sqlite"
OUT = ROOT / "output"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) psx-platform/1.0"}


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


def _stooq(code):
    import io, urllib.request
    req = urllib.request.Request(f"https://stooq.com/q/d/l/?s={code}&i=d", headers=UA)
    df = pd.read_csv(io.StringIO(urllib.request.urlopen(req, timeout=30).read().decode())).tail(10)
    return [(str(r.Date), float(r.Close)) for r in df.itertuples()]


def _yahoo(code):
    import urllib.request
    req = urllib.request.Request(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{code}?range=1mo&interval=1d",
        headers=UA)
    j = json.loads(urllib.request.urlopen(req, timeout=30).read())["chart"]["result"][0]
    closes = j["indicators"]["quote"][0]["close"]
    return [(str(pd.Timestamp(ts, unit="s").date()), float(c))
            for ts, c in zip(j["timestamp"], closes) if c is not None][-10:]


def _psx_index(code="KSE100"):
    import urllib.request
    req = urllib.request.Request(f"https://dps.psx.com.pk/timeseries/eod/{code}", headers=UA)
    j = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return sorted((str(pd.Timestamp(r[0], unit="s").date()), float(r[3]))
                  for r in j["data"])[-10:]


def fetch_macro():
    """Each series tries multiple sources; failure of one never blocks the rest."""
    con = sqlite3.connect(DB)
    plans = {"brent": [lambda: _stooq("cb.f"), lambda: _yahoo("BZ=F")],
             "usdpkr": [lambda: _stooq("usdpkr"), lambda: _yahoo("PKR=X")],
             "kse100": [lambda: _psx_index("KSE100"), lambda: _stooq("^kse")]}
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


def run_engine():
    con = sqlite3.connect(DB)
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
            res = analyze(sym, df)
            res["corporate_actions_adjusted"] = ca_events
            (OUT / f"{sym}.json").write_text(json.dumps(res, default=str))
            summary.append({"symbol": sym, "close": res["close"], "as_of": res["as_of"],
                            "verdict": res["verdict"]["call"],
                            "confidence": res["verdict"]["confidence_pct"],
                            "score": res["verdict"]["composite_score"],
                            "structure": res["market_structure"],
                            "setup_valid": res["order_table"]["setup_valid_rr_rule"],
                            "primary_buy": res["order_table"]["primary_buy_limit"],
                            "stop_loss": res["order_table"]["stop_loss"],
                            "tp1": res["order_table"]["take_profit_1"]})
        except Exception as e:
            print(f"{sym}: engine failed — {e}", file=sys.stderr)
            if first_error:  # full traceback once, to make CI logs actionable
                traceback.print_exc()
                first_error = False
    summary.sort(key=lambda x: -x["score"])
    (OUT / "summary.json").write_text(json.dumps(
        {"generated": str(date.today()), "stocks": summary,
         "excluded_stale": stale}, default=str))
    print(f"Engine done: {len(summary)}/{len(symbols)} stocks analyzed.")
    if not summary:
        sys.exit(1)  # fail the workflow loudly instead of committing empty output


if __name__ == "__main__":
    symbols = [s.split("#")[0].strip().upper() for s in (ROOT / "kse100.txt").read_text().splitlines()
               if s.split("#")[0].strip()]
    fetch_eod(symbols)
    fetch_macro()
    run_engine()
