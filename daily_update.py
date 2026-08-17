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


def fetch_macro():
    """Brent, USD/PKR, KSE-100 index from stooq CSV (browser UA required)."""
    import io, urllib.request
    con = sqlite3.connect(DB)
    for name, code in {"brent": "cb.f", "usdpkr": "usdpkr", "kse100": "^kse"}.items():
        try:
            req = urllib.request.Request(f"https://stooq.com/q/d/l/?s={code}&i=d", headers=UA)
            csv = urllib.request.urlopen(req, timeout=30).read().decode()
            df = pd.read_csv(io.StringIO(csv)).tail(10)
            rows = [(name, str(r.Date), float(r.Close)) for r in df.itertuples()]
            con.executemany("INSERT OR REPLACE INTO macro VALUES (?,?,?)", rows)
            con.commit()
            print(f"macro {name}: ok ({len(rows)} rows, last={rows[-1][1]} {rows[-1][2]})")
        except Exception as e:
            print(f"macro {name}: failed — {e}", file=sys.stderr)


def run_engine():
    con = sqlite3.connect(DB)
    symbols = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM ohlcv WHERE date LIKE '____-__-__'")]
    OUT.mkdir(exist_ok=True)
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
        {"generated": str(date.today()), "stocks": summary}, default=str))
    print(f"Engine done: {len(summary)}/{len(symbols)} stocks analyzed.")
    if not summary:
        sys.exit(1)  # fail the workflow loudly instead of committing empty output


if __name__ == "__main__":
    symbols = [s.strip().upper() for s in (ROOT / "kse100.txt").read_text().splitlines()
               if s.strip() and not s.startswith("#")]
    fetch_eod(symbols)
    fetch_macro()
    run_engine()
