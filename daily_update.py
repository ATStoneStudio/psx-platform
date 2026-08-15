"""DAILY job (5pm PKT, Mon-Fri via GitHub Actions cron):
1. Fetch today's EOD bar for every KSE-100 symbol -> append to SQLite
2. Fetch macro context: Brent crude, USD/PKR, KSE-100 index
3. Run the analysis engine on every symbol -> output/<SYMBOL>.json + output/summary.json
The static site reads these JSON files. Zero LLM calls anywhere.
"""
import json, sqlite3, sys, time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from engine.analysis import analyze  # noqa: E402

DB = ROOT / "data" / "psx.sqlite"
OUT = ROOT / "output"


def fetch_eod(symbols):
    import psxdata
    con = sqlite3.connect(DB)
    start = date.today() - timedelta(days=7)  # small overlap heals gaps/holidays
    for i, sym in enumerate(symbols, 1):
        try:
            df = psxdata.stocks(sym, start, date.today())
            if df is None or df.empty:
                continue
            df.columns = [c.title() for c in df.columns]
            rows = [(sym, str(ix.date() if hasattr(ix, "date") else ix),
                     float(r.Open), float(r.High), float(r.Low), float(r.Close), int(r.Volume))
                    for ix, r in df.iterrows()]
            con.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?)", rows)
            con.commit()
        except Exception as e:
            print(f"{sym}: fetch failed — {e}", file=sys.stderr)
        time.sleep(0.8)


def fetch_macro():
    """Brent, USD/PKR, KSE-100 index — best-effort from free endpoints (stooq CSV)."""
    import urllib.request
    con = sqlite3.connect(DB)
    series = {"brent": "cb.f", "usdpkr": "usdpkr", "kse100": "^kse"}
    for name, code in series.items():
        try:
            url = f"https://stooq.com/q/d/l/?s={code}&i=d"
            csv = urllib.request.urlopen(url, timeout=30).read().decode()
            df = pd.read_csv(pd.io.common.StringIO(csv)).tail(10)
            rows = [(name, str(r.Date), float(r.Close)) for r in df.itertuples()]
            con.executemany("INSERT OR REPLACE INTO macro VALUES (?,?,?)", rows)
            con.commit()
            print(f"macro {name}: ok ({len(rows)} rows)")
        except Exception as e:
            print(f"macro {name}: failed — {e}", file=sys.stderr)


def run_engine():
    con = sqlite3.connect(DB)
    symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM ohlcv")]
    OUT.mkdir(exist_ok=True)
    summary = []
    for sym in symbols:
        df = pd.read_sql("SELECT * FROM ohlcv WHERE symbol=? ORDER BY date", con,
                         params=(sym,), index_col="date", parse_dates=["date"])
        df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
        if len(df) < 60:
            continue
        try:
            res = analyze(sym, df)
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
    summary.sort(key=lambda x: -x["score"])
    (OUT / "summary.json").write_text(json.dumps(
        {"generated": str(date.today()), "stocks": summary}, default=str))
    print(f"Engine done: {len(summary)} stocks analyzed.")


if __name__ == "__main__":
    symbols = [s.strip().upper() for s in (ROOT / "kse100.txt").read_text().splitlines()
               if s.strip() and not s.startswith("#")]
    fetch_eod(symbols)
    fetch_macro()
    run_engine()
