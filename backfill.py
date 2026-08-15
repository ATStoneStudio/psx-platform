"""ONE-TIME backfill: ~5 years of EOD OHLCV for KSE-100 constituents -> data/psx.sqlite
Run on GitHub Actions or any machine with open internet (PSX portal is scraped
via the `psxdata` library:  pip install psxdata).

Usage:  python backfill.py [--years 5] [--symbols-file kse100.txt]
"""
import argparse, sqlite3, sys, time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DB = Path(__file__).parent / "data" / "psx.sqlite"

# KSE-100 constituents (edit kse100.txt to update membership after index reviews)
DEFAULT_SYMBOLS_FILE = Path(__file__).parent / "kse100.txt"

DDL = """
CREATE TABLE IF NOT EXISTS ohlcv (
  symbol TEXT NOT NULL, date TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL, volume INTEGER,
  PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS macro (
  series TEXT NOT NULL, date TEXT NOT NULL, value REAL,
  PRIMARY KEY (series, date)
);
"""


def upsert(con, symbol: str, df: pd.DataFrame):
    rows = [(symbol, str(ix.date() if hasattr(ix, "date") else ix),
             float(r.Open), float(r.High), float(r.Low), float(r.Close), int(r.Volume))
            for ix, r in df.iterrows()]
    con.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--symbols-file", default=str(DEFAULT_SYMBOLS_FILE))
    args = ap.parse_args()

    import psxdata  # pip install psxdata  (scrapes dps.psx.com.pk)

    symbols = [s.strip().upper() for s in Path(args.symbols_file).read_text().splitlines()
               if s.strip() and not s.startswith("#")]
    start = date.today() - timedelta(days=args.years * 365)
    end = date.today()

    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(DDL)

    failed = []
    for i, sym in enumerate(symbols, 1):
        try:
            df = psxdata.stocks(sym, start, end)
            if df is None or df.empty:
                raise ValueError("empty frame")
            df.columns = [c.title() for c in df.columns]
            n = upsert(con, sym, df[["Open", "High", "Low", "Close", "Volume"]])
            print(f"[{i}/{len(symbols)}] {sym}: {n} rows")
        except Exception as e:
            failed.append(sym)
            print(f"[{i}/{len(symbols)}] {sym}: FAILED — {e}", file=sys.stderr)
        time.sleep(1.0)  # be polite to the PSX portal

    print(f"\nDone. {len(symbols) - len(failed)} ok, {len(failed)} failed: {failed}")
    if failed:
        Path("failed_symbols.txt").write_text("\n".join(failed))
        sys.exit(1 if len(failed) > len(symbols) // 2 else 0)


if __name__ == "__main__":
    main()
