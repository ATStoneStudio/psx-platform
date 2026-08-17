"""ONE-TIME backfill: ~5 years of EOD OHLCV for KSE-100 constituents -> data/psx.sqlite
Run on GitHub Actions (open internet) — uses the `psxdata` library.

Usage:  python backfill.py [--years 5] [--symbols-file kse100.txt]
"""
import argparse, sqlite3, sys, time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from engine.dataio import normalize_frame, upsert_ohlcv, purge_bad_rows  # noqa: E402

DB = Path(__file__).parent / "data" / "psx.sqlite"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--symbols-file", default=str(DEFAULT_SYMBOLS_FILE))
    args = ap.parse_args()

    import psxdata  # pip install psxdata

    symbols = [s.strip().upper() for s in Path(args.symbols_file).read_text().splitlines()
               if s.strip() and not s.startswith("#")]
    start = date.today() - timedelta(days=args.years * 365)
    end = date.today()

    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(DDL)
    purge_bad_rows(con)

    failed = []
    for i, sym in enumerate(symbols, 1):
        try:
            raw = psxdata.stocks(sym, start, end)
            if raw is None or len(raw) == 0:
                raise ValueError("empty frame")
            df = normalize_frame(raw)
            n = upsert_ohlcv(con, sym, df)
            print(f"[{i}/{len(symbols)}] {sym}: {n} rows "
                  f"({df.index[0].date()} -> {df.index[-1].date()})")
        except Exception as e:
            failed.append(sym)
            print(f"[{i}/{len(symbols)}] {sym}: FAILED — {e}", file=sys.stderr)
        time.sleep(1.0)

    print(f"\nDone. {len(symbols) - len(failed)} ok, {len(failed)} failed: {failed}")
    Path("failed_symbols.txt").write_text("\n".join(failed))
    if len(failed) > len(symbols) // 2:
        sys.exit(1)


if __name__ == "__main__":
    main()
