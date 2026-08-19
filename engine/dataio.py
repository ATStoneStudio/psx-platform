"""Shared data-normalization layer between the fetch scripts and the DB.
Fixes the failure modes observed in the first live run:
- psxdata returns a positional index with the date in a column -> find & use it
- feed is newest-first -> sort ascending
- ~0.4% dirty rows (zero prices, high/low inconsistent) -> clean deterministically
"""
import sqlite3
import pandas as pd

DATE_NAMES = {"date", "time", "datetime", "trade_date", "session"}


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean ascending-DatetimeIndex OHLCV frame or raise ValueError."""
    df = df.copy()
    df.columns = [str(c).strip().title() for c in df.columns]

    # --- locate the date ---
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        date_col = next((c for c in df.columns if c.lower() in DATE_NAMES), None)
        if date_col is None:
            # last resort: a column that parses as dates
            for c in df.columns:
                if df[c].dtype == object:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().mean() > 0.9:
                        date_col = c
                        break
        if date_col is None:
            raise ValueError("no date column found — refusing to store positional index")
        idx = pd.to_datetime(df[date_col], errors="coerce")
        df = df.drop(columns=[date_col])
    df.index = pd.DatetimeIndex(idx)
    df = df[df.index.notna()]

    # --- required columns ---
    missing = [c for c in ("Open", "High", "Low", "Close", "Volume") if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")

    # --- clean ---
    df = df[df["Close"] > 0]
    df = df[df["High"] > 0]
    df["Open"] = df["Open"].where(df["Open"] > 0, df["Close"])
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)   # repair inconsistent extremes
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    df["Low"] = df["Low"].where(df["Low"] > 0, df[["Open", "Close"]].min(axis=1))
    df["Volume"] = df["Volume"].fillna(0).clip(lower=0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # --- order: ascending, unique dates (keep the freshest record) ---
    df = df[~df.index.duplicated(keep="first")]  # feed is newest-first; first = freshest
    df = df.sort_index()
    if len(df) == 0:
        raise ValueError("no valid rows after cleaning")
    return df


def upsert_ohlcv(con: sqlite3.Connection, symbol: str, df: pd.DataFrame) -> int:
    rows = [(symbol, ix.strftime("%Y-%m-%d"),
             float(r.Open), float(r.High), float(r.Low), float(r.Close), int(r.Volume))
            for ix, r in df.iterrows()]
    con.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    return len(rows)


def purge_bad_rows(con: sqlite3.Connection) -> int:
    """Remove rows from the first (buggy) run whose 'date' is not YYYY-MM-DD."""
    cur = con.execute("DELETE FROM ohlcv WHERE date NOT LIKE '____-__-__'")
    con.commit()
    if cur.rowcount:
        print(f"purged {cur.rowcount} rows with invalid dates from previous run")
    return cur.rowcount


def adjust_corporate_actions(df: pd.DataFrame, lo=0.75, hi=1.30):
    """Back-adjust for splits/bonuses. PSX daily price limits (~±10%) mean a
    close-to-close ratio outside [lo, hi] is a corporate action, not a trade.
    Prices BEFORE the event are scaled by the ratio (volume inversely).
    DB keeps raw prices; adjustment is applied at read time for analysis."""
    events = []
    ratio = df["Close"] / df["Close"].shift(1)
    adj = pd.Series(1.0, index=df.index)
    for i in range(1, len(df)):
        r = float(ratio.iloc[i])
        if r < lo or r > hi:
            events.append({"date": str(df.index[i].date()), "ratio": round(r, 4)})
            adj.iloc[:i] *= r
    if events:
        df = df.copy()
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col] * adj
        df["Volume"] = (df["Volume"] / adj).round()
    return df, events


def load_symbol(con: sqlite3.Connection, symbol: str, adjust: bool = True):
    df = pd.read_sql("SELECT date, open, high, low, close, volume FROM ohlcv "
                     "WHERE symbol=? AND date LIKE '____-__-__' ORDER BY date",
                     con, params=(symbol,))
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    events = []
    if adjust and len(df) > 1:
        df, events = adjust_corporate_actions(df)
    return df, events


# ======================================================================
# v2 — audit trail, calibration state, macro series access
# ======================================================================

SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS ohlcv (
  symbol TEXT NOT NULL, date TEXT NOT NULL,
  open REAL, high REAL, low REAL, close REAL, volume INTEGER,
  PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS macro (
  series TEXT NOT NULL, date TEXT NOT NULL, value REAL,
  PRIMARY KEY (series, date)
);
-- Every signal the engine publishes, frozen at publish time (audit trail).
-- outcome: NULL while open, then TP1 | TP2 | SL | EXPIRED | NOT_FILLED | VOID_CA
CREATE TABLE IF NOT EXISTS predictions_history (
  symbol TEXT NOT NULL, signal_date TEXT NOT NULL,
  verdict TEXT, composite REAL, confidence INTEGER, horizon TEXT,
  entry REAL, stop_loss REAL, tp1 REAL, tp2 REAL,
  setup_valid INTEGER, scores_json TEXT,
  outcome TEXT, fill_date TEXT, outcome_date TEXT,
  days_to_outcome INTEGER, return_pct REAL,
  source TEXT DEFAULT 'live',   -- 'live' = published pre-close; 'backfill' = simulated replay
  PRIMARY KEY (symbol, signal_date)
);
-- Key/value store for calibration state (current weights, adjustment log).
CREATE TABLE IF NOT EXISTS calibration (
  key TEXT PRIMARY KEY, value TEXT
);
"""


def ensure_schema(con: sqlite3.Connection) -> None:
    """Idempotent: safe to call at the top of every run."""
    con.executescript(SCHEMA_V2)
    cols = [r[1] for r in con.execute("PRAGMA table_info(predictions_history)")]
    if "source" not in cols:  # migrate tables created by an earlier v2 draft
        con.execute("ALTER TABLE predictions_history ADD COLUMN source TEXT DEFAULT 'live'")
    con.commit()


def load_macro_series(con: sqlite3.Connection, name: str) -> pd.Series:
    """Full stored history of one macro series as a date-indexed float Series."""
    rows = con.execute(
        "SELECT date, value FROM macro WHERE series=? AND date LIKE '____-__-__' "
        "ORDER BY date", (name,)).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series([float(v) for _, v in rows],
                     index=pd.to_datetime([d for d, _ in rows]))


def load_macro_context(con: sqlite3.Connection) -> dict:
    """Context dict consumed by engine.analysis.analyze(): brent/usdpkr/kse100."""
    return {name: load_macro_series(con, name)
            for name in ("brent", "usdpkr", "kse100")}


def get_calibration(con: sqlite3.Connection, key: str, default=None):
    import json
    row = con.execute("SELECT value FROM calibration WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_calibration(con: sqlite3.Connection, key: str, value) -> None:
    import json
    con.execute("INSERT OR REPLACE INTO calibration VALUES (?,?)", (key, json.dumps(value)))
    con.commit()