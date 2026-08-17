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
