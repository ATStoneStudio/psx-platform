"""Pure pandas/numpy technical indicators — no external TA dependency.
All formulas are the standard textbook definitions (Wilder smoothing where applicable)
so results are auditable and reproducible anywhere.
Input: DataFrame with columns Open, High, Low, Close, Volume, DatetimeIndex ascending.
"""
import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna(), np.nan)


def macd(close: pd.Series, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    signal = line.ewm(span=sig, adjust=False).mean()
    return line, signal, line - signal


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift()
    return pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                      (df["Low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr_s = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr_s
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr_s
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), pdi, mdi


def bollinger(close: pd.Series, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def stochastic(df: pd.DataFrame, n=14, d=3):
    lo, hi = df["Low"].rolling(n).min(), df["High"].rolling(n).max()
    k = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(d).mean()


def stoch_rsi(close: pd.Series, n=14, k=3, d=3):
    r = rsi(close, n)
    lo, hi = r.rolling(n).min(), r.rolling(n).max()
    sk = (100 * (r - lo) / (hi - lo).replace(0, np.nan)).rolling(k).mean()
    return sk, sk.rolling(d).mean()


def cci(df: pd.DataFrame, n=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def ichimoku(df: pd.DataFrame):
    conv = (df["High"].rolling(9).max() + df["Low"].rolling(9).min()) / 2
    base = (df["High"].rolling(26).max() + df["Low"].rolling(26).min()) / 2
    span_a = ((conv + base) / 2).shift(26)
    span_b = ((df["High"].rolling(52).max() + df["Low"].rolling(52).min()) / 2).shift(26)
    return conv, base, span_a, span_b


def psar(df: pd.DataFrame, af0=0.02, af_step=0.02, af_max=0.2) -> pd.Series:
    h, l = df["High"].to_numpy(), df["Low"].to_numpy()
    n = len(df)
    out = np.full(n, np.nan)
    if n < 3:
        return pd.Series(out, index=df.index)
    up = h[1] > h[0]
    ep = h[1] if up else l[1]
    sar, af = (l[0], af0)
    if not up:
        sar = h[0]
    for i in range(2, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], af0
            elif h[i] > ep:
                ep, af = h[i], min(af + af_step, af_max)
        else:
            sar = max(sar, h[i - 1], h[i - 2])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], af0
            elif l[i] < ep:
                ep, af = l[i], min(af + af_step, af_max)
        out[i] = sar
    return pd.Series(out, index=df.index)


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["Close"].diff()).fillna(0)
    return (sign * df["Volume"]).cumsum()


def mfi(df: pd.DataFrame, n=14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos = mf.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def cmf(df: pd.DataFrame, n=20) -> pd.Series:
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    mult = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    return (mult * df["Volume"]).rolling(n).sum() / df["Volume"].rolling(n).sum()


def rolling_vwap(df: pd.DataFrame, n=20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).rolling(n).sum() / df["Volume"].rolling(n).sum()


def pivots_classic(H: float, L: float, C: float) -> dict:
    P = (H + L + C) / 3
    return {"P": P, "S1": 2 * P - H, "S2": P - (H - L), "S3": L - 2 * (H - P),
            "R1": 2 * P - L, "R2": P + (H - L), "R3": H + 2 * (P - L)}


def pivots_fib(H: float, L: float, C: float) -> dict:
    P = (H + L + C) / 3
    r = H - L
    return {"P": P, "S1": P - 0.382 * r, "S2": P - 0.618 * r, "S3": P - r,
            "R1": P + 0.382 * r, "R2": P + 0.618 * r, "R3": P + r}


def fib_levels(swing_high: float, swing_low: float) -> dict:
    r = swing_high - swing_low
    ret = {f"{lv:.3f}": swing_high - lv * r for lv in (0.236, 0.382, 0.5, 0.618, 0.786)}
    ext = {f"{lv:.3f}": swing_high + (lv - 1) * r for lv in (1.272, 1.414, 1.618, 2.618)}
    return {"retracement": ret, "extension": ext,
            "golden_pocket": (swing_high - 0.65 * r, swing_high - 0.618 * r)}
