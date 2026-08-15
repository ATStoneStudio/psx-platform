"""Candlestick pattern detection on the last few bars. Pure rules, no libraries."""
import pandas as pd

MEANING = {
    "Hammer": "Potential bullish reversal after a decline — buyers rejected lower prices.",
    "Shooting Star": "Potential bearish reversal — sellers rejected higher prices.",
    "Doji": "Indecision; watch for confirmation next session.",
    "Bullish Engulfing": "Buyers overwhelmed sellers — bullish reversal signal.",
    "Bearish Engulfing": "Sellers overwhelmed buyers — bearish reversal signal.",
    "Bullish Harami": "Selling pressure pausing — possible bullish turn.",
    "Bearish Harami": "Buying pressure pausing — possible bearish turn.",
    "Morning Star": "Strong 3-bar bullish reversal pattern.",
    "Evening Star": "Strong 3-bar bearish reversal pattern.",
    "Bullish Pin Bar": "Long lower wick rejection — bullish.",
    "Bearish Pin Bar": "Long upper wick rejection — bearish.",
    "Inside Bar": "Consolidation / breakout setup.",
    "Outside Bar": "Volatility expansion — direction of close matters.",
}


def _bar(row):
    o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
    body = abs(c - o)
    rng = h - l if h > l else 1e-9
    return o, h, l, c, body, rng


def detect(df: pd.DataFrame, n_last: int = 3):
    out = []
    d = df.tail(n_last + 2)
    for k in range(2, len(d)):
        r, r1, r2 = d.iloc[k], d.iloc[k - 1], d.iloc[k - 2]
        date = str(d.index[k].date())
        o, h, l, c, body, rng = _bar(r)
        o1, h1, l1, c1, body1, rng1 = _bar(r1)
        up, dn = h - max(o, c), min(o, c) - l
        found = []
        if body <= 0.1 * rng:
            found.append("Doji")
        if dn >= 2 * body and up <= 0.3 * rng and body > 0:
            found.append("Hammer" if c1 < o1 else "Bullish Pin Bar")
        if up >= 2 * body and dn <= 0.3 * rng and body > 0:
            found.append("Shooting Star" if c1 > o1 else "Bearish Pin Bar")
        if c > o and c1 < o1 and c >= max(o1, c1) and o <= min(o1, c1) and body > body1:
            found.append("Bullish Engulfing")
        if c < o and c1 > o1 and o >= max(o1, c1) and c <= min(o1, c1) and body > body1:
            found.append("Bearish Engulfing")
        if body1 > 0 and max(o, c) < max(o1, c1) and min(o, c) > min(o1, c1) and body < 0.6 * body1:
            found.append("Bullish Harami" if c1 < o1 else "Bearish Harami")
        if h < h1 and l > l1:
            found.append("Inside Bar")
        if h > h1 and l < l1:
            found.append("Outside Bar")
        # 3-bar stars
        o2, h2, l2, c2, body2, rng2 = _bar(r2)
        if c2 < o2 and body1 < 0.4 * body2 and c > o and c > (o2 + c2) / 2:
            found.append("Morning Star")
        if c2 > o2 and body1 < 0.4 * body2 and c < o and c < (o2 + c2) / 2:
            found.append("Evening Star")
        for f in dict.fromkeys(found):
            out.append({"date": date, "pattern": f, "meaning": MEANING[f]})
    return out
