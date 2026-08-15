"""Market structure: ZigZag swings, HH/HL/LH/LL, S/R clustering, BOS/CHOCH,
order blocks, fair value gaps, and a rule-based Elliott wave estimate."""
import numpy as np
import pandas as pd


def zigzag(df: pd.DataFrame, pct: float = 4.0):
    """Return list of swing pivots [(idx, price, 'H'|'L')] using % reversal threshold."""
    closes = df["Close"].to_numpy()
    highs, lows = df["High"].to_numpy(), df["Low"].to_numpy()
    n = len(df)
    if n < 5:
        return []
    piv = []
    trend = 0  # 1 up, -1 down
    last_ext_i, last_ext = 0, closes[0]
    for i in range(1, n):
        if trend >= 0:
            if highs[i] > last_ext:
                last_ext, last_ext_i = highs[i], i
                trend = 1
            elif last_ext > 0 and (last_ext - lows[i]) / last_ext * 100 >= pct:
                piv.append((last_ext_i, last_ext, "H"))
                trend, last_ext, last_ext_i = -1, lows[i], i
        if trend <= 0:
            if lows[i] < last_ext:
                last_ext, last_ext_i = lows[i], i
            elif last_ext > 0 and (highs[i] - last_ext) / last_ext * 100 >= pct:
                piv.append((last_ext_i, last_ext, "L"))
                trend, last_ext, last_ext_i = 1, highs[i], i
    piv.append((last_ext_i, last_ext, "H" if trend == 1 else "L"))
    return piv


def classify_structure(pivots):
    """Label consecutive same-type swings HH/HL/LH/LL and derive trend."""
    labels = []
    lastH = lastL = None
    for i, p, t in pivots:
        if t == "H":
            labels.append((i, p, "HH" if lastH is not None and p > lastH else ("LH" if lastH is not None else "H")))
            lastH = p
        else:
            labels.append((i, p, "HL" if lastL is not None and p > lastL else ("LL" if lastL is not None else "L")))
            lastL = p
    recent = [lb for _, _, lb in labels[-4:]]
    score = sum(1 for x in recent if x in ("HH", "HL")) - sum(1 for x in recent if x in ("LH", "LL"))
    trend = "Bullish" if score >= 2 else "Bearish" if score <= -2 else "Sideways"
    return labels, trend


def support_resistance(df: pd.DataFrame, pivots, close: float, tol_pct=1.5):
    """Cluster swing prices into S/R zones; return sorted supports (below) & resistances (above)."""
    prices = sorted(p for _, p, _ in pivots)
    zones = []
    for p in prices:
        if zones and abs(p - zones[-1][-1]) / p * 100 <= tol_pct:
            zones[-1].append(p)
        else:
            zones.append([p])
    levels = [(float(np.mean(z)), len(z)) for z in zones]
    sup = sorted([lv for lv in levels if lv[0] < close], key=lambda x: -x[0])
    res = sorted([lv for lv in levels if lv[0] >= close], key=lambda x: x[0])
    return sup, res


def bos_choch(pivots, closes: pd.Series):
    """Break of Structure / Change of Character from the last confirmed swings."""
    hs = [(i, p) for i, p, t in pivots if t == "H"]
    ls = [(i, p) for i, p, t in pivots if t == "L"]
    if len(hs) < 2 or len(ls) < 2:
        return {"bos": None, "choch": None}
    c = closes.iloc[-1]
    events = {"bos": None, "choch": None}
    # BOS: close beyond most recent swing extreme in trend direction
    if c > hs[-1][1]:
        events["bos"] = "Bullish BOS (close above last swing high)"
    elif c < ls[-1][1]:
        events["bos"] = "Bearish BOS (close below last swing low)"
    # CHOCH: prior structure was making HH/HL but last swing low broken (or inverse)
    if hs[-1][1] > hs[-2][1] and c < ls[-1][1]:
        events["choch"] = "Bearish CHOCH (uptrend structure low broken)"
    elif ls[-1][1] < ls[-2][1] and c > hs[-1][1]:
        events["choch"] = "Bullish CHOCH (downtrend structure high broken)"
    return events


def fair_value_gaps(df: pd.DataFrame, lookback=60):
    """3-candle FVGs still unfilled. Bullish: low[i] > high[i-2]."""
    d = df.tail(lookback)
    out = []
    h, l = d["High"].to_numpy(), d["Low"].to_numpy()
    idx = d.index
    last_close = d["Close"].iloc[-1]
    for i in range(2, len(d)):
        if l[i] > h[i - 2]:  # bullish gap
            top, bot = l[i], h[i - 2]
            if d["Low"].iloc[i:].min() > bot:  # unfilled
                out.append({"type": "bullish", "zone": (round(bot, 2), round(top, 2)),
                            "date": str(idx[i - 1].date())})
        elif h[i] < l[i - 2]:  # bearish gap
            top, bot = l[i - 2], h[i]
            if d["High"].iloc[i:].max() < top:
                out.append({"type": "bearish", "zone": (round(bot, 2), round(top, 2)),
                            "date": str(idx[i - 1].date())})
    return out[-4:]


def order_blocks(df: pd.DataFrame, atr_s: pd.Series, lookback=90):
    """Last opposite-colour candle before an impulsive move (>1.5 ATR over next 3 bars)."""
    d = df.tail(lookback)
    a = atr_s.reindex(d.index)
    out = []
    for i in range(1, len(d) - 3):
        o, c = d["Open"].iloc[i], d["Close"].iloc[i]
        move = d["Close"].iloc[i + 3] - c
        if pd.isna(a.iloc[i]) or a.iloc[i] == 0:
            continue
        if c < o and move > 1.5 * a.iloc[i]:   # bearish candle then rally = bullish OB
            out.append({"type": "bullish", "zone": (round(d["Low"].iloc[i], 2), round(max(o, c), 2)),
                        "date": str(d.index[i].date())})
        elif c > o and move < -1.5 * a.iloc[i]:
            out.append({"type": "bearish", "zone": (round(min(o, c), 2), round(d["High"].iloc[i], 2)),
                        "date": str(d.index[i].date())})
    return out[-3:]


def elliott_estimate(pivots, close: float):
    """Heuristic wave count from last ZigZag legs. Confidence is honest: capped at Medium."""
    if len(pivots) < 5:
        return {"count": "Insufficient swings for a wave count", "phase": "Unknown", "confidence": "Low"}
    pts = [p for _, p, _ in pivots[-7:]]
    legs = np.diff(pts)
    ups = sum(1 for x in legs if x > 0)
    downs = len(legs) - ups
    rising = pts[-1] > pts[0]
    # crude impulse test: alternating legs, net direction, wave3 not shortest
    if rising and ups >= 3:
        up_legs = [x for x in legs if x > 0]
        w3_ok = len(up_legs) >= 2 and max(up_legs) != up_legs[0]
        phase = "Impulse (wave 3 or 5 of an advance)" if w3_ok else "Late impulse / possible wave 5"
        conf = "Medium" if w3_ok else "Low"
    elif not rising and downs >= 2:
        phase = "Correction (likely ABC decline)"
        conf = "Medium" if downs in (2, 3) else "Low"
    else:
        phase = "Overlapping / corrective structure"
        conf = "Low"
    return {"count": f"{ups} up-legs / {downs} down-legs in last {len(legs)} swings",
            "phase": phase, "confidence": conf,
            "note": "Rule-based estimate from ZigZag swings — Elliott counts are inherently subjective."}
