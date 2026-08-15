"""PSX analysis engine — orchestrates all deterministic analysis for one stock.
Outputs a JSON dict covering every computable section of the 14-section report,
plus the project's pivot-based order table. Zero LLM calls: pure math + templates.
"""
import numpy as np
import pandas as pd

from . import indicators as ind
from . import structure as st
from . import candles as cd

WEIGHTS = {"trend": 0.25, "momentum": 0.20, "volume": 0.15,
           "structure": 0.20, "volatility": 0.10, "mtf": 0.10}


def _tf_trend(df: pd.DataFrame) -> str:
    if len(df) < 30:
        return "Insufficient data"
    c = df["Close"]
    e20, e50 = ind.ema(c, 20).iloc[-1], ind.ema(c, 50).iloc[-1]
    last = c.iloc[-1]
    if last > e20 > e50:
        return "Bullish"
    if last < e20 < e50:
        return "Bearish"
    return "Sideways"


def _clip10(x):
    return float(np.clip(x, 0, 10))


def analyze(symbol: str, df: pd.DataFrame, context: dict | None = None) -> dict:
    """df: daily OHLCV, DatetimeIndex ascending, >= 260 rows recommended.
    context: optional dict with 'usdpkr','brent','kse100' DataFrames/last values."""
    df = df.sort_index()
    c = df["Close"]
    last = df.iloc[-1]
    close = float(last["Close"])
    prev = df.iloc[-2]

    # ---------- indicators ----------
    e = {n: ind.ema(c, n) for n in (20, 50, 100, 200)}
    rsi = ind.rsi(c)
    macd_l, macd_s, macd_h = ind.macd(c)
    adx, pdi, mdi = ind.adx(df)
    atr = ind.atr(df)
    bb_u, bb_m, bb_l = ind.bollinger(c)
    sk, sd_ = ind.stoch_rsi(c)
    cci = ind.cci(df)
    conv, base, span_a, span_b = ind.ichimoku(df)
    sar = ind.psar(df)
    obv = ind.obv(df)
    mfi = ind.mfi(df)
    cmf = ind.cmf(df)
    vwap = ind.rolling_vwap(df)

    def v(s, i=-1):
        x = s.iloc[i]
        return None if pd.isna(x) else round(float(x), 2)

    golden_cross = bool(e[50].iloc[-1] > e[200].iloc[-1] and e[50].iloc[-20] <= e[200].iloc[-20]) if len(df) > 220 else False
    death_cross = bool(e[50].iloc[-1] < e[200].iloc[-1] and e[50].iloc[-20] >= e[200].iloc[-20]) if len(df) > 220 else False

    # ---------- structure ----------
    atr_pct = float(atr.iloc[-1] / close * 100) if not pd.isna(atr.iloc[-1]) else 3.0
    zz_pct = max(3.0, min(8.0, 2.5 * atr_pct))
    pivots = st.zigzag(df.tail(500), pct=zz_pct)
    labels, structure_trend = st.classify_structure(pivots)
    sup, res = st.support_resistance(df, pivots, close)
    events = st.bos_choch(pivots, c)
    fvg = st.fair_value_gaps(df)
    obs = st.order_blocks(df, atr)
    elliott = st.elliott_estimate(pivots, close)
    patterns = cd.detect(df)

    # swing hi/lo for fib (last major leg)
    lookback = df.tail(180)
    swing_hi, swing_lo = float(lookback["High"].max()), float(lookback["Low"].min())
    fib = ind.fib_levels(swing_hi, swing_lo)

    # ---------- pivots (project methodology: previous day H/L/C) ----------
    piv = ind.pivots_classic(float(prev["High"]), float(prev["Low"]), float(prev["Close"]))
    fib382 = float(prev["Close"]) - 0.382 * (float(prev["High"]) - float(prev["Low"]))
    fib618 = float(prev["Close"]) - 0.618 * (float(prev["High"]) - float(prev["Low"]))
    primary_buy = round(max(piv["S1"], fib382), 2)
    secondary_buy = round(max(piv["S2"], fib618), 2)
    stop_loss = round(piv["S2"] * 0.98, 2)
    tp1, tp2 = round(piv["R1"], 2), round(piv["R2"], 2)
    risk = primary_buy - stop_loss
    reward = tp2 - primary_buy
    rr = round(reward / risk, 2) if risk > 0 else None
    setup_valid = rr is not None and rr >= 2.0

    # ---------- multi-timeframe ----------
    wk = df.resample("W").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    mo = df.resample("ME").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    mtf = {"Monthly": _tf_trend(mo), "Weekly": _tf_trend(wk), "Daily": _tf_trend(df),
           "4H": "N/A (EOD data only)", "1H": "N/A (EOD data only)", "15M": "N/A (EOD data only)"}

    # ---------- volume read ----------
    vol20 = float(df["Volume"].tail(20).mean())
    vol_ratio = float(last["Volume"] / vol20) if vol20 > 0 else 1.0
    obv_rising = bool(obv.iloc[-1] > obv.iloc[-20]) if len(obv) > 20 else False
    cmf_last = v(cmf) or 0
    price_flat_20 = abs(close / float(c.iloc[-20]) - 1) < 0.03 if len(c) > 20 else False
    if obv_rising and cmf_last > 0.05:
        vol_read = "Accumulation (rising OBV + positive money flow — consistent with institutional buying)"
    elif not obv_rising and cmf_last < -0.05:
        vol_read = "Distribution (falling OBV + negative money flow)"
    elif obv_rising and price_flat_20:
        vol_read = "Quiet accumulation (OBV rising while price consolidates)"
    else:
        vol_read = "Neutral / retail-dominated churn"

    # ---------- scores ----------
    s_trend = 5.0
    s_trend += 2 if close > (e[200].iloc[-1] or close) else -2
    s_trend += 1.5 if close > (e[50].iloc[-1] or close) else -1.5
    s_trend += 1.5 if structure_trend == "Bullish" else (-1.5 if structure_trend == "Bearish" else 0)
    s_trend = _clip10(s_trend)

    r_last = v(rsi) or 50
    s_mom = 5.0
    s_mom += 1.5 if macd_h.iloc[-1] > 0 else -1.5
    s_mom += 1.5 if 50 <= r_last <= 70 else (2 if r_last < 35 else (-2 if r_last > 75 else -0.5 if r_last < 50 else 0))
    s_mom += 1 if (v(adx) or 0) > 25 and pdi.iloc[-1] > mdi.iloc[-1] else (-1 if (v(adx) or 0) > 25 else 0)
    s_mom = _clip10(s_mom)

    s_vol = 5.0 + (2 if obv_rising else -1.5) + (1.5 if cmf_last > 0.05 else (-1.5 if cmf_last < -0.05 else 0)) \
        + (1 if (v(mfi) or 50) < 30 else (-1 if (v(mfi) or 50) > 80 else 0))
    s_vol = _clip10(s_vol)

    near_sup = bool(sup and (close - sup[0][0]) / close < 0.03)
    near_res = bool(res and (res[0][0] - close) / close < 0.03)
    gp_lo, gp_hi = fib["golden_pocket"]
    in_gp = gp_lo <= close <= gp_hi
    s_struct = 5.0 + (2 if near_sup else 0) + (-2 if near_res else 0) + (1.5 if in_gp else 0) \
        + (1 if events["bos"] and "Bullish" in events["bos"] else 0) \
        + (-1.5 if events["choch"] and "Bearish" in events["choch"] else 0)
    s_struct = _clip10(s_struct)

    s_vola = _clip10(7.5 - max(0.0, atr_pct - 2) * 1.2)  # calmer = safer entries
    tf_scores = {"Bullish": 1, "Sideways": 0.5, "Bearish": 0}
    s_mtf = _clip10(10 * np.mean([tf_scores.get(mtf[t], 0.5) for t in ("Monthly", "Weekly", "Daily")]))

    composite = round(sum(WEIGHTS[k] * s for k, s in
                          [("trend", s_trend), ("momentum", s_mom), ("volume", s_vol),
                           ("structure", s_struct), ("volatility", s_vola), ("mtf", s_mtf)]), 2)
    if composite >= 7.5:
        verdict = "Strong Buy"
    elif composite >= 6.2:
        verdict = "Buy"
    elif composite >= 4.5:
        verdict = "Hold"
    elif composite >= 3.2:
        verdict = "Sell"
    else:
        verdict = "Strong Sell"
    confidence = int(round(50 + abs(composite - 5) * 9))

    # ---------- targets ----------
    ext = fib["extension"]
    targets = {"short_term": tp1, "swing": tp2,
               "3m": round(min(ext["1.272"], swing_hi * 1.05), 2) if close < swing_hi else round(ext["1.272"], 2),
               "6m": round(ext["1.414"], 2), "1y": round(ext["1.618"], 2),
               "long_term_3_5y": round(ext["2.618"], 2)}

    # ---------- narrative (templates, zero tokens at runtime) ----------
    bull_pts, bear_pts = [], []
    if close > (e[200].iloc[-1] or 0):
        bull_pts.append("price above 200-EMA (long-term uptrend intact)")
    else:
        bear_pts.append("price below 200-EMA (long-term trend broken)")
    if macd_h.iloc[-1] > 0:
        bull_pts.append("MACD histogram positive")
    else:
        bear_pts.append("MACD histogram negative")
    if obv_rising:
        bull_pts.append("OBV rising — volume supports the move")
    else:
        bear_pts.append("OBV flat/falling — weak volume backing")
    if in_gp:
        bull_pts.append("price sitting in the Fibonacci golden pocket (0.618–0.65)")
    if near_res:
        bear_pts.append("price within 3% of clustered resistance")
    if r_last > 75:
        bear_pts.append(f"RSI overbought at {r_last:.0f}")
    if r_last < 30:
        bull_pts.append(f"RSI oversold at {r_last:.0f}")
    if golden_cross:
        bull_pts.append("recent golden cross (50>200 EMA)")
    if death_cross:
        bear_pts.append("recent death cross (50<200 EMA)")

    summary = (f"{symbol}: {structure_trend} structure on daily; composite score {composite}/10 → {verdict} "
               f"({confidence}% confidence). Bullish: {'; '.join(bull_pts) or 'none'}. "
               f"Bearish: {'; '.join(bear_pts) or 'none'}. "
               + ("Setup meets the 1:2 R:R rule. " if setup_valid else
                  f"Setup REJECTED under the 1:2 R:R rule (R:R = 1:{rr}). " if rr else "")
               + "Place limit orders manually in your broker terminal; these are probabilistic levels, not guarantees.")

    return {
        "symbol": symbol, "as_of": str(df.index[-1].date()),
        "close": close, "volume": int(last["Volume"]),
        "market_structure": structure_trend,
        "verdict": {"call": verdict, "confidence_pct": confidence, "composite_score": composite,
                    "scores": {"trend": round(s_trend, 1), "momentum": round(s_mom, 1),
                               "volume": round(s_vol, 1), "structure": round(s_struct, 1),
                               "volatility": round(s_vola, 1), "mtf_alignment": round(s_mtf, 1)},
                    "weights": WEIGHTS},
        "order_table": {"primary_buy_limit": primary_buy, "secondary_buy_dip": secondary_buy,
                        "stop_loss": stop_loss, "take_profit_1": tp1, "take_profit_2": tp2,
                        "risk_reward": f"1 : {rr}" if rr else "n/a", "setup_valid_rr_rule": setup_valid},
        "pivots_classic": {k: round(vv, 2) for k, vv in piv.items()},
        "fibonacci": {"swing_high": swing_hi, "swing_low": swing_lo,
                      "retracement": {k: round(vv, 2) for k, vv in fib["retracement"].items()},
                      "extension": {k: round(vv, 2) for k, vv in fib["extension"].items()},
                      "golden_pocket": [round(gp_lo, 2), round(gp_hi, 2)], "price_in_golden_pocket": in_gp},
        "indicators": {"rsi14": v(rsi), "macd": v(macd_l), "macd_signal": v(macd_s), "macd_hist": v(macd_h),
                       "adx": v(adx), "di_plus": v(pdi), "di_minus": v(mdi), "atr14": v(atr),
                       "atr_pct": round(atr_pct, 2), "cci20": v(cci),
                       "stoch_rsi_k": v(sk), "stoch_rsi_d": v(sd_),
                       "bb_upper": v(bb_u), "bb_mid": v(bb_m), "bb_lower": v(bb_l),
                       "ema20": v(e[20]), "ema50": v(e[50]), "ema100": v(e[100]), "ema200": v(e[200]),
                       "golden_cross_recent": golden_cross, "death_cross_recent": death_cross,
                       "ichimoku_conversion": v(conv), "ichimoku_base": v(base),
                       "ichimoku_span_a": v(span_a), "ichimoku_span_b": v(span_b),
                       "psar": v(sar), "obv_rising_20d": obv_rising, "mfi14": v(mfi),
                       "cmf20": cmf_last, "vwap20": v(vwap)},
        "volume_analysis": {"read": vol_read, "volume_vs_20d_avg": round(vol_ratio, 2)},
        "support_levels": [{"price": round(p, 2), "touches": t} for p, t in sup[:4]],
        "resistance_levels": [{"price": round(p, 2), "touches": t} for p, t in res[:4]],
        "structure_events": events, "fair_value_gaps": fvg, "order_blocks": obs,
        "elliott_wave": elliott, "candlestick_patterns": patterns,
        "multi_timeframe": mtf, "price_targets": targets,
        "unavailable_in_v1": ["intraday timeframes (4H/1H/15M)", "fundamentals (phase 2)",
                              "news & social sentiment", "insider activity"],
        "summary": summary,
        "disclaimer": "Automated technical analysis for information only — not investment advice. "
                      "Execute all orders manually in your broker terminal.",
    }
