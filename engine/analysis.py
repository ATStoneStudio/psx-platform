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
# Calibration bounds: auto-adjusted weights may never leave this band,
# so a bad calibration run cannot zero-out or dominate a component.
WEIGHT_FLOOR, WEIGHT_CAP = 0.10, 0.35

HORIZON_SHORT = "Short-Term (1-3 days)"
HORIZON_SWING = "Swing (2-4 weeks)"
HORIZON_LONG = "Long-Term (3-12 months)"

# KSE-100 constituents -> macro sector buckets (used only for macro tilts;
# unknown symbols fall back to "Other" = no sector adjustment).
SECTORS = {
    # Oil & Gas exploration (revenue tracks Brent, USD-linked)
    "OGDC": "E&P", "PPL": "E&P", "POL": "E&P", "MARI": "E&P",
    # Oil marketing / energy retail (inventory gains when oil rises)
    "PSO": "OMC", "APL": "OMC", "WAFI": "OMC",
    "ATRL": "Refinery", "NRL": "Refinery", "CNERGY": "Refinery",
    "SNGP": "Gas Utility", "SSGC": "Gas Utility",
    # Power (fuel nominally pass-through, but high oil strains circular debt)
    "HUBC": "Power", "KEL": "Power", "KAPCO": "Power", "NPL": "Power",
    "NCPL": "Power", "LPL": "Power", "PKGP": "Power",
    # Cement (imported coal/energy heavy — hurt by high oil & weak PKR)
    "LUCK": "Cement", "DGKC": "Cement", "MLCF": "Cement", "FCCL": "Cement",
    "KOHC": "Cement", "CHCC": "Cement", "PIOC": "Cement", "ACPL": "Cement",
    "GWLC": "Cement",
    # Fertilizer / chemicals / consumer
    "ENGROH": "Conglomerate", "EFERT": "Fertilizer", "FFC": "Fertilizer",
    "FATIMA": "Fertilizer", "EPCL": "Chemical", "LOTCHEM": "Chemical",
    "LCI": "Chemical", "COLG": "Consumer",
    # Banks & financials (rate plays; treated macro-neutral vs oil)
    "HBL": "Bank", "UBL": "Bank", "MCB": "Bank", "NBP": "Bank",
    "BAFL": "Bank", "BAHL": "Bank", "MEBL": "Bank", "AKBL": "Bank",
    "BOP": "Bank", "FABL": "Bank", "HMB": "Bank", "JSBL": "Bank",
    "SCBPL": "Bank", "PSX": "Financial", "PGLC": "Financial", "HGFA": "Financial",
    # Technology & telecom (exporters gain from weak PKR)
    "TRG": "Tech", "SYS": "Tech", "NETSOL": "Tech", "AVN": "Tech",
    "AIRLINK": "Tech", "PTC": "Telecom", "TELE": "Telecom",
    # Autos & allied (imported CKD kits — hurt by weak PKR & high oil)
    "INDU": "Auto", "HCAR": "Auto", "MTL": "Auto", "AGTL": "Auto",
    "THALL": "Auto", "SAZEW": "Auto", "GADT": "Auto",
    "PAEL": "Electronics", "WAVES": "Electronics",
    # Steel (imported scrap/HRC — PKR-sensitive)
    "ISL": "Steel", "ASTL": "Steel", "MUGHAL": "Steel", "INIL": "Steel",
    # Textiles (exporters — gain from weak PKR)
    "NML": "Textile", "NCL": "Textile", "GATM": "Textile", "ILP": "Textile",
    "KTML": "Textile", "BNWM": "Textile", "MEHT": "Textile", "YOUW": "Textile",
    # Food / pharma / other
    "FCEPL": "Food", "UNITY": "Food", "NATF": "Food", "FFL": "Food",
    "SHFA": "Healthcare", "SEARL": "Pharma", "GLAXO": "Pharma", "ABOT": "Pharma",
    "HINOON": "Pharma", "AGP": "Pharma", "FEROZ": "Pharma",
    "PAKT": "Tobacco", "PIBTL": "Logistics", "SITC": "Other",
}

# Sector groups for the macro tilts below
_OIL_WINNERS = {"E&P", "Refinery", "OMC"}
_OIL_LOSERS = {"Cement", "Auto", "Power", "Electronics"}
_PKR_WINNERS = {"Textile", "Tech", "E&P"}          # exporters / USD-linked revenue
_PKR_LOSERS = {"Auto", "Cement", "Steel", "Pharma", "Electronics"}  # import-cost heavy


def _trend_pct(series, n=20):
    """% change over the last n observations; None if history is too thin."""
    s = series.dropna() if series is not None else None
    if s is None or len(s) < max(5, n // 3):
        return None
    base = float(s.iloc[-n]) if len(s) >= n else float(s.iloc[0])
    return (float(s.iloc[-1]) / base - 1) * 100 if base else None


def macro_adjust(symbol: str, context: dict | None) -> dict:
    """Sector-aware macro tilt on the composite score, clipped to [-1.5, +1.5].
    Degrades gracefully: any missing series simply contributes nothing."""
    sector = SECTORS.get(symbol, "Other")
    out = {"sector": sector, "brent_20d_pct": None, "usdpkr_20d_pct": None,
           "kse100_vs_ema50_pct": None, "regime": "Unknown",
           "adjustment": 0.0, "notes": []}
    if not context:
        out["notes"].append("no macro context supplied — no adjustment")
        return out
    delta = 0.0

    b = _trend_pct(context.get("brent"))
    if b is not None:
        out["brent_20d_pct"] = round(b, 2)
        if b >= 5:
            if sector in _OIL_WINNERS:
                delta += 0.6; out["notes"].append(f"Brent +{b:.1f}%/20d — tailwind for {sector}")
            elif sector in _OIL_LOSERS:
                delta -= 0.4; out["notes"].append(f"Brent +{b:.1f}%/20d — cost headwind for {sector}")
        elif b <= -5:
            if sector in _OIL_WINNERS:
                delta -= 0.6; out["notes"].append(f"Brent {b:.1f}%/20d — headwind for {sector}")
            elif sector in _OIL_LOSERS:
                delta += 0.4; out["notes"].append(f"Brent {b:.1f}%/20d — cost relief for {sector}")

    u = _trend_pct(context.get("usdpkr"))
    if u is not None:
        out["usdpkr_20d_pct"] = round(u, 2)
        if u >= 2:  # PKR depreciating
            if sector in _PKR_WINNERS:
                delta += 0.4; out["notes"].append(f"PKR down {u:.1f}%/20d — exporter/USD-revenue tailwind")
            elif sector in _PKR_LOSERS:
                delta -= 0.4; out["notes"].append(f"PKR down {u:.1f}%/20d — import-cost headwind")
        elif u <= -2:
            if sector in _PKR_LOSERS:
                delta += 0.3; out["notes"].append(f"PKR up {-u:.1f}%/20d — import-cost relief")
            elif sector in _PKR_WINNERS:
                delta -= 0.3; out["notes"].append(f"PKR up {-u:.1f}%/20d — exporter headwind")

    k = context.get("kse100")
    k = k.dropna() if k is not None else None
    if k is not None and len(k) >= 50:
        ema50 = float(k.ewm(span=50, adjust=False).mean().iloc[-1])
        last = float(k.iloc[-1])
        out["kse100_vs_ema50_pct"] = round((last / ema50 - 1) * 100, 2)
        if last < ema50:
            delta -= 0.75
            out["regime"] = "Risk-Off"
            out["notes"].append("KSE-100 below its 50-EMA — index regime filter reduces all signals")
        else:
            out["regime"] = "Risk-On"
    else:
        out["notes"].append("KSE-100 history < 50 sessions — regime filter inactive")

    out["adjustment"] = round(float(np.clip(delta, -1.5, 1.5)), 2)
    return out


def normalize_weights(w: dict | None) -> dict:
    """Merge with defaults, clamp each weight to [FLOOR, CAP], renormalize to 1."""
    merged = {**WEIGHTS, **{k: v for k, v in (w or {}).items() if k in WEIGHTS}}
    clamped = {k: float(np.clip(v, WEIGHT_FLOOR, WEIGHT_CAP)) for k, v in merged.items()}
    total = sum(clamped.values())
    return {k: round(v / total, 4) for k, v in clamped.items()}


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


def analyze(symbol: str, df: pd.DataFrame, context: dict | None = None,
            weights: dict | None = None) -> dict:
    """df: daily OHLCV, DatetimeIndex ascending, >= 260 rows recommended.
    context: optional dict of date-indexed Series: 'brent','usdpkr','kse100'.
    weights: optional calibrated component weights (see normalize_weights)."""
    W = normalize_weights(weights)
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

    # ---------- time-to-target (14-day ATR velocity) ----------
    # Assumes ~0.5 ATR of net favorable progress per session — a stock rarely
    # converts its full daily range into directional movement.
    atr14 = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None

    def _est_days(target, entry):
        if not atr14 or atr14 <= 0 or target <= entry:
            return None
        return int(np.ceil((target - entry) / (0.5 * atr14)))

    time_to_target = {"tp1_days": _est_days(tp1, primary_buy),
                      "tp2_days": _est_days(tp2, primary_buy),
                      "basis": "distance / (0.5 x ATR14) — estimate, not a guarantee"}

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

    composite_technical = round(sum(W[k] * s for k, s in
                                    [("trend", s_trend), ("momentum", s_mom), ("volume", s_vol),
                                     ("structure", s_struct), ("volatility", s_vola), ("mtf", s_mtf)]), 2)

    # ---------- macro overlay (Brent / USD-PKR / KSE-100 regime) ----------
    macro = macro_adjust(symbol, context)
    composite = round(float(np.clip(composite_technical + macro["adjustment"], 0, 10)), 2)

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

    macro_txt = ""
    if macro["adjustment"]:
        macro_txt = (f"Macro overlay {'+' if macro['adjustment'] > 0 else ''}{macro['adjustment']} "
                     f"({'; '.join(macro['notes'])}). ")
    elif macro["regime"] == "Risk-Off":
        macro_txt = "Market regime: Risk-Off (KSE-100 below 50-EMA). "

    summary = (f"{symbol}: {structure_trend} structure on daily; composite score {composite}/10 → {verdict} "
               f"({confidence}% confidence). Bullish: {'; '.join(bull_pts) or 'none'}. "
               f"Bearish: {'; '.join(bear_pts) or 'none'}. " + macro_txt
               + ("Setup meets the 1:2 R:R rule. " if setup_valid else
                  f"Setup REJECTED under the 1:2 R:R rule (R:R = 1:{rr}). " if rr else "")
               + "Place limit orders manually in your broker terminal; these are probabilistic levels, not guarantees.")

    return {
        "symbol": symbol, "as_of": str(df.index[-1].date()),
        "close": close, "volume": int(last["Volume"]),
        "market_structure": structure_trend,
        "verdict": {"call": verdict, "confidence_pct": confidence, "composite_score": composite,
                    "composite_technical": composite_technical,
                    "horizon": HORIZON_SWING,
                    "scores": {"trend": round(s_trend, 1), "momentum": round(s_mom, 1),
                               "volume": round(s_vol, 1), "structure": round(s_struct, 1),
                               "volatility": round(s_vola, 1), "mtf_alignment": round(s_mtf, 1)},
                    "weights": W},
        "macro_context": macro,
        "order_table": {"primary_buy_limit": primary_buy, "secondary_buy_dip": secondary_buy,
                        "stop_loss": stop_loss, "take_profit_1": tp1, "take_profit_2": tp2,
                        "risk_reward": f"1 : {rr}" if rr else "n/a", "setup_valid_rr_rule": setup_valid,
                        "horizon": HORIZON_SHORT, "atr14": round(atr14, 2) if atr14 else None,
                        "time_to_target": time_to_target},
        "horizons": {"order_table": HORIZON_SHORT, "verdict": HORIZON_SWING,
                     "price_targets": {"short_term": HORIZON_SHORT, "swing": HORIZON_SWING,
                                       "3m": HORIZON_LONG, "6m": HORIZON_LONG,
                                       "1y": HORIZON_LONG, "long_term_3_5y": "Beyond 12 months"}},
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