"""One-off: walk-forward replay over N years to build a defensible sample.
Offline (no network). Parallel across symbols; parent does all DB writes."""
import json, sqlite3, sys, os
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT))
from engine.analysis import analyze
from engine.dataio import load_symbol, ensure_schema, load_macro_context

DB = os.environ.get("REPLAY_DB", str(ROOT / "data" / "psx.sqlite"))
YEARS = float(os.environ.get("REPLAY_YEARS", "3"))

def work(sym):
    """CORRECTNESS: engine levels are computed on split-ADJUSTED prices, but
    outcomes are scored against RAW bars. Those agree only after a symbol's last
    corporate action — before it, adjusted prices are a different scale (MARI's
    2024 split is 8.5x) and the back-adjustment factor itself is derived from a
    FUTURE event, which is lookahead. So for each symbol we replay only the
    window since its last corporate action: real traded prices, no lookahead,
    evaluation in the same space the signal was computed in."""
    con = sqlite3.connect(DB)
    try:
        df, events = load_symbol(con, sym)
    except Exception:
        return []
    macro = load_macro_context(con)
    dates = [str(d.date()) for d in df.index][-int(YEARS * 250):]
    if events:
        cutoff = max(e["date"] for e in events)
        dates = [d for d in dates if d > cutoff]
    out = []
    for d in dates:
        sub = df[df.index <= d]
        if len(sub) < 60:
            continue
        try:
            ctx = {k: (v[v.index <= d] if v is not None and len(v) else v) for k, v in macro.items()}
            r = analyze(sym, sub, context=ctx)          # default weights: no lookahead
            o, v = r["order_table"], r["verdict"]
            out.append((sym, r["as_of"], v["call"], v["composite_score"], v["confidence_pct"],
                        o["horizon"], o["primary_buy_limit"], o["stop_loss"], o["take_profit_1"],
                        o["take_profit_2"], int(o["setup_valid_rr_rule"]), json.dumps(v["scores"]), "backfill"))
        except Exception:
            pass
    return out

if __name__ == "__main__":
    con = sqlite3.connect(DB); ensure_schema(con)
    syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM ohlcv WHERE date LIKE '____-__-__'")]
    print(f"replay {YEARS}y over {len(syms)} symbols", flush=True)
    n = 0
    with Pool(processes=min(os.cpu_count() or 4, 8)) as p:
        for i, rows in enumerate(p.imap_unordered(work, syms), 1):
            if rows:
                con.executemany(
                    "INSERT OR IGNORE INTO predictions_history (symbol,signal_date,verdict,composite,"
                    "confidence,horizon,entry,stop_loss,tp1,tp2,setup_valid,scores_json,source) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                con.commit(); n += len(rows)
            if i % 10 == 0:
                print(f"  {i}/{len(syms)} symbols, {n} signals", flush=True)
    print(f"logged {n} signals", flush=True)