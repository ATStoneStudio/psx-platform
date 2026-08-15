import numpy as np, pandas as pd, json, sys
sys.path.insert(0, "/home/claude/psx-platform")
from engine.analysis import analyze

rng = np.random.default_rng(42)
n = 1250
dates = pd.bdate_range("2021-08-16", periods=n)
drift = np.concatenate([np.full(400, .0006), np.full(300, -.0008), np.full(n-700, .0012)])
ret = drift + rng.normal(0, .015, n)
close = 180*np.exp(np.cumsum(ret))
o = close*(1+rng.normal(0,.004,n)); h = np.maximum(o,close)*(1+abs(rng.normal(0,.006,n)))
l = np.minimum(o,close)*(1-abs(rng.normal(0,.006,n)))
vol = (rng.lognormal(13.5,.5,n)*(1+np.clip(np.abs(ret)*30,0,2))).astype(int)
df = pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":close,"Volume":vol}, index=dates)

res = analyze("TEST", df)
print(json.dumps(res, indent=1, default=str))
