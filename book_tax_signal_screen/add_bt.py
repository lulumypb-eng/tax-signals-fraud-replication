exec(open("build_demo.py").read().split("COLS = list(X.columns)")[0])
import json
j = json.load(open("scores_v2_plusB_rus.json"))
key = d[["cik","fyear"]].copy(); key["btd"] = X["btd"]; key["gap"] = X["tax_earn_gap"]
key = key.set_index(["cik","fyear"])
rows = []
for f in j["firms"]:
    r = key.loc[(f["c"], f["fy"])] if (f["c"], f["fy"]) in key.index else None
    if isinstance(r, pd.DataFrame): r = r.iloc[0]
    rows.append((None if r is None or pd.isna(r.btd) else float(r.btd), None if r is None or pd.isna(r.gap) else float(r.gap)))
b = pd.DataFrame(rows, columns=["btd","gap"])
pb = b.btd.rank(pct=True); pg = b.gap.rank(pct=True)
for i, f in enumerate(j["firms"]):
    f["bt"] = [None if pd.isna(b.btd[i]) else round(b.btd[i],4), None if pd.isna(pb[i]) else int(round(pb[i]*100)),
               None if pd.isna(b.gap[i]) else round(b.gap[i],4), None if pd.isna(pg[i]) else int(round(pg[i]*100))]
json.dump(j, open("scores_v3.json","w"), separators=(",",":"))
print(b.describe().round(3)); print(sum(f["bt"][0] is not None for f in j["firms"]), sum(f["bt"][2] is not None for f in j["firms"]), len(j["firms"]))
for f in j["firms"]:
    if f["n"].startswith(("Super Micro","Apple Inc","Kraft Heinz")): print(f["n"], f["bt"])
