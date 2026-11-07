"""Prototype screening tool: interpretable ratio + tax-signal XGBoost, SHAP reasons, scores latest filings."""
import json, warnings, numpy as np, pandas as pd, xgboost as xgb, shap
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
D = "../extension_2009_2022/data/"
d = pd.read_csv(D + "sec_panel.csv.gz", dtype={"cik": int}); d = d[d["at"] > 0].copy()
lab = pd.read_csv(D + "labels.csv").drop_duplicates(["cik", "fyear"])
d = d.merge(lab[["cik", "fyear", "case_id"]], on=["cik", "fyear"], how="left")
d["y"] = d.case_id.notna().astype(int)
d = d.sort_values(["cik", "fyear"]).reset_index(drop=True)
for c in ["act","ap","at","ceq","che","cogs","csho","dlc","dltis","dltt","dp","ib","invt","ivao","ivst","lct","lt","ni",
          "ppegt","pstk","re","rect","sale","sstk","txp","txt","xint","pi","txdi","txpd"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
def sd(a, b): return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
g = d.groupby("cik"); L = lambda c: g[c].shift(1)
cons = (d.fyear - L("fyear")) == 1
avg = (d["at"] + L("at")) / 2
d["str"] = np.where(d.fyear >= 2018, .21, .35)
d["pti"] = d["pi"].where(d["pi"].notna(), d["ib"] + d["txt"])
F = {}
F["roa"] = sd(d.ib, d["at"])
F["d_roa"] = np.where(cons, F["roa"] - sd(L("ib"), L("at")), np.nan)
F["leverage"] = sd(d["lt"], d["at"])
F["size"] = np.log(d["at"])
F["d_rec"] = np.where(cons, sd(d.rect - L("rect"), avg), np.nan)
F["d_inv"] = np.where(cons, sd(d.invt - L("invt"), avg), np.nan)
F["soft_assets"] = sd(d["at"] - d.ppegt.fillna(0) - d.che.fillna(0), d["at"])
cs = d.sale - (d.rect - L("rect"))
F["d_cash_sales"] = np.where(cons, sd(cs, L("sale")) - 1, np.nan)
F["sales_growth"] = np.where(cons, sd(d.sale, L("sale")) - 1, np.nan)
F["wc_accruals"] = np.where(cons, sd((d.act - L("act")) - (d.che.fillna(0) - L("che").fillna(0))
                                     - (d.lct - L("lct")) + (d.dlc.fillna(0) - L("dlc").fillna(0)), avg), np.nan)
F["issued_equity"] = (d.sstk.fillna(0) > 0).astype(float)
F["issued_debt"] = (d.dltis.fillna(0) > 0).astype(float)
F["gross_margin"] = sd(d.sale - d.cogs, d.sale)
F["etr"] = np.where(d.pti > 0, sd(d.txt, d.pti), np.nan)
F["btd"] = sd(d.pti - d.txt / d["str"], d["at"])
F["profit_low_tax"] = ((d.pti > 0) & (d.txt <= 0)).astype(float)
F["tax_earn_gap"] = np.where(cons, sd(d.pti - L("pti"), avg) - sd(d.txt / d["str"] - L("txt") / g["str"].shift(1), avg), np.nan)
F["cash_etr"] = np.where(d.pti > 0, sd(d.txpd, d.pti), np.nan)
F["d_tax_payable"] = np.where(cons, sd(d.txp - L("txp"), avg), np.nan)
X = pd.DataFrame(F, index=d.index)
for c in X:
    if X[c].nunique() > 2:
        lo = X.groupby(d.fyear)[c].transform(lambda s: s.quantile(.01)); hi = X.groupby(d.fyear)[c].transform(lambda s: s.quantile(.99))
        X[c] = X[c].clip(lo, hi)
X["etr"] = X["etr"].clip(0, 1); X["cash_etr"] = X["cash_etr"].clip(0, 1)
COLS = list(X.columns)
def mdl(pw): return xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.05, subsample=.8, colsample_bytree=.8,
                                     min_child_weight=10, scale_pos_weight=pw, random_state=0, n_jobs=4, tree_method="hist")
# honest out-of-time check: train <= t-2, test t (cases in test removed from training), 2015-2022
res = []
for t in range(2015, 2023):
    tr = d.fyear.between(2009, t - 2); te = d.fyear == t
    cases = set(d.loc[te & (d.y == 1), "case_id"])
    y = d.y.where(~d.case_id.isin(cases), 0)[tr].values
    if d.y[te].sum() == 0: continue
    s = mdl((y == 0).sum() / y.sum()).fit(X[tr].values, y).predict_proba(X[te].values)[:, 1]
    yt = d.y[te].values; k = int(round(len(yt) * .05)); top = np.argsort(-s)[:k]
    res.append(dict(year=t, n=len(yt), frauds=int(yt.sum()), auc=roc_auc_score(yt, s), top5_capture=yt[top].sum() / yt.sum()))
res = pd.DataFrame(res); print(res.round(3)); print(res[["auc", "top5_capture"]].mean().round(3))
# final model: all labelled years 2009-2022, score latest 10-K per company with fiscal year 2023-2025
tr = d.fyear.between(2009, 2022); y = d.y[tr].values
m = mdl((y == 0).sum() / y.sum()).fit(X[tr].values, y)
latest = d[d.fyear >= 2023].sort_values("fyear").groupby("cik").tail(1)
latest = latest[(latest["at"] >= 10) & latest.sale.notna() & (latest.sale > 0)]  # >= $10m? units are $; filter tiny shells below
latest = latest[latest["at"] >= 1e7]
Xl = X.loc[latest.index]
p = m.predict_proba(Xl.values)[:, 1]
ref = m.predict_proba(X[d.fyear.between(2019, 2022)].values)[:, 1]
pct = (np.searchsorted(np.sort(ref), p) / len(ref) * 100).round(1)
sv = shap.TreeExplainer(m).shap_values(Xl.values)
imp = pd.Series(np.abs(shap.TreeExplainer(m).shap_values(X[tr].values)).mean(0), index=COLS).sort_values(ascending=False)
print(imp.round(4))
out = []
for i, (idx, r) in enumerate(latest.iterrows()):
    order = np.argsort(-np.abs(sv[i]))[:4]
    out.append(dict(c=int(r.cik), n=str(r["name"]).title(), fy=int(r.fyear), sic=None if pd.isna(r.sic) else int(r.sic),
                    pct=float(pct[i]), at=round(float(r["at"]) / 1e6, 1),
                    r=[[COLS[j], None if pd.isna(Xl.iloc[i, j]) else round(float(Xl.iloc[i, j]), 4), round(float(sv[i, j]), 3)] for j in order]))
# typical (median) values for context
med = {c: (None if X.loc[latest.index, c].dropna().empty else round(float(X.loc[latest.index, c].median()), 4)) for c in COLS}
json.dump(dict(firms=out, med=med, val=res.round(3).to_dict(orient="records"),
               imp=imp.round(4).to_dict(), n_train=int(tr.sum()), frauds_train=int(y.sum())),
          open("scores.json", "w"), separators=(",", ":"))
print(len(out), "firms scored; fy mix", latest.fyear.value_counts().to_dict())
