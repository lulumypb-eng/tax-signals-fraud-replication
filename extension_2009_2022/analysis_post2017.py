"""
Extension: tax signals and fraud detection on SEC XBRL data, fiscal 2009-2022.
Inputs: sec_panel.csv.gz (from build_sec_panel.py), labels.csv (cik, fyear, case_id, borderline)
"""
import sys, json, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier
import xgboost as xgb, shap
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
PANEL, LABELS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EXCL_BORDER = len(sys.argv) > 4 and sys.argv[4] == "strict"

d = pd.read_csv(PANEL, dtype={"cik": int})
d = d[d["at"] > 0].copy()
lab = pd.read_csv(LABELS)
if EXCL_BORDER:
    lab = lab[lab.borderline == 0]
lab = lab.drop_duplicates(["cik", "fyear"])
d = d.merge(lab[["cik", "fyear", "case_id"]], on=["cik", "fyear"], how="left")
d["misstate"] = d["case_id"].notna().astype(int)
d = d.sort_values(["cik", "fyear"]).reset_index(drop=True)

RAW = ["act","ap","at","ceq","che","cogs","csho","dlc","dltis","dltt","dp","ib","invt","ivao",
       "ivst","lct","lt","ni","ppegt","pstk","re","rect","sale","sstk","txp","txt","xint"]
# Missing XBRL elements: zero for flows/components that are often simply not reported, keep NaN for core totals
for c in RAW + ["pi", "txdi", "txpd"]:
    if c in d:
        d[c] = pd.to_numeric(d[c], errors="coerce")
for c in ["dltis","ivao","ivst","pstk","sstk","dlc","invt","txp","xint","cogs","dp","rect","ppegt","dltt","ap","che","csho"]:
    d[c] = d[c].fillna(0.0)

def sdiv(a, b):
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

d["str"] = np.where(d["fyear"] >= 2018, 0.21, 0.35)
d["pti"] = d["pi"].where(d["pi"].notna(), d["ib"] + d["txt"])
d["etr"] = np.where(d["pti"] > 0, sdiv(d["txt"], d["pti"]), np.nan); d["etr"] = d["etr"].clip(0, 1)
d["txt_at"] = sdiv(d["txt"], d["at"]); d["txp_at"] = sdiv(d["txp"], d["at"])
d["btd_at"] = sdiv(d["pti"] - d["txt"] / d["str"], d["at"])
d["profit_low_tax"] = ((d["pti"] > 0) & (d["txt"] <= 0)).astype(float)
# New measures from XBRL: cash taxes paid (cash-flow statement) and deferred tax expense
d["cash_etr"] = np.where(d["pti"] > 0, sdiv(d["txpd"], d["pti"]), np.nan); d["cash_etr"] = d["cash_etr"].clip(0, 1)
d["btd_cash_at"] = sdiv(d["pti"] - d["txpd"] / d["str"], d["at"])
d["txpd_at"] = sdiv(d["txpd"], d["at"])
d["txdi_at"] = sdiv(d["txdi"], d["at"])
d["cash_book_tax"] = sdiv(d["txpd"] - d["txt"], d["at"])   # cash taxes paid minus tax expense

g = d.groupby("cik")
consec = (d["fyear"] - g["fyear"].shift(1)) == 1
avg_at = (d["at"] + g["at"].shift(1)) / 2
lag = lambda c: g[c].shift(1)
d["d_txp"] = np.where(consec, sdiv(d["txp"] - lag("txp"), avg_at), np.nan)
d["d_etr"] = np.where(consec, d["etr"] - lag("etr"), np.nan)
d["d_btd"] = np.where(consec, d["btd_at"] - lag("btd_at"), np.nan)
d["tax_earn_gap"] = np.where(consec, sdiv(d["pti"] - lag("pti"), avg_at)
                             - sdiv(d["txt"] / d["str"] - lag("txt") / g["str"].shift(1), avg_at), np.nan)
d["cash_gap"] = np.where(consec, sdiv(d["pti"] - lag("pti"), avg_at)
                         - sdiv(d["txpd"] / d["str"] - lag("txpd") / g["str"].shift(1), avg_at), np.nan)

TAX_OLD = ["etr","txt_at","txp_at","btd_at","profit_low_tax","d_txp","d_etr","d_btd","tax_earn_gap"]
TAX_NEW = ["cash_etr","btd_cash_at","txpd_at","txdi_at","cash_book_tax","cash_gap"]
CONT = [c for c in TAX_OLD + TAX_NEW if c != "profit_low_tax"]
for c in CONT:
    lo = d.groupby("fyear")[c].transform(lambda s: s.quantile(0.01))
    hi = d.groupby("fyear")[c].transform(lambda s: s.quantile(0.99))
    d[c + "_miss"] = d[c].isna().astype(float)
    d[c] = d[c].clip(lo, hi).fillna(0.0)
for c in ["ib", "ni", "txt", "lt", "lct", "act", "ceq", "re", "sale"]:
    d[c] = d[c].fillna(0.0)
MISS_OLD = [c + "_miss" for c in ["etr","d_txp","d_etr","d_btd","tax_earn_gap"]]
MISS_NEW = [c + "_miss" for c in TAX_NEW]
FEATS = {"raw27": RAW, "raw27_tax": RAW + TAX_OLD + MISS_OLD,
         "raw27_tax_full": RAW + TAX_OLD + MISS_OLD + TAX_NEW + MISS_NEW}

def ndcg(y, s, top=0.01):
    k = max(1, int(round(len(y) * top))); idx = np.argsort(-s, kind="stable")[:k]
    dcg = sum(1/np.log2(i+2) for i, j in enumerate(idx) if y[j] == 1)
    z = sum(1/np.log2(i+2) for i in range(min(k, int(y.sum()))))
    return dcg / z if z else 0.0
def topk(y, s, top=0.01):
    k = max(1, int(round(len(y) * top))); idx = np.argsort(-s, kind="stable")[:k]
    return y[idx].sum() / max(y.sum(), 1), y[idx].sum() / k
def model(name, pw, seed=0):
    if name == "rusboost":
        return RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5), n_estimators=300,
                                  learning_rate=0.1, sampling_strategy=1.0, random_state=seed)
    return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=pw,
                             random_state=seed, n_jobs=4, tree_method="hist")
def fitpred(m, pw, Xtr, y, Xte, seed=0):
    for k in range(5):
        try:
            return model(m, pw, seed + 100 * k).fit(Xtr, y).predict_proba(Xte)[:, 1]
        except ValueError:
            continue
    raise RuntimeError("model failed to fit")
def split(t, start=2009):
    tr = d[(d.fyear >= start) & (d.fyear <= t - 2)]; te = d[d.fyear == t]
    cases = set(te.loc[te.misstate == 1, "case_id"].dropna())
    y = tr.misstate.values.copy(); y[tr.case_id.isin(cases).values] = 0
    return tr, y, te, te.misstate.values

summary = {"n_by_year": d.groupby("fyear").agg(n=("misstate", "size"), frauds=("misstate", "sum")).to_dict(orient="index")}
TEST = list(range(2015, 2023))
rows, preds = [], {}
for t in TEST:
    tr, y, te, yt = split(t)
    if yt.sum() == 0 or y.sum() < 5:
        continue
    pw = (y == 0).sum() / y.sum()
    for m in ["rusboost", "xgb"]:
        for fs, cols in FEATS.items():
            s = fitpred(m, pw, tr[cols].values, y, te[cols].values)
            sens, prec = topk(yt, s)
            rows.append(dict(year=t, model=m, features=fs, n=len(yt), frauds=int(yt.sum()),
                             auc=roc_auc_score(yt, s), ndcg=ndcg(yt, s), sens=sens, prec=prec))
            preds[(m, fs, t)] = (yt, s)
            print(t, m, fs, round(rows[-1]["auc"], 3), flush=True)
r = pd.DataFrame(rows); r.to_csv(f"{OUT}/ext_results_by_year.csv", index=False)
def summ(sub): return sub.groupby(["model", "features"])[["auc", "ndcg", "sens", "prec"]].mean()
for nm, yrs in {"main_2015_2020": range(2015, 2021), "pre_tcja_2015_2017": range(2015, 2018),
                "post_tcja_2018_2020": range(2018, 2021), "all_2015_2022": range(2015, 2023)}.items():
    s = summ(r[r.year.isin(yrs)]); s.to_csv(f"{OUT}/ext_summary_{nm}.csv"); print(nm); print(s.round(3))

rng = np.random.default_rng(42)
def boot(a, b, yrs, B=1000):
    da, dn = [], []
    yrs = [t for t in yrs if (*a, t) in preds]
    for _ in range(B):
        xa, xn = [], []
        for t in yrs:
            y, sa = preds[(*a, t)]; _, sb = preds[(*b, t)]
            i = rng.integers(0, len(y), len(y))
            if y[i].sum() == 0: continue
            xa.append(roc_auc_score(y[i], sb[i]) - roc_auc_score(y[i], sa[i])); xn.append(ndcg(y[i], sb[i]) - ndcg(y[i], sa[i]))
        da.append(np.mean(xa)); dn.append(np.mean(xn))
    da, dn = np.array(da), np.array(dn)
    f = lambda v: dict(diff=float(v.mean()), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                       p=float(2 * min((v <= 0).mean(), (v >= 0).mean())))
    return {"auc": f(da), "ndcg": f(dn)}
bt = {}
for m in ["rusboost", "xgb"]:
    for nm, yrs in {"2015_2020": range(2015, 2021), "post_tcja": range(2018, 2021)}.items():
        bt[f"{m} raw->tax {nm}"] = boot((m, "raw27"), (m, "raw27_tax"), yrs)
        bt[f"{m} raw->tax_full {nm}"] = boot((m, "raw27"), (m, "raw27_tax_full"), yrs)
json.dump(bt, open(f"{OUT}/ext_bootstrap.json", "w"), indent=1); print(json.dumps(bt, indent=1))

# multi-seed RUSBoost
ms = []
for seed in range(5):
    for t in range(2015, 2021):
        tr, y, te, yt = split(t)
        if yt.sum() == 0: continue
        for fs in ["raw27", "raw27_tax_full"]:
            s = fitpred("rusboost", 1, tr[FEATS[fs]].values, y, te[FEATS[fs]].values, seed)
            ms.append(dict(seed=seed, year=t, features=fs, auc=roc_auc_score(yt, s), ndcg=ndcg(yt, s)))
ms = pd.DataFrame(ms); ms.to_csv(f"{OUT}/ext_multiseed.csv", index=False)
print(ms.groupby("features")[["auc", "ndcg"]].mean())

# regressions, fiscal 2009-2020
x = d[d.fyear.between(2009, 2020)].copy()
x["size"] = np.log(x["at"]); x["roa"] = sdiv(x["ib"], x["at"]); x["lev"] = sdiv(x["lt"], x["at"])
x["sgrow"] = np.where(consec.loc[x.index], sdiv(x["sale"], lag("sale").loc[x.index]) - 1, np.nan)
x["issue_eq"] = (x["sstk"] > 0).astype(float)
for c in ["roa", "lev", "sgrow"]:
    lo, hi = x[c].quantile([0.01, 0.99]); x[c] = x[c].clip(lo, hi)
x = x.dropna(subset=["roa", "lev", "sgrow"])
REGV = ["etr", "txt_at", "txp_at", "btd_at", "tax_earn_gap", "cash_etr", "btd_cash_at", "cash_book_tax", "cash_gap"]
for c in REGV:
    x[c + "_z"] = (x[c] - x[c].mean()) / x[c].std()
ctrl = "size + roa + lev + sgrow + issue_eq + C(fyear)"
reg = {}
for c in REGV:
    try:
        m = smf.logit(f"misstate ~ {c}_z + {ctrl}", data=x).fit(disp=0, cov_type="cluster", cov_kwds={"groups": x["cik"]})
        reg[c] = [float(m.params[c + "_z"]), float(m.bse[c + "_z"]), float(m.pvalues[c + "_z"])]
    except Exception as e:
        reg[c] = str(e)
reg["n"] = int(len(x)); reg["frauds"] = int(x.misstate.sum())
post = x[x.fyear >= 2018]
for c in ["tax_earn_gap", "cash_gap", "cash_etr", "btd_cash_at"]:
    try:
        m = smf.logit(f"misstate ~ {c}_z + {ctrl}", data=post).fit(disp=0, cov_type="cluster", cov_kwds={"groups": post["cik"]})
        reg[c + "_post2018"] = [float(m.params[c + "_z"]), float(m.bse[c + "_z"]), float(m.pvalues[c + "_z"])]
    except Exception as e:
        reg[c + "_post2018"] = str(e)
reg["n_post2018"] = int(len(post)); reg["frauds_post2018"] = int(post.misstate.sum())
json.dump(reg, open(f"{OUT}/ext_regressions.json", "w"), indent=1); print(json.dumps(reg, indent=1))

# SHAP on XGB full-tax model, test years 2015-2020
imp = {}
for t in range(2015, 2021):
    tr, y, te, yt = split(t)
    if yt.sum() == 0: continue
    cols = FEATS["raw27_tax_full"]
    m = model("xgb", (y == 0).sum() / y.sum()).fit(tr[cols].values, y)
    imp[t] = pd.Series(np.abs(shap.TreeExplainer(m).shap_values(te[cols].values)).mean(0), index=cols)
imp = pd.DataFrame(imp); imp.to_csv(f"{OUT}/ext_shap_importance.csv")
mi = imp.mean(1).sort_values(ascending=False)
taxset = set(TAX_OLD + TAX_NEW + MISS_OLD + MISS_NEW + ["txt", "txp"])
summary["shap_tax_share"] = float(mi[[c for c in mi.index if c in taxset]].sum() / mi.sum())
summary["shap_top15"] = mi.head(15).round(4).to_dict()
summary["shap_rank_new_tax"] = {c: int(list(mi.index).index(c)) + 1 for c in TAX_NEW}
from scipy.stats import spearmanr
yrs = list(imp.columns)
summary["shap_spearman_mean"] = float(np.mean([spearmanr(imp[a], imp[b]).correlation for i, a in enumerate(yrs) for b in yrs[i+1:]]))
json.dump(summary, open(f"{OUT}/ext_summary.json", "w"), indent=1, default=str); print(json.dumps(summary, indent=1, default=str))
