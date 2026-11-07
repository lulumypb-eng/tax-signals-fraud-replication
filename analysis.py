"""
Tax signals and explainable ML for accounting fraud detection.
Data: Bao, Ke, Li, Yu & Zhang (2020, JAR) public dataset.
Design follows Bao et al.: train on 1991..(t-2), test on year t, serial-fraud correction.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.ensemble import RUSBoostClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")
DATA = sys.argv[1]
OUT = sys.argv[2]

d = pd.read_csv(DATA).sort_values(["gvkey", "fyear"]).reset_index(drop=True)
RAW = ["act","ap","at","ceq","che","cogs","csho","dlc","dltis","dltt","dp","ib","invt","ivao",
       "ivst","lct","lt","ni","ppegt","pstk","re","rect","sale","sstk","txp","txt","xint","prcc_f"]
RATIOS = ["dch_wc","ch_rsst","dch_rec","dch_inv","soft_assets","ch_cs","ch_cm","ch_roa","issue",
          "bm","dpi","reoa","EBIT","ch_fcf"]

# ---------------- tax features ----------------
def safe_div(a, b):
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)

d["pti"] = d["ib"] + d["txt"]                      # pre-tax income proxy
d["str"] = np.where(d["fyear"] >= 1993, 0.35, 0.34)  # federal statutory rate (sample ends 2014)
d["etr"] = np.where(d["pti"] > 0, safe_div(d["txt"], d["pti"]), np.nan)
d["etr"] = d["etr"].clip(0, 1)
d["txt_at"] = safe_div(d["txt"], d["at"])
d["txp_at"] = safe_div(d["txp"], d["at"])
d["btd_at"] = safe_div(d["pti"] - d["txt"] / d["str"], d["at"])   # implied book-tax gap
d["profit_low_tax"] = ((d["pti"] > 0) & (d["txt"] <= 0)).astype(float)

g = d.groupby("gvkey")
consec = (d["fyear"] - g["fyear"].shift(1)) == 1
avg_at = (d["at"] + g["at"].shift(1)) / 2
d["d_txp"] = np.where(consec, safe_div(d["txp"] - g["txp"].shift(1), avg_at), np.nan)
d["d_etr"] = np.where(consec, d["etr"] - g["etr"].shift(1), np.nan)
d["d_btd"] = np.where(consec, d["btd_at"] - g["btd_at"].shift(1), np.nan)
# tax expense growth vs. earnings growth gap (book earnings rising faster than tax)
d["tax_earn_gap"] = np.where(consec,
    safe_div(d["pti"] - g["pti"].shift(1), avg_at) - safe_div((d["txt"] - g["txt"].shift(1)) / d["str"], avg_at),
    np.nan)

TAX = ["etr","txt_at","txp_at","btd_at","profit_low_tax","d_txp","d_etr","d_btd","tax_earn_gap"]

# winsorize continuous tax features at 1/99 by year (within-year, no look-ahead)
for c in ["txt_at","txp_at","btd_at","d_txp","d_etr","d_btd","tax_earn_gap"]:
    lo = d.groupby("fyear")[c].transform(lambda s: s.quantile(0.01))
    hi = d.groupby("fyear")[c].transform(lambda s: s.quantile(0.99))
    d[c] = d[c].clip(lo, hi)
for c in TAX:
    d[c + "_miss"] = d[c].isna().astype(float) if c != "profit_low_tax" else 0.0
    d[c] = d[c].fillna(0.0)
TAX_ALL = TAX + [c + "_miss" for c in ["etr","d_txp","d_etr","d_btd","tax_earn_gap"]]

# Non-tax raw vars (to test tax block separately)
RAW_NOTAX = [c for c in RAW if c not in ("txt","txp")]

# ratios: fill missing with 0 + winsor per year
for c in RATIOS:
    lo = d.groupby("fyear")[c].transform(lambda s: s.quantile(0.01))
    hi = d.groupby("fyear")[c].transform(lambda s: s.quantile(0.99))
    d[c] = d[c].clip(lo, hi).fillna(0.0)

FEATSETS = {
    "raw28": RAW,
    "raw26_notax": RAW_NOTAX,
    "raw28_tax": RAW + TAX_ALL,
    "ratios14": RATIOS,
    "ratios14_tax": RATIOS + TAX_ALL,
}

# ---------------- metrics ----------------
def ndcg_at_k(y, s, top=0.01):
    k = int(round(len(y) * top))
    idx = np.argsort(-s, kind="stable")[:k]
    dcg = sum(1 / np.log2(i + 2) for i, j in enumerate(idx) if y[j] == 1)
    kz = min(k, int(y.sum()))
    z = sum(1 / np.log2(i + 2) for i in range(kz))
    return dcg / z if z > 0 else 0.0

def topk(y, s, top=0.01):
    k = int(round(len(y) * top))
    idx = np.argsort(-s, kind="stable")[:k]
    tp = y[idx].sum()
    return tp / y.sum(), tp / k

# ---------------- models ----------------
def make_model(name, seed=0, pos_w=1.0):
    if name == "rusboost":
        return RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5),
                                  n_estimators=300, learning_rate=0.1, sampling_strategy=1.0,
                                  random_state=seed)
    if name == "xgb":
        return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                 scale_pos_weight=pos_w, eval_metric="auc",
                                 random_state=seed, n_jobs=8, tree_method="hist")
    if name == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))

RUNS = [("logit", "ratios14"), ("logit", "ratios14_tax"),
        ("rusboost", "raw28"), ("rusboost", "raw26_notax"), ("rusboost", "raw28_tax"),
        ("xgb", "raw28"), ("xgb", "raw28_tax")]

rows, preds = [], {}
for year in range(2003, 2015):
    tr = d[(d.fyear >= 1991) & (d.fyear <= year - 2)].copy()
    te = d[d.fyear == year].copy()
    test_paaer = set(te.loc[te.misstate == 1, "p_aaer"].dropna())
    y_tr = tr["misstate"].values.copy()
    y_tr[tr["p_aaer"].isin(test_paaer).values] = 0   # serial fraud correction
    y_te = te["misstate"].values
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    for model, fs in RUNS:
        cols = FEATSETS[fs]
        m = make_model(model, seed=0, pos_w=pos_w)
        m.fit(tr[cols].values, y_tr)
        s = m.predict_proba(te[cols].values)[:, 1]
        sens, prec = topk(y_te, s)
        rows.append(dict(year=year, model=model, features=fs, n=len(y_te), frauds=int(y_te.sum()),
                         auc=roc_auc_score(y_te, s), ndcg=ndcg_at_k(y_te, s),
                         sens_top1=sens, prec_top1=prec))
        preds[(model, fs, year)] = (y_te, s)
        print(year, model, fs, round(rows[-1]["auc"], 4), round(rows[-1]["ndcg"], 4), flush=True)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/results_by_year.csv", index=False)

def summarize(r):
    return r.groupby(["model", "features"])[["auc", "ndcg", "sens_top1", "prec_top1"]].mean()

main = summarize(res[res.year.between(2003, 2008)])
ext = summarize(res)
main.to_csv(f"{OUT}/summary_2003_2008.csv"); ext.to_csv(f"{OUT}/summary_2003_2014.csv")
print(main.round(4).to_string()); print(ext.round(4).to_string())

# ---------- bootstrap of differences (pooled 2003-2008, resampling within year) ----------
rng = np.random.default_rng(42)
def boot_diff(a, b, B=1000):
    years = range(2003, 2009)
    out_auc, out_ndcg = [], []
    for _ in range(B):
        da, dn = [], []
        for yr in years:
            y, sa = preds[(*a, yr)]; _, sb = preds[(*b, yr)]
            idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            if yy.sum() == 0: continue
            da.append(roc_auc_score(yy, sb[idx]) - roc_auc_score(yy, sa[idx]))
            dn.append(ndcg_at_k(yy, sb[idx]) - ndcg_at_k(yy, sa[idx]))
        out_auc.append(np.mean(da)); out_ndcg.append(np.mean(dn))
    out_auc, out_ndcg = np.array(out_auc), np.array(out_ndcg)
    return dict(auc_diff=float(out_auc.mean()), auc_ci=[float(np.percentile(out_auc, 2.5)), float(np.percentile(out_auc, 97.5))],
                auc_p=float(2 * min((out_auc <= 0).mean(), (out_auc >= 0).mean())),
                ndcg_diff=float(out_ndcg.mean()), ndcg_ci=[float(np.percentile(out_ndcg, 2.5)), float(np.percentile(out_ndcg, 97.5))],
                ndcg_p=float(2 * min((out_ndcg <= 0).mean(), (out_ndcg >= 0).mean())))

comps = {
    "rusboost raw28 -> raw28_tax": (("rusboost", "raw28"), ("rusboost", "raw28_tax")),
    "rusboost raw26_notax -> raw28": (("rusboost", "raw26_notax"), ("rusboost", "raw28")),
    "xgb raw28 -> raw28_tax": (("xgb", "raw28"), ("xgb", "raw28_tax")),
    "logit ratios14 -> ratios14_tax": (("logit", "ratios14"), ("logit", "ratios14_tax")),
    "rusboost raw28 -> xgb raw28_tax": (("rusboost", "raw28"), ("xgb", "raw28_tax")),
}
boot = {k: boot_diff(*v) for k, v in comps.items()}
json.dump(boot, open(f"{OUT}/bootstrap.json", "w"), indent=2)
print(json.dumps(boot, indent=2))

# ---------- descriptive stats: fraud vs non-fraud tax features (train-era sample 1991-2008) ----------
desc = d[d.fyear.between(1991, 2008)]
tab = desc.groupby("misstate")[TAX].mean().T
tab.columns = ["non_fraud", "fraud"]
from scipy import stats
tab["t_p"] = [stats.ttest_ind(desc.loc[desc.misstate == 1, c], desc.loc[desc.misstate == 0, c], equal_var=False).pvalue for c in TAX]
tab.to_csv(f"{OUT}/tax_descriptives.csv"); print(tab.round(4).to_string())
counts = dict(n_obs=int(len(desc)), n_fraud=int(desc.misstate.sum()), n_firms=int(desc.gvkey.nunique()))
json.dump(counts, open(f"{OUT}/sample_counts.json", "w"))
d.to_pickle(f"{OUT}/panel.pkl")
