"""SHAP explanations, explanation stability, sparse-model test, multi-seed RUSBoost check."""
import sys, json, warnings
import numpy as np, pandas as pd
import shap, xgboost as xgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier
warnings.filterwarnings("ignore")
OUT = sys.argv[1]
d = pd.read_pickle(f"{OUT}/panel.pkl")

RAW = ["act","ap","at","ceq","che","cogs","csho","dlc","dltis","dltt","dp","ib","invt","ivao",
       "ivst","lct","lt","ni","ppegt","pstk","re","rect","sale","sstk","txp","txt","xint","prcc_f"]
TAX = ["etr","txt_at","txp_at","btd_at","profit_low_tax","d_txp","d_etr","d_btd","tax_earn_gap"]
TAX_ALL = TAX + [c + "_miss" for c in ["etr","d_txp","d_etr","d_btd","tax_earn_gap"]]
FEATS = RAW + TAX_ALL
TAXSET = set(TAX_ALL) | {"txt", "txp"}

def ndcg_at_k(y, s, top=0.01):
    k = int(round(len(y) * top)); idx = np.argsort(-s, kind="stable")[:k]
    dcg = sum(1/np.log2(i+2) for i, j in enumerate(idx) if y[j] == 1)
    z = sum(1/np.log2(i+2) for i in range(min(k, int(y.sum()))))
    return dcg / z if z else 0.0

def split(year):
    tr = d[(d.fyear >= 1991) & (d.fyear <= year - 2)]; te = d[d.fyear == year]
    tp = set(te.loc[te.misstate == 1, "p_aaer"].dropna())
    y = tr.misstate.values.copy(); y[tr.p_aaer.isin(tp).values] = 0
    return tr, y, te, te.misstate.values

def xgbm(pw, seed=0):
    return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=pw,
                             random_state=seed, n_jobs=2, tree_method="hist")

# ---- 1. SHAP per test year; stability of global importance ----
imp = {}
for year in range(2003, 2009):
    tr, y, te, yt = split(year)
    m = xgbm((y == 0).sum() / y.sum()).fit(tr[FEATS].values, y)
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(te[FEATS].values)
    imp[year] = pd.Series(np.abs(sv).mean(0), index=FEATS)
    if year == 2008:
        sv08, te08, m08, ex08 = sv, te, m, ex
imp = pd.DataFrame(imp)
imp.to_csv(f"{OUT}/shap_importance_by_year.csv")
years = list(imp.columns)
rho = [spearmanr(imp[a], imp[b]).correlation for i, a in enumerate(years) for b in years[i+1:]]
top10 = [set(imp[y].nlargest(10).index) for y in years]
jacc = [len(a & b) / len(a | b) for i, a in enumerate(top10) for b in top10[i+1:]]
mean_imp = imp.mean(1).sort_values(ascending=False)
share_tax = float(mean_imp[[c for c in mean_imp.index if c in TAXSET]].sum() / mean_imp.sum())
stab = dict(spearman_mean=float(np.mean(rho)), spearman_min=float(np.min(rho)),
            top10_jaccard_mean=float(np.mean(jacc)), tax_share_of_total_shap=share_tax,
            top15=mean_imp.head(15).round(5).to_dict(),
            tax_ranks={c: int(list(mean_imp.index).index(c)) + 1 for c in TAX + ["txt", "txp"]})
json.dump(stab, open(f"{OUT}/shap_stability.json", "w"), indent=2)
print(json.dumps(stab, indent=2))

# Figure 1: global SHAP importance (mean over 2003-2008 test years)
LAB = {"at":"Total assets","lt":"Total liabilities","csho":"Shares outstanding","sstk":"Stock issued",
       "prcc_f":"Share price","rect":"Receivables","re":"Retained earnings","ceq":"Common equity",
       "dltis":"LT debt issued","ivao":"Other investments","act":"Current assets","ppegt":"Gross PP&E",
       "che":"Cash","sale":"Sales","cogs":"COGS","invt":"Inventory","ap":"Accounts payable","dp":"Depreciation",
       "dlc":"Debt in current liab.","dltt":"Long-term debt","ib":"Income before EI","lct":"Current liabilities",
       "ni":"Net income","pstk":"Preferred stock","ivst":"ST investments","xint":"Interest expense",
       "txt":"Total income tax (TAX)","txp":"Income tax payable (TAX)","etr":"Effective tax rate (TAX)",
       "txt_at":"Tax expense / assets (TAX)","txp_at":"Tax payable / assets (TAX)","btd_at":"Implied book-tax gap (TAX)",
       "profit_low_tax":"Profitable, no tax (TAX)","d_txp":"Change in tax payable (TAX)","d_etr":"Change in ETR (TAX)",
       "d_btd":"Change in book-tax gap (TAX)","tax_earn_gap":"Earnings vs tax growth gap (TAX)"}
top = mean_imp.head(20)[::-1]
fig, ax = plt.subplots(figsize=(7.5, 6.2))
colors = ["#b5523b" if c in TAXSET else "#3d5a80" for c in top.index]
ax.barh([LAB.get(c, c) for c in top.index], top.values, color=colors)
ax.set_xlabel("Mean |SHAP value| (log-odds), averaged over 2003–2008 test years")
ax.set_title("Figure 1. Global feature importance, gradient-boosted model\n(tax features in red)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_shap_global.png", dpi=200); plt.close()

# Figure 2: local explanation for highest-scored true fraud firm-year in 2008
s08 = m08.predict_proba(te08[FEATS].values)[:, 1]
yt08 = te08.misstate.values
order = np.argsort(-s08)
rank_fraud = [r for r, i in enumerate(order) if yt08[i] == 1]
i_star = order[rank_fraud[0]]
contrib = pd.Series(sv08[i_star], index=FEATS)
topc = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)[::-1]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.barh([LAB.get(c, c) for c in topc.index], topc.values,
        color=["#b5523b" if v > 0 else "#6c8ebf" for v in topc.values])
ax.axvline(0, color="#333", lw=0.8)
ax.set_xlabel("SHAP contribution to fraud score (log-odds)")
ax.set_title(f"Figure 2. Why one 2008 firm-year was flagged\n(true AAER case, ranked {rank_fraud[0]+1} of {len(s08):,})", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_shap_local.png", dpi=200); plt.close()
local = dict(rank=int(rank_fraud[0] + 1), n=int(len(s08)), base=float(np.ravel(ex08.expected_value)[0]),
             top_contrib={k: float(v) for k, v in topc[::-1].items()},
             fraud_ranks_2008=[int(r + 1) for r in rank_fraud])
json.dump(local, open(f"{OUT}/shap_local.json", "w"), indent=2); print(local)

# ---- 2. sparse model: top-k SHAP features chosen on training window only (no look-ahead) ----
sparse = []
for k in [5, 10, 15]:
    for year in range(2003, 2009):
        tr, y, te, yt = split(year)
        pw = (y == 0).sum() / y.sum()
        # select features with SHAP on the training data of the full model
        mfull = xgbm(pw).fit(tr[FEATS].values, y)
        sv_tr = shap.TreeExplainer(mfull).shap_values(tr[FEATS].values[::5])
        sel = list(pd.Series(np.abs(sv_tr).mean(0), index=FEATS).nlargest(k).index)
        ms = xgbm(pw).fit(tr[sel].values, y)
        s = ms.predict_proba(te[sel].values)[:, 1]
        sparse.append(dict(k=k, year=year, auc=roc_auc_score(yt, s), ndcg=ndcg_at_k(yt, s)))
sp = pd.DataFrame(sparse); sp.to_csv(f"{OUT}/sparse_by_year.csv", index=False)
print(sp.groupby("k")[["auc", "ndcg"]].mean())

# ---- 3. multi-seed RUSBoost: raw28 vs raw28_tax (2003-2008) ----
ms_rows = []
for seed in range(5):
    for year in range(2003, 2009):
        tr, y, te, yt = split(year)
        for name, cols in [("raw28", RAW), ("raw28_tax", FEATS)]:
            m = RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5), n_estimators=300,
                                   learning_rate=0.1, sampling_strategy=1.0, random_state=seed)
            m.fit(tr[cols].values, y); s = m.predict_proba(te[cols].values)[:, 1]
            ms_rows.append(dict(seed=seed, year=year, features=name, auc=roc_auc_score(yt, s), ndcg=ndcg_at_k(yt, s)))
    print("seed done", seed, flush=True)
msr = pd.DataFrame(ms_rows); msr.to_csv(f"{OUT}/multiseed.csv", index=False)
agg = msr.groupby(["seed", "features"])[["auc", "ndcg"]].mean().unstack()
print(agg); agg.to_csv(f"{OUT}/multiseed_summary.csv")
