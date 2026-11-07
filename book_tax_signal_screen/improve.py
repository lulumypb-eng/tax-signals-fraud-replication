exec(open("variants.py").read().split("for name, M, mk")[0])
from scipy.stats import rankdata
# Beneish-style components
lagg = lambda c: g[c].shift(1)
B = pd.DataFrame(index=d.index)
B["dsri"] = np.where(cons, sd(sd(d.rect, d.sale), sd(lagg("rect"), lagg("sale"))), np.nan)
gm = sd(d.sale - d.cogs, d.sale); B["gmi"] = np.where(cons, sd(gm.groupby(d.cik).shift(1), gm), np.nan)
aq = 1 - sd(d.act + d.ppegt.fillna(0), d["at"]); B["aqi"] = np.where(cons, sd(aq, aq.groupby(d.cik).shift(1)), np.nan)
B["sgi"] = np.where(cons, sd(d.sale, lagg("sale")), np.nan)
dep = sd(d.dp, d.dp + d.ppegt); B["depi"] = np.where(cons, sd(dep.groupby(d.cik).shift(1), dep), np.nan)
lv = sd(d["lt"], d["at"]); B["lvgi"] = np.where(cons, sd(lv, lv.groupby(d.cik).shift(1)), np.nan)
B["tata"] = sd(d.ib - (d.ib + d.dp.fillna(0)), d["at"])  # placeholder replaced below
B["tata"] = X["wc_accruals"]
for c in B:
    lo = B.groupby(d.fyear)[c].transform(lambda s: s.quantile(.01)); hi = B.groupby(d.fyear)[c].transform(lambda s: s.quantile(.99)); B[c] = B[c].clip(lo, hi)
sic = pd.to_numeric(d.sic, errors="coerce").fillna(0)
B["sic2"] = (sic // 100); B["fin"] = sic.between(6000, 6999).astype(float)
XN = pd.concat([XB, B], axis=1)
def rus(s): return RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5), n_estimators=300, learning_rate=.1, sampling_strategy=1., random_state=s)
def xg(pw): return xgb.XGBClassifier(n_estimators=400,max_depth=3,learning_rate=.03,subsample=.8,colsample_bytree=.6,min_child_weight=10,scale_pos_weight=pw,random_state=0,n_jobs=4,tree_method="hist")
out = {}
for t in range(2015, 2021):
    tr = d.fyear.between(2009, t-2); te = d.fyear == t
    cases = set(d.loc[te&(d.y==1),"case_id"]); y = d.y.where(~d.case_id.isin(cases), 0)[tr].values; yt = d.y[te].values
    res = {}
    for nm, M in [("base", XB), ("plusB", XN)]:
        M0 = M.fillna(0)
        res[nm+"_rus"] = sum(rus(s).fit(M0[tr].values, y).predict_proba(M0[te].values)[:,1] for s in (0,1,2))/3
        res[nm+"_xgb"] = xg((y==0).sum()/y.sum()).fit(M[tr].values, y).predict_proba(M[te].values)[:,1]
        res[nm+"_ens"] = rankdata(res[nm+"_rus"]) + rankdata(res[nm+"_xgb"])
    for k, s in res.items():
        k10 = int(round(len(yt)*.1)); top = np.argsort(-s)[:k10]
        out.setdefault(k, []).append((roc_auc_score(yt, s), yt[top].sum(), yt.sum()))
    print(t, {k: round(v[-1][0],3) for k,v in out.items()}, flush=True)
for k, v in out.items():
    v = np.array(v); print(k, "AUC", v[:,0].mean().round(3), "top10 capture", (v[:,1].sum()/v[:,2].sum()).round(3))
