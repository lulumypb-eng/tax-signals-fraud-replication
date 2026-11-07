"""Final build: honest rank history (each year scored by a model trained only on data up to t-2), benchmarks, surrogate reasons."""
import sys
exec(open("improve.py").read().split("def rus(s)")[0])
from scipy.stats import rankdata
from sklearn.metrics import r2_score
USE_B = sys.argv[1] == "plusB"; ENS = sys.argv[2] == "ens"
FEAT = (XN if USE_B else XB)
def rus(s): return RUSBoostClassifier(estimator=DecisionTreeClassifier(min_samples_leaf=5), n_estimators=300, learning_rate=.1, sampling_strategy=1., random_state=s)
def xg(pw): return xgb.XGBClassifier(n_estimators=400,max_depth=3,learning_rate=.03,subsample=.8,colsample_bytree=.6,min_child_weight=10,scale_pos_weight=pw,random_state=0,n_jobs=4,tree_method="hist")
def fit_score(trm, tem):
    y = d.y[trm].values; M0 = FEAT.fillna(0)
    r = sum(rus(s).fit(M0[trm].values, y).predict_proba(M0[tem].values)[:,1] for s in (0,1,2))/3
    if not ENS: return r
    x = xg((y==0).sum()/y.sum()).fit(FEAT[trm].values, y).predict_proba(FEAT[tem].values)[:,1]
    return (rankdata(r) + rankdata(x)) / (2*len(r))
d["at_dec"] = d.groupby("fyear")["at"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
keep = (d["at"] >= 1e7) & (d.sale > 0)
d["score"] = np.nan
for t in range(2019, 2025):
    te = (d.fyear == t) & keep; tr = d.fyear.between(2009, min(t-2, 2022))
    d.loc[te, "score"] = fit_score(tr, te); print("scored", t, flush=True)
sub = d[keep & d.fyear.between(2019, 2024)].copy()
sub["dec"] = sub.groupby("fyear")["at"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
sub["sp"] = (sub.groupby(["fyear","dec"])["score"].rank(pct=True)*100).round().clip(upper=99).astype(int)
# benchmarks on test years 2015-2020: Beneish M (no SGAI) and Dechow F (approx.) vs model
Bm = -4.84 + .92*B.dsri + .528*B.gmi + .404*B.aqi + .892*B.sgi + .115*B.depi - .172*1 + 4.679*B.tata - .327*B.lvgi
Fz = -7.893 + .790*X.wc_accruals + 2.518*X.d_rec + 1.191*X.d_inv + 1.979*X.soft_assets + .171*X.d_cash_sales - .932*X.d_roa + 1.029*X.issued_equity
bench = []
for t in range(2015, 2021):
    te = d.fyear == t; yt = d.y[te].values
    for nm, s in [("Beneish M-score", Bm[te]), ("Dechow F-score", Fz[te])]:
        s = s.fillna(s.median()).values; k = int(round(len(yt)*.1)); top = np.argsort(-s)[:k]
        bench.append(dict(model=nm, year=t, auc=roc_auc_score(yt, s), hit=int(yt[top].sum()), n=int(yt.sum())))
bench = pd.DataFrame(bench); bsum = bench.groupby("model").apply(lambda g: pd.Series(dict(auc=g.auc.mean(), cap=g.hit.sum()/g.n.sum()))).round(3)
print(bsum)
# surrogate reasons for latest filing
latest = sub.sort_values("fyear").groupby("cik").tail(1); latest = latest[latest.fyear >= 2023]
allidx = sub.index; tgt = np.log(np.clip(sub.score, 1e-4, 1-1e-4)/(1-np.clip(sub.score, 1e-4, 1-1e-4)))
XS = X.copy()
if USE_B: XS = pd.concat([X, B[["dsri","gmi","aqi","sgi","depi","lvgi"]]], axis=1)
COLS2 = list(XS.columns)
sur = xgb.XGBRegressor(n_estimators=600, max_depth=5, learning_rate=.05, subsample=.8, colsample_bytree=.8, random_state=0, n_jobs=4)
msk = np.random.default_rng(0).random(len(allidx)) < .2
sur.fit(XS.loc[allidx[~msk]].values, tgt.values[~msk]); r2 = r2_score(tgt.values[msk], sur.predict(XS.loc[allidx[msk]].values)); print("R2", round(r2,3))
sur.fit(XS.loc[allidx].values, tgt.values)
sv = shap.TreeExplainer(sur).shap_values(XS.loc[latest.index].values)
pc = {c: XS.loc[latest.index, c].rank(pct=True) for c in COLS2 if XS[c].nunique() > 2}
hist = sub.groupby("cik").apply(lambda g: {int(f): int(s) for f, s in zip(g.fyear, g.sp)}).to_dict()
SIC = {1:"Mining & construction",2:"Manufacturing",3:"Manufacturing",4:"Transport & utilities",5:"Trade",6:"Finance & real estate",7:"Services",8:"Services"}
firms = []
Xl = XS.loc[latest.index]
for i, (idx, r) in enumerate(latest.iterrows()):
    def item(j):
        v = Xl.iloc[i, j]; c = COLS2[j]
        return [c, None if pd.isna(v) else round(float(v),4), None if c not in pc or pd.isna(pc[c].iloc[i]) else int(round(pc[c].iloc[i]*100)), round(float(sv[i,j]),3)]
    up = [j for j in np.argsort(-sv[i]) if sv[i,j] > 0 and COLS2[j] != "size"][:3]
    dn = [j for j in np.argsort(sv[i]) if sv[i,j] < 0 and COLS2[j] != "size"][:1]
    s = pd.to_numeric(r.sic, errors="coerce")
    firms.append(dict(c=int(r.cik), n=str(r["name"]).title(), fy=int(r.fyear), at=round(float(r["at"])/1e6,1), sp=int(r.sp), dec=int(r.dec)+1,
                      ind=SIC.get(int(s//1000), "Other") if pd.notna(s) else "Other", sic=None if pd.isna(s) else int(s),
                      h=hist.get(r.cik, {}), up=[item(j) for j in up], dn=[item(j) for j in dn]))
edges = latest.assign(dec=[f["dec"]-1 for f in firms]).groupby("dec")["at"].agg(["min","max"]).div(1e6).round(1).values.tolist()
json.dump(dict(firms=firms, dec_edges=edges, r2=round(r2,3), bench=bsum.reset_index().to_dict(orient="records"), n=len(firms)),
          open(f"scores_v2_{sys.argv[1]}_{sys.argv[2]}.json","w"), separators=(",",":"))
print(len(firms), "done")
