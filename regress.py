"""Conditional association between tax signals and AAER fraud (logit, year FE, firm-clustered SEs)."""
import sys, json, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
OUT = sys.argv[1]
d = pd.read_pickle(f"{OUT}/panel.pkl")
d = d[d.fyear.between(1991, 2008)].copy()
d["size"] = np.log(d["at"].clip(lower=0.001))
d["roa"] = (d["ib"] / d["at"].replace(0, np.nan))
d["lev"] = (d["lt"] / d["at"].replace(0, np.nan))
g = d.groupby("gvkey")
d["sgrow"] = np.where((d.fyear - g.fyear.shift(1)) == 1, d["sale"] / g["sale"].shift(1).replace(0, np.nan) - 1, np.nan)
d["issue_eq"] = (d["sstk"] > 0).astype(float)
for c in ["roa", "lev", "sgrow"]:
    lo, hi = d[c].quantile([0.01, 0.99]); d[c] = d[c].clip(lo, hi)
d = d.dropna(subset=["roa", "lev", "sgrow"])
# standardize tax vars for comparability
TAXC = ["etr", "txt_at", "txp_at", "tax_earn_gap", "profit_low_tax"]
for c in TAXC[:-1]:
    d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
ctrl = "size + roa + lev + sgrow + issue_eq + C(fyear)"
specs = {
    "controls_only": f"misstate ~ {ctrl}",
    "etr": f"misstate ~ etr_z + {ctrl}",
    "txt_at": f"misstate ~ txt_at_z + {ctrl}",
    "txp_at": f"misstate ~ txp_at_z + {ctrl}",
    "tax_earn_gap": f"misstate ~ tax_earn_gap_z + {ctrl}",
    "profit_low_tax": f"misstate ~ profit_low_tax + {ctrl}",
    "all_tax": f"misstate ~ etr_z + txt_at_z + txp_at_z + tax_earn_gap_z + profit_low_tax + {ctrl}",
}
res = {}
for k, f in specs.items():
    m = smf.logit(f, data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d["gvkey"]}, maxiter=200)
    keep = [p for p in m.params.index if not p.startswith("C(") and p != "Intercept"]
    res[k] = dict(n=int(m.nobs), pseudo_r2=float(m.prsquared),
                  coef={p: [float(m.params[p]), float(m.bse[p]), float(m.pvalues[p])] for p in keep})
    print(k, m.nobs, round(m.prsquared, 4)); print(pd.DataFrame(res[k]["coef"], index=["b", "se", "p"]).T.round(4))
# profitable-firm subsample
dp = d[d.pti > 0]
m = smf.logit(f"misstate ~ etr_z + txt_at_z + txp_at_z + tax_earn_gap_z + {ctrl}", data=dp).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": dp["gvkey"]}, maxiter=200)
keep = [p for p in m.params.index if not p.startswith("C(") and p != "Intercept"]
res["all_tax_profitable"] = dict(n=int(m.nobs), pseudo_r2=float(m.prsquared),
    coef={p: [float(m.params[p]), float(m.bse[p]), float(m.pvalues[p])] for p in keep})
print("profitable", m.nobs); print(pd.DataFrame(res["all_tax_profitable"]["coef"], index=["b", "se", "p"]).T.round(4))
res["n_fraud"] = int(d.misstate.sum()); res["n_fraud_profitable"] = int(dp.misstate.sum())
json.dump(res, open(f"{OUT}/regressions.json", "w"), indent=2)
