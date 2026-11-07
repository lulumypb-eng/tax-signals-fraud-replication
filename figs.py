import sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = sys.argv[1]
plt.rcParams.update({"font.size": 10})
LAB = {"ppegt":"Gross PP&E","act":"Current assets","csho":"Shares outstanding","lct":"Current liabilities",
       "ap":"Accounts payable","xint":"Interest expense","rect":"Receivables","dp":"Depreciation",
       "etr":"Effective tax rate (tax)","sstk":"Stock issued"}
TAXSET = {"etr","txt_at","txp_at","btd_at","profit_low_tax","d_txp","d_etr","d_btd","tax_earn_gap","txt","txp"}

# Fig 1 again without embedded title
imp = pd.read_csv(f"{OUT}/shap_importance_by_year.csv", index_col=0).mean(1).sort_values(ascending=False)
L1 = {"at":"Total assets","lt":"Total liabilities","csho":"Shares outstanding","sstk":"Stock issued",
      "prcc_f":"Share price","rect":"Receivables","re":"Retained earnings","ceq":"Common equity",
      "ivst":"Short-term investments","act":"Current assets","ppegt":"Gross PP&E","che":"Cash","sale":"Sales",
      "invt":"Inventory","ap":"Accounts payable","dp":"Depreciation","dlc":"Debt in current liabilities",
      "lct":"Current liabilities","btd_at":"Implied book-tax gap (tax)","txt_at":"Tax expense / assets (tax)"}
top = imp.head(20)[::-1]
fig, ax = plt.subplots(figsize=(7, 5.6))
ax.barh([L1.get(c, c) for c in top.index], top.values,
        color=["#b5523b" if c in TAXSET else "#3d5a80" for c in top.index])
ax.set_xlabel("Mean |SHAP value| (log-odds), averaged over 2003–2008 test years")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1.png", dpi=200); plt.close()

# Fig 2 local explanation; colours = direction
loc = json.load(open(f"{OUT}/shap_local.json"))
c = pd.Series(loc["top_contrib"])[::-1]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh([LAB.get(k, k) for k in c.index], c.values, color=["#c9822b" if v > 0 else "#8a8f98" for v in c.values])
ax.axvline(0, color="#333", lw=0.8)
ax.set_xlabel("SHAP contribution (log-odds): right = raises fraud score, left = lowers it")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2.png", dpi=200); plt.close()

# Fig 3 AUC by year
r = pd.read_csv(f"{OUT}/results_by_year.csv")
fig, ax = plt.subplots(figsize=(7, 3.8))
for (m, f), lab, st, col in [(("rusboost", "raw28"), "RUSBoost, 28 raw items", "-", "#3d5a80"),
                             (("rusboost", "raw28_tax"), "RUSBoost, + tax signals", "--", "#b5523b"),
                             (("logit", "ratios14"), "Logit, 14 ratios", ":", "#555555")]:
    s = r[(r.model == m) & (r.features == f)].sort_values("year")
    ax.plot(s.year, s.auc, st, marker="o", ms=4, color=col, label=lab)
ax.axvspan(2008.5, 2014.5, color="#eeeeee", zorder=0)
ax.text(2011.5, 0.62, "extended period\n(fewer detected frauds)", ha="center", fontsize=8, color="#555")
ax.set_ylabel("Out-of-sample AUC"); ax.set_xticks(range(2003, 2015)); ax.set_ylim(0.6, 0.9)
ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3.png", dpi=200); plt.close()
print("ok")
