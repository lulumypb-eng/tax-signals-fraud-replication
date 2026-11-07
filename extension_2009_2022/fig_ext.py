"""Figure 2 of the paper: RUSBoost out-of-sample AUC by test year on SEC XBRL data, 2015-2022.
Usage: python fig_ext.py out    (reads out/ext_results_by_year.csv, writes out/fig_ext.png)"""
import sys
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = sys.argv[1]
r = pd.read_csv(f"{OUT}/ext_results_by_year.csv")
r = r[(r.model == "rusboost") & r.year.between(2015, 2022)]
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.axvspan(2017.5, 2022.5, color="#e8e8e8", lw=0)
ax.axvspan(2020.5, 2022.5, color="#f4f4f4", lw=0)
SETS = sorted(r.features.unique(), key=len)
STY = ["-", "--", ":"]; COL = ["#3d5a80", "#b5523b", "#7a7f87"]
NAME = {SETS[0]: "27 raw items"}
if len(SETS) > 1: NAME[SETS[1]] = "+ tax features"
if len(SETS) > 2: NAME[SETS[2]] = "+ tax and cash-tax features"
for i, f in enumerate(SETS):
    s = r[r.features == f].sort_values("year")
    ax.plot(s.year, s.auc, STY[i % 3], color=COL[i % 3], marker="o", ms=4, label=NAME.get(f, f))
ax.axhline(0.5, color="#999", lw=0.8)
ax.set_xlabel("Test year"); ax.set_ylabel("AUC")
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_ext.png", dpi=200); plt.close()
