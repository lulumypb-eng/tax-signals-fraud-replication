"""
Combine fraud labels for fiscal 2009-2024:
  (a) Bao et al. (2020) AAER firm-years (CIK already given), AAERs through no. 4012
  (b) hand-collected AAERs no. 4013-4605 (new_aaer_labels.csv), matched to CIK by company name
Writes labels.csv (cik, fyear, case_id, borderline) and name_matches.csv for manual review.
"""
import re, sys
import pandas as pd

PANEL, BAO, NEW, OUT = sys.argv[1:5]
OVERRIDES = {}   # aaer -> cik, filled in after manual review of name_matches.csv
if len(sys.argv) > 5:
    OVERRIDES = {int(k): int(v) for k, v in (x.split(":") for x in sys.argv[5].split(","))}

panel = pd.read_csv(PANEL, usecols=["cik", "name", "fyear"])
names = panel.sort_values("fyear").drop_duplicates("cik", keep="last")[["cik", "name"]]

def norm(s):
    s = str(s).upper()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|LP|L P|PLC|HOLDINGS?|GROUP|THE|NV|SA)\b", " ", s)
    return " ".join(s.split())
names["key"] = names["name"].map(norm)

bao = pd.read_csv(BAO)
bao = bao[(bao.YEARA >= 2009) & (bao.UNDERSTATEMENT == 0)]
a = pd.DataFrame({"cik": bao.CIK, "fyear": bao.YEARA, "case_id": "bao_" + bao.P_AAER.astype(str), "borderline": 0})

new = pd.read_csv(NEW)
matches, rows = [], []
for _, r in new.iterrows():
    key = norm(r.company)
    if r.aaer in OVERRIDES:
        cik, how = OVERRIDES[r.aaer], "override"
    else:
        hit = names[names.key == key]
        if hit.empty:
            first = key.split()[0] if key else ""
            hit = names[names.key.str.startswith(first + " ") | (names.key == first)] if first else hit
        cik = int(hit.cik.iloc[0]) if len(hit) == 1 else None
        how = "exact" if len(hit) == 1 else f"{len(hit)} candidates"
        matches.append(dict(aaer=r.aaer, company=r.company, how=how,
                            candidates="; ".join(f"{c}={n}" for c, n in zip(hit.cik.head(8), hit.name.head(8)))))
    if cik is None:
        continue
    for fy in str(r.fyears).split(";"):
        rows.append(dict(cik=cik, fyear=int(fy), case_id=f"new_{r.aaer}", borderline=int(r.borderline)))
pd.DataFrame(matches).to_csv(OUT.replace(".csv", "_name_matches.csv"), index=False)
lab = pd.concat([a, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["cik", "fyear"])
lab.to_csv(OUT, index=False)
print(lab.groupby("fyear").size().to_string())
print(f"{len(lab)} labelled firm-years; unmatched new cases:",
      [m["company"] for m in matches if not m["how"] == "exact"])
