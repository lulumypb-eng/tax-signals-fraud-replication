"""
Build an annual firm-year panel of financial-statement items from the SEC
Financial Statement Data Sets (XBRL), fiscal years 2009-2024.

Source: https://www.sec.gov/data/financial-statement-data-sets
Output: sec_panel.csv.gz  (one row per CIK x fiscal year, original 10-K filings)
"""
import io, os, sys, time, zipfile
import numpy as np, pandas as pd, requests

USER_AGENT = sys.argv[1] if len(sys.argv) > 1 else "Research Project research@example.com"
OUT = "sec_panel.csv.gz"
BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{}.zip"
QUARTERS = [f"{y}q{q}" for y in range(2009, 2027) for q in range(1, 5)]
QUARTERS = [x for x in QUARTERS if "2009q2" <= x <= "2026q1"]

# Bao et al. (2020) raw items -> candidate XBRL tags, in priority order.
TAGS = {
    "act":   ["AssetsCurrent"],
    "ap":    ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    "at":    ["Assets"],
    "ceq":   ["StockholdersEquity"],
    "che":   ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"],
    "cogs":  ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold", "CostOfServices"],
    "csho":  ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
    "dlc":   ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"],
    "dltis": ["ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromIssuanceOfDebt"],
    "dltt":  ["LongTermDebtNoncurrent", "LongTermDebt"],
    "dp":    ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet", "Depreciation"],
    "ib":    ["IncomeLossFromContinuingOperations", "NetIncomeLoss", "ProfitLoss"],
    "invt":  ["InventoryNet"],
    "ivao":  ["LongTermInvestments", "OtherLongTermInvestments"],
    "ivst":  ["ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "lct":   ["LiabilitiesCurrent"],
    "lt":    ["Liabilities"],
    "ni":    ["NetIncomeLoss", "ProfitLoss"],
    "ppegt": ["PropertyPlantAndEquipmentGross"],
    "pstk":  ["PreferredStockValue"],
    "re":    ["RetainedEarningsAccumulatedDeficit"],
    "rect":  ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "sale":  ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "sstk":  ["ProceedsFromIssuanceOfCommonStock"],
    "txp":   ["AccruedIncomeTaxesCurrent", "TaxesPayableCurrent"],
    "txt":   ["IncomeTaxExpenseBenefit"],
    "xint":  ["InterestExpense", "InterestExpenseDebt"],
    # Extra tax items not in the Bao data
    "pi":    ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
              "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "txc":   ["CurrentIncomeTaxExpenseBenefit"],
    "txdi":  ["DeferredIncomeTaxExpenseBenefit", "DeferredIncomeTaxesAndTaxCredits"],
    "txpd":  ["IncomeTaxesPaidNet", "IncomeTaxesPaid"],
    "utb":   ["UnrecognizedTaxBenefits"],
}
STOCK = {"act", "ap", "at", "ceq", "che", "csho", "dlc", "dltt", "invt", "ivao", "ivst", "lct", "lt",
         "ppegt", "pstk", "re", "rect", "txp", "utb"}
ALLTAGS = sorted({t for v in TAGS.values() for t in v})

def fetch(q, tries=4):
    for i in range(tries):
        try:
            r = requests.get(BASE.format(q), headers={"User-Agent": USER_AGENT}, timeout=180)
            if r.status_code == 200:
                return r.content
            print(f"  {q}: HTTP {r.status_code}")
            if r.status_code == 404:
                return None
        except Exception as e:
            print(f"  {q}: {e}")
        time.sleep(5 * (i + 1))
    return None

rows = []
for q in QUARTERS:
    t0 = time.time()
    blob = fetch(q)
    if blob is None:
        print(f"{q}: skipped"); continue
    z = zipfile.ZipFile(io.BytesIO(blob))
    sub = pd.read_csv(z.open("sub.txt"), sep="\t", dtype=str,
                      usecols=lambda c: c in {"adsh", "cik", "name", "sic", "countryba", "form", "period", "fy", "fp", "filed", "afs"})
    sub = sub[sub["form"] == "10-K"]
    if sub.empty:
        print(f"{q}: no 10-K"); continue
    keep = set(sub["adsh"])
    parts = []
    for ch in pd.read_csv(z.open("num.txt"), sep="\t", dtype=str, chunksize=2_000_000,
                          on_bad_lines="skip", quoting=3):
        ch = ch[ch["adsh"].isin(keep) & ch["tag"].isin(ALLTAGS)]
        if "coreg" in ch.columns:
            ch = ch[ch["coreg"].isna()]
        if "segments" in ch.columns:
            ch = ch[ch["segments"].isna()]
        parts.append(ch[["adsh", "tag", "ddate", "qtrs", "uom", "value"]])
    num = pd.concat(parts)
    num = num.merge(sub[["adsh", "period"]], on="adsh")
    num = num[num["ddate"] == num["period"]]
    num["value"] = pd.to_numeric(num["value"], errors="coerce")
    num = num.dropna(subset=["value"])
    num = num[num["qtrs"].isin(["0", "4"])]
    wide = {}
    for item, tags in TAGS.items():
        want_q = "0" if item in STOCK else "4"
        uom = "shares" if item == "csho" else "USD"
        sel = num[(num["tag"].isin(tags)) & (num["qtrs"] == want_q) & (num["uom"] == uom)].copy()
        sel["prio"] = sel["tag"].map({t: i for i, t in enumerate(tags)})
        sel = sel.sort_values(["adsh", "prio"]).drop_duplicates("adsh")
        wide[item] = sel.set_index("adsh")["value"]
    w = pd.DataFrame(wide)
    w = sub.set_index("adsh").join(w, how="left").reset_index()
    rows.append(w)
    print(f"{q}: {len(sub):,} 10-Ks, {time.time()-t0:.0f}s", flush=True)
    time.sleep(1)

df = pd.concat(rows, ignore_index=True)
df["period"] = pd.to_datetime(df["period"], format="%Y%m%d", errors="coerce")
# Compustat fiscal-year convention: year-ends Jan-May belong to the prior fiscal year
df["fyear"] = np.where(df["period"].dt.month <= 5, df["period"].dt.year - 1, df["period"].dt.year)
df = df[df["fyear"].between(2009, 2024)]
# One original 10-K per firm-year: the first filed
df = df.sort_values(["cik", "fyear", "filed"]).drop_duplicates(["cik", "fyear"], keep="first")
df.to_csv(OUT, index=False, compression="gzip")
print(f"Done: {len(df):,} firm-years, {df['cik'].nunique():,} firms -> {OUT}")
print(df.groupby("fyear").size().to_string())
