# Tax signals and machine-learning fraud detection: replication code

Code and data for Pelumi K. Bello, "Do Tax Signals Improve Machine-Learning Detection of Accounting Fraud? Evidence from U.S. Public Firms Using Explainable AI" (working paper, 2026), and for the Book–Tax Signal Screen prototype built on it.

Live prototype: https://lulumypb-eng.github.io/book-tax-signal-screen/

| Folder | What it reproduces |
| --- | --- |
| this folder | Main sample, fiscal 1991–2014 (Sections 3–5 and 7): Tables 1–5 and 8, Figures 1, 3 and 4 |
| `extension_2009_2022/` | SEC XBRL sample, fiscal 2009–2022 (Section 6): Tables 6–7, Figure 2 |
| `book_tax_signal_screen/` | The Book–Tax Signal Screen web page |

## Install

Python 3.10 or later.

```
pip install pandas numpy scipy scikit-learn imbalanced-learn xgboost shap statsmodels matplotlib
```

The paper used Python 3.13, scikit-learn 1.9.1, imbalanced-learn 0.14.2, XGBoost 3.4.1, SHAP 0.53.0 and statsmodels 0.15.0. All random seeds are fixed; other package versions may shift results slightly.

## 1. Main sample (1991–2014)

Get the public data of Bao et al. (2020, Journal of Accounting Research), and cite that paper if you use it:

```
git clone https://github.com/JarFraud/FraudDetection.git
mkdir out
python analysis.py FraudDetection/data_FraudDetection_JAR2020.csv out   # Tables 1, 2, 4, 5 (about 20-30 minutes)
python regress.py out                                                    # Table 3
python explain.py out                                                    # Section 7 SHAP results, Table 8, multi-seed check (about 20 minutes)
python figs.py out                                                       # Figures 1, 3, 4
```

| Paper item | File in `out/` |
| --- | --- |
| Table 1 | `results_by_year.csv` (columns `n`, `frauds`) |
| Table 2 | `tax_descriptives.csv`, `sample_counts.json` |
| Table 3 | `regressions.json` |
| Table 4 | `summary_2003_2008.csv`; 2003–2014 in `summary_2003_2014.csv` |
| Table 5 | `bootstrap.json` |
| Multi-seed RUSBoost check (Section 5.2) | `multiseed.csv`, `multiseed_summary.csv` |
| Section 7.1 SHAP ranks; Section 7.3 stability | `shap_importance_by_year.csv`, `shap_stability.json` |
| Section 7.2 local explanation | `shap_local.json` |
| Table 8 | `sparse_by_year.csv` |
| Figure 1 (AUC by year) | `fig3.png` |
| Figure 3 (global SHAP) | `fig1.png` |
| Figure 4 (local SHAP) | `fig2.png` |

## 2. SEC XBRL sample (2009–2022)

See `extension_2009_2022/README.md`. The built financial panel (`data/sec_panel.csv.gz`) and fraud labels (`data/labels.csv`) are included, so you can skip straight to the analysis:

```
cd extension_2009_2022
mkdir out
python analysis_post2017.py data/sec_panel.csv.gz data/labels.csv out   # Tables 6-7, Section 6.3 SHAP results
python fig_ext.py out                                                   # Figure 2 (fig_ext.png)
```

`new_aaer_labels.csv` lists every hand-collected SEC enforcement case (AAERs 4013–4605) with the misstated fiscal years, the basis for inclusion and a link to the SEC order.

## 3. Book–Tax Signal Screen

```
cd book_tax_signal_screen
python improve.py              # optional: compares model versions, 2015-2020 (about 20 minutes)
python history.py plusB rus    # scores companies' 10-Ks for fiscal 2019-2024 (about 20 minutes)
python add_bt.py               # adds the book-tax check
python make_page.py            # writes index.html
```

The page is a research prototype and screening aid. A high rank is not an allegation of wrongdoing.

## Data sources

- Bao, Y., Ke, B., Li, B., Yu, Y. J., & Zhang, J. (2020). Detecting accounting fraud in publicly traded U.S. firms using a machine learning approach. Journal of Accounting Research, 58(1), 199–235.
- U.S. Securities and Exchange Commission, Financial Statement Data Sets (https://www.sec.gov/data/financial-statement-data-sets) and Accounting and Auditing Enforcement Releases.

## Use of AI

An AI assistant (Claude, Anthropic) helped write this code. See the paper's declarations.

## License

Code: MIT License. Data remain subject to their original sources' terms.
