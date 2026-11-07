# Section 6 extension: SEC XBRL data, fiscal 2009-2022

1. Build the financial data (about 30-60 minutes; easiest in Google Colab with build_sec_dataset_colab.ipynb):
   python build_sec_panel.py "Your Name your.email@example.com"
   -> sec_panel.csv.gz (a built copy is already in data/)
2. Build fraud labels (Bao et al. AAER file through AAER 4012 + hand-collected AAERs 4013-4605):
   python build_labels.py data/sec_panel.csv.gz ../FraudDetection/AAER_firm_year.csv new_aaer_labels.csv data/labels.csv "4110:709283,4153:885590,4277:1648428,4555:887247,4581:1015383,4440:896493,4318:1100441"
   The last argument fixes seven company-name matches by hand (renamed or re-registered companies); see data/labels_name_matches.csv.
3. Run the analysis (Tables 6-7, Figure 2 inputs):
   mkdir out && python analysis_post2017.py data/sec_panel.csv.gz data/labels.csv out
   Then: python fig_ext.py out   (Figure 2)
   Add "strict" as a fourth argument to drop the seven borderline cases.

new_aaer_labels.csv lists every included case with the misstated fiscal years (Compustat fiscal-year convention), whether it is a borderline judgment call, the basis, and a link to the SEC order.
