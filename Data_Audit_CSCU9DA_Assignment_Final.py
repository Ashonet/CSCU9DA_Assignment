"""
Data & code audit: Frank, Ahn & Moro (2025), "AI exposure predicts unemployment risk", PNAS Nexus, doi:10.1093/pnasnexus/pgaf107.

Raw data: https://github.com/mrfrank8176/unemploymentRisk  (demoData_*.csv)
Usage:  python Data_Audit_CSCU9DA_Final            (downloads the CSVs into ./data if absent)
Needs:  numpy, matplotlib  
"""
import csv
import os
import urllib.request
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "https://raw.githubusercontent.com/mrfrank8176/unemploymentRisk/main/"
FILES = ["demoData_unemploymentRisk.csv", "demoData_automationByState.csv"]
DATA, FIG = "data", "figures"
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
for f in FILES:
    if not os.path.exists(os.path.join(DATA, f)):
        urllib.request.urlretrieve(BASE + f, os.path.join(DATA, f))

MISSING_CODES = {"", "NA", "N/A", "nan", "NaN", "-999", "-9999", "."}


def load(fname):
    # Return header and dict of columns (strings); first unnamed column is the pandas index.
    with open(os.path.join(DATA, fname), encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    ragged = sum(len(r) != len(header) for r in body)
    cols = {h: [r[i] for r in body] for i, h in enumerate(header)}
    return header, cols, len(body), ragged


def num(col):
    return np.array([np.nan if x.strip() in MISSING_CODES else float(x) for x in col])