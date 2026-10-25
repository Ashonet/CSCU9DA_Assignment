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

# AUDIT
h, u, n_u, ragged = load(FILES[0])
print("=== demoData_unemploymentRisk.csv ===")
print(f"rows={n_u:,}  columns={len(h)}  ragged rows={ragged}")
print("PAPER REPORTS N = 140,274  ->", "MATCH" if n_u == 140_274 else "MISMATCH")
print("header:", h)
print("unnamed first column is a leaked pandas index:", u[""][:3], "...")

for k in h[1:]:
    col = u[k]
    n_miss = sum(x.strip() in MISSING_CODES for x in col)
    try:
        v = num(col)
        print(f"{k:30s} numeric  missing={n_miss}  min={np.nanmin(v):.4g}  "
              f"median={np.nanmedian(v):.4g}  max={np.nanmax(v):.4g}")
    except ValueError:
        print(f"{k:30s} text     missing={n_miss}  unique={len(set(col))}")

year, month = num(u["year"]).astype(int), num(u["month"]).astype(int)
soc = num(u["2SOC"]).astype(int)
risk, rate, count = num(u["Unemployment Risk"]), num(u["Total Unemployment Rate (%)"]), num(u["Unemployment Count"])
states = np.array(u["state"])

n_states, n_soc = len(set(states)), len(set(soc))
n_months = len(set(zip(year, month)))
expected = n_states * n_months * n_soc
print(f"\nstates={n_states} (DC present: {'District of Columbia' in set(states)})  "
      f"SOC groups={n_soc}  months={n_months} ({year.min()}-{year.max()})")
print(f"balanced panel would have {expected:,} rows; observed {n_u:,}; "
      f"absent state-month-occupation cells = {expected - n_u:,} ({(expected - n_u) / expected:.1%})")
dups = sum(v > 1 for v in Counter(zip(states, year, month, soc)).values())
print("duplicate (state, year, month, SOC) keys:", dups)
print("rows per year:", dict(sorted(Counter(year.tolist()).items())))

print(f"\nUnemployment Risk > 1 (more claimants than 100% of workforce): {(risk > 1).sum()} rows")
print(f"Unemployment Risk > 0.1: {(risk > 0.1).sum()} rows;  Count == 1: {(count == 1).sum()} rows")
big = np.argsort(-risk)[:5]
for i in big:
    print("  top risk:", states[i], year[i], month[i], "SOC", soc[i], f"risk={risk[i]:.3f} count={count[i]:.0f}")
print("Total Unemployment Rate > 15% (rows):", (rate > 15).sum(),
      " of which in 2020:", ((rate > 15) & (year == 2020)).sum())

# rate should be constant within state-month (it is a state-level covariate repeated per occupation)
sm = {}
for s, y, m, r in zip(states, year, month, rate):
    sm.setdefault((s, y, m), set()).add(round(r, 9))
print("state-months with >1 distinct 'Total Unemployment Rate':", sum(len(v) > 1 for v in sm.values()),
      "of", len(sm))
print("share of rows in 2020:", f"{(year == 2020).mean():.1%}")

h2, a, n_a, ragged2 = load(FILES[1])
print("\n=== demoData_automationByState.csv ===")
print(f"rows={n_a}  columns={len(h2)}  ragged={ragged2}")
print("distinct months present:", sorted(set(a["month"]), key=int), " rows per year:",
      dict(sorted(Counter(a["year"]).items())))
score_cols = h2[4:]
miss2 = sum(x.strip() in MISSING_CODES for k in score_cols for x in a[k])
print("missing exposure-score cells:", miss2)