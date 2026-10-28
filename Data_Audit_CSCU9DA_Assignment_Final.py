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

# EDA
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#2a6f97", "#d1495b"

# Fig 1: distribution of the target (log10 risk), as used in the paper's regressions
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].hist(risk[risk <= 0.05], bins=80, color=BLUE)
ax[0].set(title="Raw unemployment risk (<=0.05)", xlabel="share of occupation receiving UI", ylabel="rows")
ax[1].hist(np.log10(risk), bins=80, color=BLUE)
ax[1].axvline(0, color=ORANGE, ls="--", lw=1)
ax[1].set(title="log10(risk): approx. symmetric", xlabel="log10(risk); dashed = risk of 1")
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_risk_distribution.png"); plt.close(fig)

# Fig 2: mean monthly risk over time, by SOC group (top/bottom) + national trend
tkey = year * 12 + (month - 1)
t_all = np.array(sorted(set(tkey)))
def series(mask):
    return np.array([np.nanmedian(risk[mask & (tkey == t)]) if (mask & (tkey == t)).any() else np.nan for t in t_all])
med_by_soc = {s: np.nanmedian(risk[soc == s]) for s in set(soc)}
order = sorted(med_by_soc, key=med_by_soc.get)
fig, ax = plt.subplots(figsize=(8, 3.2))
x = 2010 + (t_all - 2010 * 12) / 12
for s in set(soc):
    ax.plot(x, series(soc == s), color="#bbbbbb", lw=0.7)
for s, c, lab in [(order[0], BLUE, f"SOC {order[0]} (lowest median)"), (order[-1], ORANGE, f"SOC {order[-1]} (highest median)")]:
    ax.plot(x, series(soc == s), color=c, lw=1.8, label=lab)
ax.set_yscale("log"); ax.axvline(2020.17, color="k", ls=":", lw=1)
ax.text(2020.1, ax.get_ylim()[1], "COVID-19", ha="right", va="top", fontsize=8)
ax.set(title="Median unemployment risk by 2-digit SOC group (grey = all 22 groups)",
       xlabel="year", ylabel="median risk (log scale)"); ax.legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_risk_over_time.png"); plt.close(fig)

# Fig 3: correlation among the 11 exposure scores (the motivation for an ensemble / LASSO)
M = np.column_stack([num(a[k]) for k in score_cols])
C = np.corrcoef(M, rowvar=False)
fig, ax = plt.subplots(figsize=(6.4, 5.2))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(score_cols))); ax.set_yticks(range(len(score_cols)))
ax.set_xticklabels(score_cols, rotation=75, ha="right", fontsize=7); ax.set_yticklabels(score_cols, fontsize=7)
for i in range(len(score_cols)):
    for j in range(len(score_cols)):
        ax.text(j, i, f"{C[i, j]:.1f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, shrink=0.8); ax.set_title("Pearson correlation of AI/automation exposure scores\n(state-month level, n=549)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_exposure_correlations.png"); plt.close(fig)

off = C[np.triu_indices_from(C, 1)]
print(f"\nexposure scores: median |r| = {np.median(np.abs(off)):.2f}; "
      f"{(np.abs(off) > 0.9).sum()} of {len(off)} pairs have |r| > 0.9; "
      f"{(off < -0.5).sum()} pairs correlate below -0.5")
print("figures written to", FIG)
