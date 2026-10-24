"""
12_ipssm_benchmark.py  (Stage 8, part 1)
========================================
The published IPSS-M score as a benchmark: no fitting, published weights.

Implementation check (done once while building this stage): the scores from
src/ipssm.py were compared with the official IPSS-M calculator (JavaScript,
npm package "ipssm" 1.0.4) on 300 training patients with complete data:
all 300 agreed exactly (to the calculator's 2 decimals).

IMPORTANT caveat for interpretation:
  The IPSS-M was developed on an international cohort (IWG-PM) that very
  probably overlaps with this challenge's training centres. If so, its
  weights were partly learned from these very patients, and its score on
  our training data is partly IN-SAMPLE (optimistic). The fair comparison
  is on the test centre, i.e. the leaderboard.

Outputs:
  figures/fig25_ipssm.png
  results/tables/ipssm_cv.csv              (IPSS-M on each of the 15 folds)
  results/tables/ipssm_cv_summary.csv      (paired comparisons)
  results/tables/ipssm_categories.csv
  data/processed/ipssm_scores.csv
  submissions/stage8_ipssm_score.csv

Run from the project's top folder, AFTER 09_ensemble.py (a few seconds):
    python src/12_ipssm_benchmark.py
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

import config
import cv_utils as U
import ipssm as M

plt = config.set_plot_style()
config.make_folders()

train, test, groups, folds = U.load_stage3()
y_all = U.survival_target(train)

# ===========================================================================
# 1. Scores and categories
# ===========================================================================
score_tr, contrib_tr = M.ipssm_score(train)
score_te, _ = M.ipssm_score(test)
cat_tr, cat_te = M.ipssm_category(score_tr), M.ipssm_category(score_te)
pd.DataFrame({"ipssm_score": pd.concat([score_tr, score_te]),
              "set": ["train"] * len(score_tr) + ["test"] * len(score_te)}).to_csv(
    config.DATA_PROCESSED / "ipssm_scores.csv")

print(f"IPSS-M score, training: median {score_tr.median():.2f}, range {score_tr.min():.2f} to {score_tr.max():.2f}")
print(f"IPSS-M score, test:     median {score_te.median():.2f}")
mv = M.ipssm_inputs(train)
print(f"TP53 multi-hit (IPSS-M definition): {int(mv['TP53multi'].sum())} training patients")

# Survival by IPSS-M category (training)
km_rows = []
for c in M.CATEGORIES:
    g = train[cat_tr == c]
    k = KaplanMeierFitter().fit(g["OS_YEARS"], g["OS_STATUS"])
    km_rows.append({"category": c, "train_pct": round(100 * (cat_tr == c).mean(), 1),
                    "test_pct": round(100 * (cat_te == c).mean(), 1), "patients": len(g),
                    "median_years": round(k.median_survival_time_, 2),
                    "S_at_3_years": round(float(k.survival_function_at_times(3).iloc[0]), 3)})
cats = pd.DataFrame(km_rows)
cats.to_csv(config.TABLES / "ipssm_categories.csv", index=False)
lr = multivariate_logrank_test(train["OS_YEARS"], cat_tr.astype(str), train["OS_STATUS"])
print("\nIPSS-M categories:")
print(cats.to_string(index=False))
print(f"log-rank chi2 = {lr.test_statistic:.1f} (5 df)")

# ===========================================================================
# 2. IPSS-M on the 15 centre-grouped folds, paired with our models
# ===========================================================================
rows = []
for r in range(3):
    for k in range(5):
        valid = (folds[f"repeat{r}"] == k).to_numpy()
        rows.append({"repeat": r, "fold": k,
                     "ipcw_C": U.ipcw_c(y_all[~valid], y_all[valid], score_tr[valid].to_numpy())})
ipssm_cv = pd.DataFrame(rows).set_index(["repeat", "fold"])["ipcw_C"]
ipssm_cv.to_csv(config.TABLES / "ipssm_cv.csv")

ens = pd.read_csv(config.TABLES / "ensemble_cv_folds.csv")
others = {code: ens[ens["code"] == code].set_index(["repeat", "fold"])["ipcw_C"]
          for code in ["A", "D", "G", "E1"]}
names = {"A": "IPSS-R score", "D": "Elastic-net Cox", "G": "XGBoost Cox (5 seeds)",
         "E1": "Ensemble (final)"}
summ = [{"comparison": "IPSS-M score (no fitting)", "mean_ipcw_C": round(ipssm_cv.mean(), 4),
         "sd": round(ipssm_cv.std(), 4), "mean_diff": np.nan, "sd_diff": np.nan, "folds_won": "-"}]
for code, s in others.items():
    d = s - ipssm_cv                        # positive = our model better than IPSS-M
    summ.append({"comparison": f"{names[code]} minus IPSS-M", "mean_ipcw_C": round(s.mean(), 4),
                 "sd": round(s.std(), 4), "mean_diff": round(d.mean(), 4),
                 "sd_diff": round(d.std(), 4), "folds_won": f"{int((d > 0).sum())}/15"})
summ = pd.DataFrame(summ)
summ.to_csv(config.TABLES / "ipssm_cv_summary.csv", index=False)
print("\nIPSS-M vs our models on the same 15 folds (positive diff = our model better):")
print(summ.to_string(index=False))

# ===========================================================================
# 3. Figure 25: survival by IPSS-M category, and train vs test shares
# ===========================================================================
ramp = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
for c, colour in zip(M.CATEGORIES, ramp):
    g = train[cat_tr == c]
    sf = KaplanMeierFitter().fit(g["OS_YEARS"], g["OS_STATUS"]).survival_function_
    ax.step(sf.index, sf.iloc[:, 0], where="post", color=colour, linewidth=2, label=f"{c} (n={len(g)})")
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Years since diagnosis")
ax.set_ylabel("Probability of being alive")
ax.set_title(f"Training patients by IPSS-M category\nlog-rank chi2 = {lr.test_statistic:.1f} (5 df)", fontsize=10)
ax.legend(loc="upper right", fontsize=8)

ax = axes[1]
pos = np.arange(len(M.CATEGORIES))
w = 0.38
ax.bar(pos - w / 2, cats["train_pct"], width=w, color=config.COLOR_TRAIN, label="Training")
ax.bar(pos + w / 2, cats["test_pct"], width=w, color=config.COLOR_TEST, label="Test (KYW)")
for i, (a, b) in enumerate(zip(cats["train_pct"], cats["test_pct"])):
    ax.text(i - w / 2, a + 0.8, f"{a:.0f}", ha="center", fontsize=7.5, color=config.TEXT_SOFT)
    ax.text(i + w / 2, b + 0.8, f"{b:.0f}", ha="center", fontsize=7.5, color=config.TEXT_SOFT)
ax.set_xticks(pos)
ax.set_xticklabels([c.replace(" ", "\n") for c in M.CATEGORIES], fontsize=8)
ax.set_ylabel("Patients (%)")
ax.set_ylim(0, 50)
ax.set_title("IPSS-M categories: training vs test", fontsize=10)
ax.legend(loc="upper left", fontsize=8.5)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig25_ipssm.png")
plt.close(fig)

# ===========================================================================
# 4. Submission file: the IPSS-M score of every test patient
# ===========================================================================
sub = pd.DataFrame({"risk_score": score_te.to_numpy()}, index=test.index)
sub.index.name = "ID"
assert len(sub) == 1193 and sub["risk_score"].notna().all(), "submission problem!"
sub.to_csv(config.SUBMISSIONS / "stage8_ipssm_score.csv")
print("\nSaved submissions/stage8_ipssm_score.csv")
print("\nDone: 1 figure, 3 tables, 1 submission file.")
