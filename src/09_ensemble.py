"""
09_ensemble.py  (Stage 6, part 1)
=================================
Combines the best models into ensembles and checks, fold by fold, whether
the combination beats the elastic-net Cox.

How the ensemble works (rank averaging):
  The three models give scores on different scales (a linear predictor, a
  cumulative hazard, a boosted log-risk). Only the ORDER matters for the
  metric, so each model's scores are turned into ranks between 0 and 1
  inside the group of patients being scored, and the ranks are averaged.
  Example with 4 patients:
      elastic-net scores  1.2, -0.3, 0.4, 2.0  -> ranks 0.75, 0.25, 0.50, 1.00
      XGBoost scores      0.9,  0.1, 0.8, 0.5  -> ranks 1.00, 0.25, 0.75, 0.50
      average rank                              ->       0.875, 0.25, 0.625, 0.75
  The models disagree about patients 1 and 4; the average splits the difference.

No model is refitted here: the out-of-fold predictions saved by
06_cox_models.py and 08_tree_models.py are combined, so each ensemble is
scored on exactly the same 15 validation folds as the single models.

Outputs:
  figures/fig22_ensemble_cv.png
  results/tables/ensemble_cv_folds.csv
  results/tables/ensemble_cv_summary.csv
  results/tables/oof_correlations.csv
  submissions/stage6_ensemble_enet_xgb.csv
  submissions/stage6_ensemble_all3.csv

Run from the project's top folder, AFTER 06 and 08 (takes a few seconds):
    python src/09_ensemble.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
import cv_utils as U

plt = config.set_plot_style()
config.make_folders()

train, test, groups, folds = U.load_stage3()
y_all = U.survival_target(train)

# ===========================================================================
# 1. Load the saved out-of-fold predictions (long form) and reshape
# ===========================================================================
long = pd.concat([pd.read_csv(config.DATA_PROCESSED / "oof_cox.csv"),
                  pd.read_csv(config.DATA_PROCESSED / "oof_trees.csv")], ignore_index=True)
CODES = {"A": "A.", "D": "D.", "F": "F.", "G": "G."}          # the models we use
long["code"] = long["model"].str[0]
long = long[long["code"].isin(CODES)]
# One table per repeat: rows = patients, columns = model codes.
pred = {r: long[long["repeat"] == f"repeat{r}"].pivot(index="ID", columns="code", values="risk")
            .loc[train.index] for r in range(3)}

NAMES = {
    "A": "A. IPSS-R score",
    "D": "D. Elastic-net Cox",
    "F": "F. Random survival forest",
    "G": "G. XGBoost Cox (5 seeds)",
    "E1": "E1. Ensemble: elastic-net + XGBoost",
    "E2": "E2. Ensemble: elastic-net + XGBoost + forest",
}
ENSEMBLES = {"E1": ["D", "G"], "E2": ["D", "G", "F"]}


def rank01(x):
    """Ranks scaled to (0, 1]: the patient with the highest score gets 1."""
    return pd.Series(x).rank(method="average").to_numpy() / len(x)


# ===========================================================================
# 2. Score every model and ensemble on the same 15 folds
# ===========================================================================
rows = []
for r in range(3):
    for k in range(5):
        valid = (folds[f"repeat{r}"] == k).to_numpy()
        p = pred[r][valid]
        scores = {c: p[c].to_numpy() for c in ["A", "D", "F", "G"]}
        for e, members in ENSEMBLES.items():
            # Ranks are computed INSIDE the validation fold, then averaged.
            scores[e] = np.mean([rank01(p[m].to_numpy()) for m in members], axis=0)
        for code, risk in scores.items():
            rows.append({"model": NAMES[code], "code": code, "repeat": r, "fold": k,
                         "ipcw_C": U.ipcw_c(y_all[~valid], y_all[valid], risk)})
cv = pd.DataFrame(rows)
cv.to_csv(config.TABLES / "ensemble_cv_folds.csv", index=False)


def scores_of(code):
    return cv[cv["code"] == code].set_index(["repeat", "fold"])["ipcw_C"]


ref = scores_of("D")
summary = []
for code in NAMES:
    s = scores_of(code)
    d = s - ref
    summary.append({"model": NAMES[code], "mean_ipcw_C": round(s.mean(), 4), "sd": round(s.std(), 4),
                    "paired_vs_elasticnet": round(d.mean(), 4), "sd_paired": round(d.std(), 4),
                    "folds_won_vs_elasticnet": "-" if code in ("A", "D") else f"{int((d > 0).sum())}/15"})
summary = pd.DataFrame(summary)
summary.to_csv(config.TABLES / "ensemble_cv_summary.csv", index=False)
print("Cross-validation summary (15 centre-grouped folds):")
print(summary.to_string(index=False))

# E1 against XGBoost alone: does adding the elastic-net help the trees?
d = scores_of("E1") - scores_of("G")
print(f"\nE1 - G: mean {d.mean():+.4f}, sd {d.std():.4f}, folds won {(d > 0).sum()}/15")

# ===========================================================================
# 3. How similar are the models on held-out patients?
# ===========================================================================
# Average Spearman correlation of the risk scores inside each validation fold.
codes = ["D", "F", "G"]
corr = pd.DataFrame(0.0, index=codes, columns=codes)
for r in range(3):
    for k in range(5):
        valid = (folds[f"repeat{r}"] == k).to_numpy()
        p = pred[r][valid]
        corr += p[codes].corr(method="spearman") / 15
corr.index = corr.columns = [NAMES[c] for c in codes]
corr.round(3).to_csv(config.TABLES / "oof_correlations.csv")
print("\nMean Spearman correlation of out-of-fold risk scores:")
print(corr.round(3).to_string())

# ===========================================================================
# 4. Figure 22: paired differences with the elastic-net
# ===========================================================================
show = ["F", "G", "E1", "E2"]
short = {"F": "F. Random\nsurvival forest", "G": "G. XGBoost\n(5 seeds)",
         "E1": "E1. Elastic-net\n+ XGBoost", "E2": "E2. Elastic-net\n+ XGBoost + forest"}
rng = np.random.default_rng(config.SEED)
fig, ax = plt.subplots(figsize=(7.6, 4.2))
for i, code in enumerate(show):
    dd = (scores_of(code) - ref).to_numpy()
    colours = [config.COLOR_TRAIN if v > 0 else config.COLOR_TEST for v in dd]
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(dd)), dd, s=24, c=colours, alpha=0.8,
               edgecolor="white", linewidth=0.6)
    ax.plot([i - 0.28, i + 0.28], [dd.mean()] * 2, color=config.TEXT_MAIN, linewidth=2.2)
    ax.text(i + 0.31, dd.mean(), f"{dd.mean():+.4f}\n{(dd > 0).sum()}/15", va="center",
            fontsize=8, color=config.TEXT_MAIN)
ax.axhline(0, color=config.TEXT_MAIN, linewidth=1)
ax.set_xticks(range(len(show)))
ax.set_xticklabels([short[c] for c in show], fontsize=8.5)
ax.set_xlim(-0.5, len(show) - 0.15)
ax.set_ylabel("Model C minus elastic-net C, same fold")
ax.set_title("Single models and ensembles vs the elastic-net Cox\n(blue = better, orange = worse)")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig22_ensemble_cv.png")
plt.close(fig)

# ===========================================================================
# 5. Ensemble submission files (from the saved single-model submissions)
# ===========================================================================
sub_files = {"D": "stage4_elasticnet_cox.csv", "F": "stage5_rsf.csv", "G": "stage5_xgboost_cox.csv"}
test_risk = {c: pd.read_csv(config.SUBMISSIONS / f, index_col="ID")["risk_score"].loc[test.index]
             for c, f in sub_files.items()}
for e, file_name in [("E1", "stage6_ensemble_enet_xgb.csv"), ("E2", "stage6_ensemble_all3.csv")]:
    risk = np.mean([rank01(test_risk[m].to_numpy()) for m in ENSEMBLES[e]], axis=0)
    sub = pd.DataFrame({"risk_score": risk}, index=test.index)
    sub.index.name = "ID"
    assert len(sub) == 1193 and sub["risk_score"].notna().all(), "submission problem!"
    sub.to_csv(config.SUBMISSIONS / file_name)
    print(f"Saved submissions/{file_name}")

print("\nTest-centre Spearman correlations of the single-model submissions:")
for a, b in [("D", "G"), ("D", "F"), ("G", "F")]:
    print(f"  {a}-{b}: {spearmanr(test_risk[a], test_risk[b])[0]:.3f}")

print("\nDone: 1 figure, 3 tables, 2 submission files.")
