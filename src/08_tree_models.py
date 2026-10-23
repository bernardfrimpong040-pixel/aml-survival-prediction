"""
08_tree_models.py  (Stage 5)
============================
Machine-learning survival models, evaluated exactly like the Cox models:
the same features, the same 15 centre-grouped folds, the same metric.

Models:
  F. Random survival forest (RSF)  - 300 survival trees, fixed sensible settings
  G. XGBoost with a Cox loss       - tree depth and number of trees chosen by
                                     an inner centre-grouped CV (nested CV)
Compared with:
  A. IPSS-R score and D. elastic-net Cox, read from Stage 4's results
     (same folds file, so the comparison is paired fold by fold).

Also:
  * how the C-index changes with the horizon tau (1, 2, 3, 5, 7 years),
  * which features the boosted trees use most,
  * submission files for F and G.

Outputs:
  figures/fig19_tree_cv.png
  figures/fig20_horizon.png
  figures/fig21_xgb_importance.png
  results/tables/tree_cv_folds.csv
  results/tables/tree_cv_summary.csv
  results/tables/tree_cv_pairwise.csv
  results/tables/cindex_by_horizon.csv
  results/tables/xgb_importance.csv
  results/tables/xgb_selected.json
  submissions/stage5_rsf.csv
  submissions/stage5_xgboost_cox.csv

Run from the project's top folder, AFTER 06_cox_models.py (about 3-6 minutes):
    python src/08_tree_models.py
"""

import json
import warnings
import numpy as np
import pandas as pd

import config
import cv_utils as U

warnings.filterwarnings("ignore")
plt = config.set_plot_style()
config.make_folders()

# ===========================================================================
# 0. Data and features (the same 84 features as the elastic-net, model D)
# ===========================================================================
train, test, groups, folds = U.load_stage3()
FULL = [c for fam in ["clinical", "cytogenetic", "molecular_summary", "genes", "scores"]
        for c in groups[fam] if c != "log_MONOCYTES"]
print(f"Features: {len(FULL)}   Training patients: {len(train)}")


# ===========================================================================
# 1. The two tree models as fit_predict functions
# ===========================================================================
def rsf(train_part, valid_part):
    model = U.make_rsf().fit(train_part[FULL], U.survival_target(train_part))
    return model.predict(valid_part[FULL])


xgb_choices = []          # remember which settings the inner CV picked in each fold


def xgb_nested(train_part, valid_part):
    model, depth, n_trees, _ = U.fit_xgb_nested(train_part, FULL)
    xgb_choices.append((depth, n_trees))
    return U.xgb_risk(model, valid_part[FULL], n_trees)


MODELS = {"F. Random survival forest": rsf, "G. XGBoost Cox (nested)": xgb_nested}

# ===========================================================================
# 2. Cross-validation (15 folds)
# ===========================================================================
results, oof = [], {}
for name, fp in MODELS.items():
    print(f"\nEvaluating {name} ...")
    res, oof_r0 = U.cross_validate(fp, train, folds, verbose_name=name)
    res.insert(0, "model", name)
    results.append(res)
    oof[name] = oof_r0
    print(f"  mean IPCW C = {res['ipcw_C'].mean():.4f} (sd {res['ipcw_C'].std():.4f})")

# Bring in the Stage 4 results on the SAME folds.
cox = pd.read_csv(config.TABLES / "cox_cv_folds.csv")
cox = cox[cox["model"].str[:2].isin(["A.", "D."])]
cv = pd.concat([cox] + results, ignore_index=True)
cv.to_csv(config.TABLES / "tree_cv_folds.csv", index=False)

print("\nDepth / number of trees chosen by the inner CV in the 15 folds:")
print(pd.Series([f"depth {d}, {n} trees" for d, n in xgb_choices]).value_counts().to_string())


def fold_scores(prefix):
    return cv[cv["model"].str.startswith(prefix)].set_index(["repeat", "fold"])["ipcw_C"]


names = ["A.", "D.", "F.", "G."]
labels = {p: cv.loc[cv["model"].str.startswith(p), "model"].iloc[0] for p in names}
ref_A, ref_D = fold_scores("A."), fold_scores("D.")
summary = []
for p in names:
    s = fold_scores(p)
    summary.append({
        "model": labels[p], "mean_ipcw_C": round(s.mean(), 4), "sd": round(s.std(), 4),
        "min": round(s.min(), 4), "max": round(s.max(), 4),
        "paired_vs_IPSSR": round((s - ref_A).mean(), 4),
        "paired_vs_elasticnet": round((s - ref_D).mean(), 4),
        "folds_won_vs_elasticnet": "-" if p in ("A.", "D.") else f"{int(((s - ref_D) > 0).sum())}/15",
    })
summary = pd.DataFrame(summary)
summary.to_csv(config.TABLES / "tree_cv_summary.csv", index=False)
print("\nCross-validation summary (15 centre-grouped folds):")
print(summary.to_string(index=False))

pairs = []
for a, b in [("F.", "D."), ("G.", "D."), ("G.", "F.")]:
    d = fold_scores(a) - fold_scores(b)
    pairs.append({"comparison": f"{a[0]} - {b[0]}", "mean_diff": round(d.mean(), 4),
                  "sd_diff": round(d.std(), 4), "folds_won": f"{int((d > 0).sum())}/15"})
pairs = pd.DataFrame(pairs)
pairs.to_csv(config.TABLES / "tree_cv_pairwise.csv", index=False)
print("\nPaired comparisons:")
print(pairs.to_string(index=False))

# ---- Figure 19: fold scores and paired differences with the elastic-net -----
short = {"A.": "A. IPSS-R", "D.": "D. Elastic-net Cox", "F.": "F. Random\nsurvival forest",
         "G.": "G. XGBoost Cox"}
rng = np.random.default_rng(config.SEED)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
for i, p in enumerate(names):
    vals = fold_scores(p).to_numpy()
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=22, color=config.COLOR_TRAIN,
               alpha=0.55, edgecolor="white", linewidth=0.6)
    ax.plot([i - 0.28, i + 0.28], [vals.mean()] * 2, color=config.TEXT_MAIN, linewidth=2.2)
    ax.text(i + 0.31, vals.mean(), f"{vals.mean():.3f}", va="center", fontsize=8, color=config.TEXT_MAIN)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([short[p] for p in names], fontsize=8.5)
ax.set_xlim(-0.5, len(names) - 0.2)
ax.set_ylabel("IPCW C-index on validation fold")
ax.set_title("Each dot = one of 15 folds; bar = mean")
ax.grid(axis="x", visible=False)

ax = axes[1]
for i, p in enumerate(["F.", "G."]):
    d = (fold_scores(p) - ref_D).to_numpy()
    colours = [config.COLOR_TRAIN if v > 0 else config.COLOR_TEST for v in d]
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=22, c=colours, alpha=0.8,
               edgecolor="white", linewidth=0.6)
    ax.plot([i - 0.28, i + 0.28], [d.mean()] * 2, color=config.TEXT_MAIN, linewidth=2.2)
    ax.text(i + 0.31, d.mean(), f"{d.mean():+.4f}\n{(d > 0).sum()}/15 won", va="center",
            fontsize=8, color=config.TEXT_MAIN)
ax.axhline(0, color=config.TEXT_MAIN, linewidth=1)
ax.set_xticks([0, 1])
ax.set_xticklabels([short["F."], short["G."]], fontsize=8.5)
ax.set_xlim(-0.5, 1.75)
ax.set_ylabel("Model C minus elastic-net C, same fold")
ax.set_title("Paired difference with the elastic-net Cox\n(blue = tree model better, orange = worse)")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig19_tree_cv.png")
plt.close(fig)

# ===========================================================================
# 3. C-index by horizon tau (out-of-fold predictions, repeat 0)
# ===========================================================================
# tau = 1 asks "who dies in the first year?"; tau = 7 is the challenge metric.
# Each fold's predictions come from a model that never saw that fold's centres.
oof_cox = pd.read_csv(config.DATA_PROCESSED / "oof_repeat0.csv", index_col="ID")
oof_all = pd.DataFrame({
    "A.": oof_cox[[c for c in oof_cox.columns if c.startswith("A.")][0]],
    "D.": oof_cox[[c for c in oof_cox.columns if c.startswith("D.")][0]],
    "F.": oof["F. Random survival forest"],
    "G.": oof["G. XGBoost Cox (nested)"],
}).loc[train.index]
y_all = U.survival_target(train)
TAUS = [1, 2, 3, 5, 7]
rows = []
for p in names:
    for tau in TAUS:
        scores = []
        for k in range(5):
            valid = (folds["repeat0"] == k).to_numpy()
            scores.append(U.ipcw_c(y_all[~valid], y_all[valid], oof_all.loc[valid, p], tau=tau))
        rows.append({"model": labels[p], "tau_years": tau, "mean_ipcw_C": round(np.mean(scores), 4)})
horizon = pd.DataFrame(rows)
horizon.to_csv(config.TABLES / "cindex_by_horizon.csv", index=False)
print("\nIPCW C-index by horizon tau (repeat 0, mean of 5 folds):")
print(horizon.pivot(index="model", columns="tau_years", values="mean_ipcw_C").to_string())

fig, ax = plt.subplots(figsize=(7.2, 4.0))
style = {"A.": (config.TEXT_SOFT, "--"), "D.": (config.COLOR_TRAIN, "-"),
         "F.": (config.COLOR_GROUP3, "-"), "G.": (config.COLOR_TEST, "-")}
for p in names:
    h = horizon[horizon["model"] == labels[p]]
    colour, ls = style[p]
    ax.plot(h["tau_years"], h["mean_ipcw_C"], color=colour, linestyle=ls, linewidth=2,
            marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5,
            label=short[p].replace("\n", " "))
ax.set_xticks(TAUS)
ax.set_xlabel("Horizon tau (years)")
ax.set_ylabel("IPCW C-index (out-of-fold)")
ax.set_title("Ranking accuracy by horizon: short-term vs long-term deaths")
ax.legend(loc="lower left", fontsize=8.5)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig20_horizon.png")
plt.close(fig)

# ===========================================================================
# 4. Final models on ALL training patients -> submissions
# ===========================================================================
print("\nFitting final models on all 3,173 training patients ...")
final_rsf = U.make_rsf().fit(train[FULL], y_all)
final_xgb, depth, n_trees, inner = U.fit_xgb_nested(train, FULL, n_inner=5)
with open(config.TABLES / "xgb_selected.json", "w", encoding="utf-8") as f:
    json.dump({"max_depth": depth, "n_trees": n_trees, "learning_rate": 0.03,
               "inner_cv_ipcw_C": round(float(inner), 4)}, f, indent=2)
print(f"Final XGBoost: depth {depth}, {n_trees} trees, inner CV C = {inner:.4f}")

subs = {"stage5_rsf.csv": final_rsf.predict(test[FULL]),
        "stage5_xgboost_cox.csv": U.xgb_risk(final_xgb, test[FULL], n_trees)}
for file_name, risk in subs.items():
    sub = pd.DataFrame({"risk_score": np.asarray(risk, dtype=float)}, index=test.index)
    sub.index.name = "ID"
    assert len(sub) == 1193 and sub["risk_score"].notna().all(), "submission problem!"
    sub.to_csv(config.SUBMISSIONS / file_name)
    print(f"Saved submissions/{file_name}: {len(sub)} rows")

# ===========================================================================
# 5. What do the boosted trees use? (gain importance)
# ===========================================================================
# "Gain" = how much the Cox loss improved, in total, from splits on a feature.
# It shows what the model USES, not cause and effect; correlated features
# share the credit. Stage 6 will add permutation importance on held-out data.
gain = pd.Series(final_xgb.get_booster().get_score(importance_type="total_gain"))
gain = (gain / gain.sum() * 100).reindex(FULL).fillna(0).sort_values(ascending=False)
gain.round(2).rename("percent_of_total_gain").to_csv(config.TABLES / "xgb_importance.csv")
print("\nTop 10 features by share of total gain (%):")
print(gain.head(10).round(1).to_string())

top = gain.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.2, 5.8))
ax.barh(top.index, top.to_numpy(), color=config.COLOR_TRAIN, height=0.7)
for i, v in enumerate(top):
    ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7.5, color=config.TEXT_SOFT)
ax.set_xlim(0, top.max() * 1.15)
ax.set_xlabel("Share of total gain (%)")
ax.set_title("XGBoost Cox: 20 most used features")
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig21_xgb_importance.png")
plt.close(fig)

print("\nDone: 3 figures, 6 tables, 2 submission files.")
