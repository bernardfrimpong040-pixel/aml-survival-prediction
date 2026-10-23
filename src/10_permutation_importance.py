"""
10_permutation_importance.py  (Stage 6, part 2)
===============================================
Which kinds of information do the models rely on, measured on patients
from centres the model has NOT seen?

Permutation importance, in plain language:
  1. Fit the model on the training part of a fold.
  2. Score the validation fold normally -> baseline C-index.
  3. Shuffle ONE group of columns among the validation patients (each
     patient gets somebody else's values for those columns), score again.
  4. Importance = baseline C - shuffled C. If the model really used that
     information, scrambling it hurts; if not, nothing changes.
  Repeat the shuffle 5 times and average, over the 5 folds of repeat 0.

Why groups of features? Related features (e.g. the 18 karyotype columns)
carry overlapping information. Shuffling one alone barely hurts because
its partners still hold the information. Shuffling the whole group
measures what the model loses without that KIND of information.

Models: D. elastic-net Cox and G. XGBoost Cox (5 seeds).

Outputs:
  figures/fig23_permutation_importance.png
  results/tables/permutation_importance.csv

Run from the project's top folder (about 4-8 minutes):
    python src/10_permutation_importance.py
"""

import warnings
import numpy as np
import pandas as pd

import config
import cv_utils as U

warnings.filterwarnings("ignore")
plt = config.set_plot_style()
config.make_folders()

train, test, groups, folds = U.load_stage3()
FULL = [c for fam in ["clinical", "cytogenetic", "molecular_summary", "genes", "scores"]
        for c in groups[fam] if c != "log_MONOCYTES"]
y_all = U.survival_target(train)

# ===========================================================================
# 1. Feature groups (every one of the 84 features belongs to exactly one)
# ===========================================================================
TP53 = ["tp53_n_mut", "tp53_max_vaf", "tp53_multihit", "mut_TP53"]
BURDEN = ["mol_n_mut", "mol_n_genes", "mol_max_vaf", "mol_n_truncating"]
GROUPS = {
    "Bone-marrow blasts": ["BM_BLAST"],
    "Haemoglobin": ["HB"],
    "Platelets": ["log_PLT"],
    "White cells and neutrophils": ["log_WBC", "log_ANC"],
    "Karyotype (18 features)": groups["cytogenetic"],
    "TP53 (4 features)": TP53,
    "Mutation burden (4 features)": BURDEN,
    "Pathways, FLT3-ITD, MLL-PTD (8)": [c for c in groups["molecular_summary"]
                                        if c.startswith("path_") or c in ("mol_FLT3_ITD", "mol_MLL_PTD")],
    "Other gene indicators (44)": [c for c in groups["genes"] if c != "mut_TP53"],
    "IPSS-R score": ["ipssr_score"],
}
covered = sorted(c for cols in GROUPS.values() for c in cols)
assert covered == sorted(FULL), "every feature must be in exactly one group"
N_SHUFFLES = 5
rng = np.random.default_rng(config.SEED)

# ===========================================================================
# 2. Permutation importance on the 5 folds of repeat 0
# ===========================================================================
rows = []
for k in range(5):
    valid = (folds["repeat0"] == k).to_numpy()
    tr_part, va_part = train[~valid], train[valid]
    y_tr, y_va = y_all[~valid], y_all[valid]
    print(f"Fold {k}: fitting models ...")

    enet, alpha, _, _ = U.fit_coxnet_nested(tr_part[FULL], y_tr, tr_part["CENTER"].to_numpy())
    xgbs, depth, n_trees, _ = U.fit_xgb_nested(tr_part, FULL)
    predictors = {
        "D. Elastic-net Cox": lambda X: enet.predict(X, alpha=alpha),
        "G. XGBoost Cox": lambda X: U.xgb_risk(xgbs, X, n_trees),
    }

    for model_name, predict in predictors.items():
        X_va = va_part[FULL]
        base = U.ipcw_c(y_tr, y_va, predict(X_va))
        for group_name, cols in GROUPS.items():
            drops = []
            for _ in range(N_SHUFFLES):
                X_shuf = X_va.copy()
                order = rng.permutation(len(X_shuf))
                # Shuffle the group's columns TOGETHER (same new order for all of them),
                # so values that belong together stay together.
                X_shuf[cols] = X_va[cols].to_numpy()[order]
                drops.append(base - U.ipcw_c(y_tr, y_va, predict(X_shuf)))
            rows.append({"model": model_name, "group": group_name, "fold": k,
                         "baseline_C": round(base, 4), "mean_drop": np.mean(drops)})

perm = pd.DataFrame(rows)
table = (perm.groupby(["group", "model"])["mean_drop"]
         .agg(["mean", "std"]).round(4).unstack("model"))
table.columns = [f"{stat}_{model.split('.')[0]}" for stat, model in table.columns]
table = table.sort_values("mean_D", ascending=False)
table.to_csv(config.TABLES / "permutation_importance.csv")
print("\nDrop in IPCW C-index when the group is shuffled (mean and sd over 5 folds):")
print(table.to_string())

# ===========================================================================
# 3. Figure 23
# ===========================================================================
order = table.index[::-1]
pos = np.arange(len(order))
h = 0.38
fig, ax = plt.subplots(figsize=(7.6, 5.6))
for offset, model, colour, label in [(+h / 2, "D", config.COLOR_TRAIN, "D. Elastic-net Cox"),
                                     (-h / 2, "G", config.COLOR_TEST, "G. XGBoost Cox")]:
    vals = table.loc[order, f"mean_{model}"].to_numpy()
    errs = table.loc[order, f"std_{model}"].to_numpy()
    ax.barh(pos + offset, vals, height=h, color=colour, label=label)
    ax.errorbar(vals, pos + offset, xerr=errs, fmt="none", ecolor=config.TEXT_SOFT,
                elinewidth=1, capsize=2)
ax.axvline(0, color=config.TEXT_MAIN, linewidth=1)
ax.set_yticks(pos)
ax.set_yticklabels(order, fontsize=8.5)
ax.set_xlabel("Drop in IPCW C-index when the group is shuffled\n(held-out centres; bar = mean, line = sd over 5 folds)")
ax.set_title("Permutation importance by feature group")
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig23_permutation_importance.png")
plt.close(fig)

print("\nDone: 1 figure, 1 table.")
