"""
11_final_summary.py  (Stage 7)
==============================
Final checks for the write-up:
  1. The leaderboard record: cross-validation score, the prediction written
     down BEFORE uploading, and the public score of every submission.
  2. How often TP53 multi-hit and a complex karyotype occur in the SAME
     patients (the Stage 6 question about co-occurrence).
  3. Does XGBoost rank the highest-risk patients worse than the elastic-net?
     (A test of one possible explanation for XGBoost's lower public score.)

Outputs:
  figures/fig24_leaderboard.png
  results/tables/leaderboard.csv
  results/tables/tp53_cooccurrence.csv
  results/tables/subgroup_xgb_vs_enet.csv

Run from the project's top folder, AFTER 06 and 08 (takes a few seconds):
    python src/11_final_summary.py
"""

import numpy as np
import pandas as pd

import config
import cv_utils as U

plt = config.set_plot_style()
config.make_folders()

train, test, groups, folds = U.load_stage3()
y_all = U.survival_target(train)

# ===========================================================================
# 1. Leaderboard record (typed in by hand after each upload)
# ===========================================================================
# cv        = mean IPCW C-index over the 15 centre-grouped folds (author's run)
# predicted = public score predicted BEFORE uploading (None = no number written)
# public    = score shown by Challenge Data
LEADERBOARD = [
    {"date": "2026-10-23", "file": "stage4_elasticnet_cox.csv", "model": "Elastic-net Cox",
     "cv": 0.7218, "predicted": None, "public": 0.7543},
    {"date": "2026-10-23", "file": "stage4_ipssm_cox.csv", "model": "IPSS-M-style Cox",
     "cv": 0.7159, "predicted": None, "public": 0.7374},
    {"date": "2026-10-24", "file": "stage6_ensemble_enet_xgb.csv", "model": "Ensemble (final)",
     "cv": 0.7250, "predicted": 0.7575, "public": 0.7552},
    {"date": "2026-10-24", "file": "stage5_xgboost_cox.csv", "model": "XGBoost Cox (5 seeds)",
     "cv": 0.7244, "predicted": 0.7569, "public": 0.7504},
]
lb = pd.DataFrame(LEADERBOARD)
lb["public_minus_cv"] = (lb["public"] - lb["cv"]).round(4)
lb["prediction_error"] = (lb["public"] - lb["predicted"]).round(4)
lb.to_csv(config.TABLES / "leaderboard.csv", index=False)
print("Leaderboard record:")
print(lb.drop(columns=["file"]).to_string(index=False))

# ---- Figure 24: cross-validation vs public score ----------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
# Where each label goes (points offset) and which side of the dot it sits on.
offsets = {"Elastic-net Cox": ((-10, 0), "right"), "IPSS-M-style Cox": ((8, -4), "left"),
           "Ensemble (final)": ((8, 6), "left"), "XGBoost Cox (5 seeds)": ((8, -10), "left")}
for _, row in lb.iterrows():
    final = row["model"].startswith("Ensemble")
    colour = config.COLOR_TEST if final else config.COLOR_TRAIN
    if pd.notna(row["predicted"]):
        # hollow marker = prediction; thin line from prediction to the real score
        ax.plot([row["cv"], row["cv"]], [row["predicted"], row["public"]],
                color=config.TEXT_SOFT, linewidth=1, linestyle=":")
        ax.plot(row["cv"], row["predicted"], "o", markersize=9, markerfacecolor="white",
                markeredgecolor=colour, markeredgewidth=2)
    ax.plot(row["cv"], row["public"], "o", markersize=9, color=colour,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"{row['model']}  {row['public']:.4f}", (row["cv"], row["public"]),
                xytext=offsets[row["model"]][0], textcoords="offset points", fontsize=8.5,
                color=config.TEXT_MAIN, va="center", ha=offsets[row["model"]][1])
ax.plot([], [], "o", markersize=8, markerfacecolor="white", markeredgecolor=config.TEXT_SOFT,
        markeredgewidth=2, linestyle="none", label="predicted before upload")
ax.plot([], [], "o", markersize=8, color=config.TEXT_SOFT, linestyle="none", label="public score")
ax.legend(loc="lower right", fontsize=8.5)
ax.set_xlim(0.714, 0.7285)
ax.set_ylim(0.734, 0.760)
ax.set_xlabel("Cross-validated IPCW C-index (15 centre-grouped folds)")
ax.set_ylabel("Public leaderboard IPCW C-index")
ax.set_title("Cross-validation vs the unseen test centre")
fig.tight_layout()
fig.savefig(config.FIGURES / "fig24_leaderboard.png")
plt.close(fig)

# ===========================================================================
# 2. TP53 multi-hit and complex karyotype in the same patients
# ===========================================================================
k = train[train["cyto_missing"] == 0]          # patients with a usable karyotype
g = k.groupby("tp53_multihit")
co = pd.DataFrame({
    "patients": g.size(),
    "complex_karyotype_pct": (100 * g["cyto_complex"].mean()).round(1),
    "monosomal_karyotype_pct": (100 * g["cyto_monosomal"].mean()).round(1),
    "median_mutated_genes": g["mol_n_genes"].median(),
    "median_max_vaf": g["mol_max_vaf"].median().round(2),
})
co.index = ["TP53 not multi-hit", "TP53 multi-hit"]
co.to_csv(config.TABLES / "tp53_cooccurrence.csv")
share = 100 * k.loc[k["cyto_complex"] == 1, "tp53_multihit"].mean()
print("\nTP53 multi-hit vs other patients (training, usable karyotype):")
print(co.to_string())
print(f"Share of complex karyotypes that are TP53 multi-hit: {share:.1f}%")

# ===========================================================================
# 3. XGBoost vs elastic-net inside high-risk subgroups (out-of-fold)
# ===========================================================================
# Idea being tested: "trees rank the highest-risk patients poorly", which
# could hurt at the test centre, where such patients are common.
# Within each validation fold we score ONLY the subgroup's patients.
long = pd.concat([pd.read_csv(config.DATA_PROCESSED / "oof_cox.csv"),
                  pd.read_csv(config.DATA_PROCESSED / "oof_trees.csv")], ignore_index=True)
long["code"] = long["model"].str[0]
pred = {r: long[long["repeat"] == f"repeat{r}"].pivot(index="ID", columns="code", values="risk")
            .loc[train.index] for r in range(3)}
SUBGROUPS = {
    "IPSS-R very high (score > 6)": train["ipssr_score"] > 6,
    "IPSS-R high or very high (> 4.5)": train["ipssr_score"] > 4.5,
    "Blasts >= 10%": train["BM_BLAST"] >= 10,
    "Complex karyotype": train["cyto_complex"] == 1,
    "All other patients (IPSS-R <= 4.5)": train["ipssr_score"] <= 4.5,
}
rows = []
for name, mask in SUBGROUPS.items():
    mask = mask.to_numpy()
    c_d, c_g = [], []
    for r in range(3):
        for kf in range(5):
            valid = (folds[f"repeat{r}"] == kf).to_numpy()
            sub = valid & mask
            # skip folds with too few patients or deaths in the subgroup to score
            if sub.sum() < 30 or y_all[sub]["event"].sum() < 10:
                continue
            c_d.append(U.ipcw_c(y_all[~valid], y_all[sub], pred[r].loc[sub, "D"]))
            c_g.append(U.ipcw_c(y_all[~valid], y_all[sub], pred[r].loc[sub, "G"]))
    d = np.array(c_g) - np.array(c_d)
    rows.append({"subgroup": name, "patients": int(mask.sum()), "folds_scored": len(d),
                 "C_elasticnet": round(np.mean(c_d), 4), "C_xgboost": round(np.mean(c_g), 4),
                 "xgb_minus_enet": round(d.mean(), 4), "xgb_wins": f"{int((d > 0).sum())}/{len(d)}"})
subg = pd.DataFrame(rows)
subg.to_csv(config.TABLES / "subgroup_xgb_vs_enet.csv", index=False)
print("\nXGBoost vs elastic-net inside subgroups (out-of-fold IPCW C-index):")
print(subg.to_string(index=False))

print("\nDone: 1 figure, 3 tables.")
