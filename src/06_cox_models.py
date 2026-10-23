"""
06_cox_models.py  (Stage 4, part 1)
===================================
Compares Cox models with honest centre-grouped cross-validation, then makes
the submission files.

Models compared on the SAME 15 validation folds:
  A. IPSS-R score            - published clinical score, no fitting (reference)
  B. Benchmark-style Cox     - BM_BLAST, HB, PLT and number of mutations
                               (my reconstruction of the challenge benchmark)
  C. IPSS-M-style Cox        - 5 blood values, IPSS-R cytogenetic category,
                               16 IPSS-M genes and the residual-gene count
  D. Elastic-net Cox         - all 84 features (no monocytes, no missing flags);
                               penalty chosen by nested centre-grouped CV
  E. Elastic-net Cox + extra - D plus log monocytes and the 6 missing flags
                               (tests the Stage 1 worry about MONOCYTES)

For each model we report the mean IPCW C-index, and - the fair comparison -
the PAIRED difference with the IPSS-R score on each fold.

Outputs:
  figures/fig14_cox_cv.png
  results/tables/cox_cv_folds.csv        (every model, every fold)
  results/tables/cox_cv_summary.csv
  results/tables/cox_cv_pairwise.csv
  results/tables/coxnet_selected.json    (settings of the final elastic-net)
  data/processed/oof_repeat0.csv         (out-of-fold risk scores, used in 07)
  data/processed/oof_cox.csv             (all 3 repeats, used in Stage 6)
  submissions/stage4_benchmark_cox.csv
  submissions/stage4_ipssm_cox.csv
  submissions/stage4_elasticnet_cox.csv

Run from the project's top folder (takes about 2-4 minutes):
    python src/06_cox_models.py
"""

import json
import warnings
import numpy as np
import pandas as pd

import config
import cv_utils as U
from load_data import load_raw

warnings.filterwarnings("ignore")          # hide convergence chatter from the solver
plt = config.set_plot_style()
config.make_folders()

# ===========================================================================
# 0. Data
# ===========================================================================
train, test, groups, folds = U.load_stage3()
train, test = U.add_ipssm_residual(train), U.add_ipssm_residual(test)

# Benchmark-style inputs: raw platelets and the number of mutations counted
# over ALL genes in the raw file (as a simple benchmark would do).
raw = load_raw()
for table, mol in [(train, raw["molecular_train"]), (test, raw["molecular_test"])]:
    table["PLT"] = np.expm1(table["log_PLT"])                   # undo log(1 + x)
    table["n_mut_all_genes"] = mol.groupby("ID").size().reindex(table.index, fill_value=0)

# ===========================================================================
# 1. Feature sets
# ===========================================================================
CLIN5 = ["BM_BLAST", "HB", "log_WBC", "log_ANC", "log_PLT"]
SETS = {
    "benchmark": ["BM_BLAST", "HB", "PLT", "n_mut_all_genes"],
    "ipssm": CLIN5 + ["cyto_ipssr_cat", "cyto_missing"] + U.IPSSM_MAIN + ["ipssm_residual"],
    "full": [c for fam in ["clinical", "cytogenetic", "molecular_summary", "genes", "scores"]
             for c in groups[fam] if c != "log_MONOCYTES"],
}
SETS["full_plus"] = SETS["full"] + ["log_MONOCYTES"] + groups["clinical_missing"]
for name, cols in SETS.items():
    print(f"Feature set '{name}': {len(cols)} columns")

# ===========================================================================
# 2. The models as fit_predict functions
# ===========================================================================
# Each function is trained on train_part only and returns risk scores for
# valid_part. Imputation and scaling live INSIDE the pipeline, so they are
# also learned on train_part only.

def ipssr(train_part, valid_part):
    return valid_part["ipssr_score"].to_numpy()


def cox_on(cols):
    def fit_predict(train_part, valid_part):
        model = U.make_cox().fit(train_part[cols], U.survival_target(train_part))
        return model.predict(valid_part[cols])          # predict = linear predictor = log relative risk
    return fit_predict


def coxnet_on(cols):
    def fit_predict(train_part, valid_part):
        model, alpha, _, _ = U.fit_coxnet_nested(
            train_part[cols], U.survival_target(train_part), train_part["CENTER"].to_numpy())
        return model.predict(valid_part[cols], alpha=alpha)
    return fit_predict


MODELS = {
    "A. IPSS-R score (no fitting)": ipssr,
    "B. Benchmark-style Cox (4)": cox_on(SETS["benchmark"]),
    "C. IPSS-M-style Cox (24)": cox_on(SETS["ipssm"]),
    "D. Elastic-net Cox (84)": coxnet_on(SETS["full"]),
    "E. Elastic-net Cox + monocytes/flags (91)": coxnet_on(SETS["full_plus"]),
}

# ===========================================================================
# 3. Cross-validation on the 15 centre-grouped folds
# ===========================================================================
all_folds, oof, oof_all_repeats = [], {}, {}
for name, fp in MODELS.items():
    print(f"\nEvaluating {name} ...")
    res, preds = U.cross_validate(fp, train, folds, verbose_name=name)
    res.insert(0, "model", name)
    all_folds.append(res)
    oof[name] = preds["repeat0"]
    oof_all_repeats[name] = preds
    print(f"  mean IPCW C = {res['ipcw_C'].mean():.4f}  (sd {res['ipcw_C'].std():.4f})")
cv = pd.concat(all_folds, ignore_index=True)
cv.to_csv(config.TABLES / "cox_cv_folds.csv", index=False)
# All 3 repeats of out-of-fold predictions, for the Stage 6 ensembles.
U.save_oof(oof_all_repeats, config.DATA_PROCESSED / "oof_cox.csv")
oof_table = pd.DataFrame(oof)
oof_table["OS_YEARS"] = train["OS_YEARS"]
oof_table["OS_STATUS"] = train["OS_STATUS"]
oof_table.to_csv(config.DATA_PROCESSED / "oof_repeat0.csv")

# ---- Paired comparison with the IPSS-R score -------------------------------
# For each fold: difference = model C - IPSS-R C on that same fold.
# Fold difficulty cancels out; what is left is the model's own advantage.
ref = cv[cv["model"].str.startswith("A.")].set_index(["repeat", "fold"])["ipcw_C"]
summary_rows = []
for name in MODELS:
    m = cv[cv["model"] == name].set_index(["repeat", "fold"])["ipcw_C"]
    diff = m - ref
    summary_rows.append({
        "model": name,
        "mean_ipcw_C": round(m.mean(), 4), "sd": round(m.std(), 4),
        "min": round(m.min(), 4), "max": round(m.max(), 4),
        "mean_harrell_C": round(cv.loc[cv["model"] == name, "harrell_C"].mean(), 4),
        "paired_diff_vs_IPSSR": round(diff.mean(), 4),
        "sd_of_paired_diff": round(diff.std(), 4),
        "folds_won_vs_IPSSR": f"{int((diff > 0).sum())}/15" if not name.startswith("A.") else "-",
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(config.TABLES / "cox_cv_summary.csv", index=False)
print("\nCross-validation summary (15 centre-grouped folds):")
print(summary.to_string(index=False))

# ---- Other paired comparisons that answer specific questions --------------
#   C vs B : does a clinically chosen feature set beat the simple benchmark?
#   D vs C : do 84 penalised features beat 24 clinically chosen ones?
#   E vs D : do monocytes and the missing flags help (Stage 1 question)?
def fold_scores(prefix):
    return cv[cv["model"].str.startswith(prefix)].set_index(["repeat", "fold"])["ipcw_C"]


pair_rows = []
for a, b, question in [("C", "B", "clinical feature set vs benchmark"),
                       ("D", "C", "84 penalised features vs 24 chosen"),
                       ("E", "D", "adding monocytes + missing flags")]:
    d = fold_scores(a + ".") - fold_scores(b + ".")
    pair_rows.append({"comparison": f"{a} - {b}", "question": question,
                      "mean_diff": round(d.mean(), 4), "sd_diff": round(d.std(), 4),
                      "folds_won": f"{int((d > 0).sum())}/15"})
pairs = pd.DataFrame(pair_rows)
pairs.to_csv(config.TABLES / "cox_cv_pairwise.csv", index=False)
print("\nOther paired comparisons:")
print(pairs.to_string(index=False))

# ---- Figure 14 -------------------------------------------------------------
names = list(MODELS)
short = ["A. IPSS-R", "B. Benchmark Cox", "C. IPSS-M-style Cox", "D. Elastic-net", "E. Elastic-net +"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
rng = np.random.default_rng(config.SEED)
ax = axes[0]
for i, name in enumerate(names):
    vals = cv.loc[cv["model"] == name, "ipcw_C"].to_numpy()
    jitter = rng.uniform(-0.12, 0.12, len(vals))          # spread the dots sideways a little
    ax.scatter(np.full(len(vals), i) + jitter, vals, s=22, color=config.COLOR_TRAIN, alpha=0.55,
               edgecolor="white", linewidth=0.6)
    ax.plot([i - 0.28, i + 0.28], [vals.mean()] * 2, color=config.TEXT_MAIN, linewidth=2.2)
    ax.text(i + 0.31, vals.mean(), f"{vals.mean():.3f}", va="center", fontsize=8, color=config.TEXT_MAIN)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(short, rotation=25, ha="right", fontsize=8.5)
ax.set_ylabel("IPCW C-index on validation fold")
ax.set_title("Each dot = one of 15 folds; bar = mean")
ax.set_xlim(-0.5, len(names) - 0.2)
ax.grid(axis="x", visible=False)

ax = axes[1]
for i, name in enumerate(names[1:], start=1):
    m = cv[cv["model"] == name].set_index(["repeat", "fold"])["ipcw_C"]
    d = (m - ref).to_numpy()
    jitter = rng.uniform(-0.12, 0.12, len(d))
    colours = [config.COLOR_TRAIN if v > 0 else config.COLOR_TEST for v in d]
    ax.scatter(np.full(len(d), i) + jitter, d, s=22, c=colours, alpha=0.75, edgecolor="white", linewidth=0.6)
    ax.plot([i - 0.28, i + 0.28], [d.mean()] * 2, color=config.TEXT_MAIN, linewidth=2.2)
    ax.text(i + 0.31, d.mean(), f"{d.mean():+.3f}\n{(d > 0).sum()}/15 won", va="center",
            fontsize=7.5, color=config.TEXT_MAIN)
ax.axhline(0, color=config.TEXT_MAIN, linewidth=1)
ax.set_xticks(range(1, len(names)))
ax.set_xticklabels(short[1:], rotation=25, ha="right", fontsize=8.5)
ax.set_xlim(0.5, len(names) - 0.15)
ax.set_ylabel("Model C minus IPSS-R C, same fold")
ax.set_title("Paired difference with IPSS-R\n(blue = model better, orange = worse)")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig14_cox_cv.png")
plt.close(fig)

# ===========================================================================
# 4. Final models on ALL training patients -> submission files
# ===========================================================================
y_all = U.survival_target(train)
submissions = {}

bench = U.make_cox().fit(train[SETS["benchmark"]], y_all)
submissions["stage4_benchmark_cox.csv"] = bench.predict(test[SETS["benchmark"]])

ipssm = U.make_cox().fit(train[SETS["ipssm"]], y_all)
submissions["stage4_ipssm_cox.csv"] = ipssm.predict(test[SETS["ipssm"]])

# Elastic-net: the same nested procedure, now with all 3,173 patients. The inner
# split again holds out whole centres.
enet, alpha, l1, inner = U.fit_coxnet_nested(train[SETS["full"]], y_all, train["CENTER"].to_numpy(), n_inner=5)
submissions["stage4_elasticnet_cox.csv"] = enet.predict(test[SETS["full"]], alpha=alpha)
coef = pd.Series(enet[-1].coef_[:, list(enet[-1].alphas_).index(alpha)], index=SETS["full"])
with open(config.TABLES / "coxnet_selected.json", "w", encoding="utf-8") as f:
    json.dump({"l1_ratio": l1, "alpha": float(alpha), "inner_cv_ipcw_C": round(float(inner), 4),
               "n_nonzero_coefficients": int((coef != 0).sum()),
               "coefficients": {k: round(float(v), 5) for k, v in coef[coef != 0].sort_values().items()}},
              f, indent=2)
print(f"\nFinal elastic-net: l1_ratio = {l1}, alpha = {alpha:.4f}, inner CV C = {inner:.4f}, "
      f"{int((coef != 0).sum())} of {len(coef)} coefficients non-zero")

# Save each submission: column "ID" as index, column "risk_score".
for file_name, risk in submissions.items():
    sub = pd.DataFrame({"risk_score": np.asarray(risk, dtype=float)}, index=test.index)
    sub.index.name = "ID"
    assert len(sub) == 1193 and sub["risk_score"].notna().all(), "submission problem!"
    sub.to_csv(config.SUBMISSIONS / file_name)
    print(f"Saved submissions/{file_name}: {len(sub)} rows, "
          f"risk from {sub['risk_score'].min():.3f} to {sub['risk_score'].max():.3f}")

# Sanity check: the test centre should look higher risk than training on average.
train_risk = enet.predict(train[SETS["full"]], alpha=alpha)
test_risk = submissions["stage4_elasticnet_cox.csv"]
print(f"\nElastic-net mean risk score: train {np.mean(train_risk):+.3f}, test {np.mean(test_risk):+.3f} "
      "(test should be higher)")
print("\nDone: 1 figure, 4 tables, 3 submission files.")
