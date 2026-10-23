"""
05_build_features.py  (Stage 3)
===============================
Builds the feature tables used by every model from Stage 4 on, checks them,
and creates the cross-validation folds.

Steps:
  1. Decide which genes to use (shared gene panel, frequent enough).
  2. Build clinical + cytogenetic + molecular features for train and test.
  3. Check the karyotype parser (examples, lesion frequencies, train vs test).
  4. Compute the IPSS-R clinical risk score (a published, no-fitting benchmark).
  5. Kaplan-Meier curves by IPSS-R cytogenetic category and by IPSS-R risk group.
  6. Single-feature IPCW C-index for every feature (exploratory).
  7. Centre-grouped cross-validation folds (3 repeats x 5 folds), and a
     demonstration of why missing values must be filled INSIDE each fold.

Outputs:
  data/processed/features_train.csv    (one row per patient, with OS_YEARS/OS_STATUS)
  data/processed/features_test.csv
  data/processed/cv_folds.csv          (fold number of each training patient)
  data/processed/feature_groups.json   (which columns belong to which family)
  figures/fig11_km_ipssr.png
  figures/fig12_risk_shift.png
  figures/fig13_feature_cindex.png
  results/tables/gene_panel.csv
  results/tables/cyto_parse_examples.csv
  results/tables/lesion_frequencies.csv
  results/tables/km_ipssr_groups.csv
  results/tables/ipssr_by_fold.csv
  results/tables/feature_cindex.csv
  results/tables/fold_summary.csv
  results/tables/fold_medians.csv

Run from the project's top folder:
    python src/05_build_features.py
"""

import json
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

import config
import features as F
from load_data import load_raw

plt = config.set_plot_style()
config.make_folders()

data = load_raw()
clin_tr, clin_te = data["clinical_train"], data["clinical_test"]
mol_tr, mol_te = data["molecular_train"], data["molecular_test"]

# Outcome: keep training patients WITH an outcome only.
target = data["target_train"].dropna(subset=["OS_YEARS", "OS_STATUS"])
clin_tr = clin_tr[clin_tr["ID"].isin(target["ID"])].reset_index(drop=True)
print(f"Training patients with outcome: {len(clin_tr)}   Test patients: {len(clin_te)}")


# ===========================================================================
# 1. Gene panel
# ===========================================================================
# Only genes found in BOTH centres are used. A gene never seen in the test
# file was probably not sequenced there, so a model must not rely on it.
genes_train = set(mol_tr["GENE"])
genes_test = set(mol_te["GENE"])
panel = sorted(genes_train & genes_test)

# A gene gets its own 0/1 column if at least 1% of training patients carry it.
patients_per_gene = mol_tr.groupby("GENE")["ID"].nunique()
min_patients = int(np.ceil(0.01 * len(clin_tr)))          # 1% of 3,173 = 32 patients
gene_list = sorted([g for g in panel if patients_per_gene.get(g, 0) >= min_patients],
                   key=lambda g: -patients_per_gene[g])

dropped_frequent = sorted([g for g in genes_train - genes_test
                           if patients_per_gene[g] >= min_patients],
                          key=lambda g: -patients_per_gene[g])
print(f"\nGenes: {len(genes_train)} in train, {len(genes_test)} in test, {len(panel)} shared.")
print(f"Gene columns (shared and in >= {min_patients} training patients): {len(gene_list)}")
print(f"Frequent training genes dropped because never seen in test: {dropped_frequent}")
pd.DataFrame({
    "gene": gene_list,
    "patients_train": [int(patients_per_gene[g]) for g in gene_list],
    "patients_test": [int(mol_te.loc[mol_te["GENE"] == g, "ID"].nunique()) for g in gene_list],
}).to_csv(config.TABLES / "gene_panel.csv", index=False)

n_train_only = mol_tr["GENE"].isin(genes_train - genes_test).sum()
print(f"Training mutations in genes not on the shared panel (not counted): {n_train_only}")


# ===========================================================================
# 2. Build the features
# ===========================================================================
def build(clinical, molecular):
    """Clinical + cytogenetic + molecular + IPSS-R, one row per patient."""
    clin = F.clinical_features(clinical)
    cyto = F.cytogenetic_features(clinical)
    mol = F.molecular_features(molecular, clinical["ID"].to_numpy(), gene_list, panel)
    table = pd.concat([clin, cyto, mol], axis=1)
    # TP53 "multi-hit": two or more TP53 mutations, a high VAF (>= 0.5, often
    # meaning both copies are hit), or a TP53 mutation plus a 17p loss.
    table["tp53_multihit"] = (
        (table["tp53_n_mut"] >= 2) | (table["tp53_max_vaf"] >= 0.5)
        | ((table["tp53_n_mut"] >= 1) & (table["cyto_17p"] == 1))
    ).astype(int)
    return table


X_train = build(clin_tr, mol_tr)
X_test = build(clin_te, mol_te)

# ---- IPSS-R score -----------------------------------------------------------
# Points per component; a MISSING component gets the median points of the
# TRAINING patients (a fixed number, the same for the test set). No outcome
# is used, so this does not leak survival information.
pts_tr = F.ipssr_points(clin_tr, X_train["cyto_ipssr_cat"])
pts_te = F.ipssr_points(clin_te, X_test["cyto_ipssr_cat"])
fill_points = pts_tr.median()
print("\nIPSS-R: points given to a missing component (training medians):",
      fill_points.to_dict())
X_train["ipssr_score"] = pts_tr.fillna(fill_points).sum(axis=1).to_numpy()
X_test["ipssr_score"] = pts_te.fillna(fill_points).sum(axis=1).to_numpy()

# Attach the outcome to the training table.
y = target.set_index("ID").loc[X_train.index]
X_train["OS_YEARS"] = y["OS_YEARS"].to_numpy()
X_train["OS_STATUS"] = y["OS_STATUS"].astype(int).to_numpy()
X_train.insert(0, "CENTER", clin_tr["CENTER"].to_numpy())
X_test.insert(0, "CENTER", clin_te["CENTER"].to_numpy())

# Feature families (saved so later scripts pick columns by name).
groups = {
    "clinical": ["BM_BLAST", "HB", "log_WBC", "log_ANC", "log_MONOCYTES", "log_PLT"],
    "clinical_missing": [c for c in X_train.columns if c.startswith("missing_")],
    "cytogenetic": [c for c in F.CYTO_COLUMNS] + ["cyto_missing"],
    "molecular_summary": ["mol_n_mut", "mol_n_genes", "mol_max_vaf", "mol_n_truncating",
                          "tp53_n_mut", "tp53_max_vaf", "tp53_multihit",
                          "mol_FLT3_ITD", "mol_MLL_PTD"] + list(F.PATHWAYS),
    "genes": ["mut_" + g for g in gene_list],
    "scores": ["ipssr_score"],
}
with open(config.DATA_PROCESSED / "feature_groups.json", "w", encoding="utf-8") as f:
    json.dump(groups, f, indent=2)

X_train.to_csv(config.DATA_PROCESSED / "features_train.csv")
X_test.to_csv(config.DATA_PROCESSED / "features_test.csv")
n_features = sum(len(v) for v in groups.values())
print(f"\nSaved features: train {X_train.shape}, test {X_test.shape}; {n_features} feature columns.")


# ===========================================================================
# 3. Check the karyotype parser
# ===========================================================================
examples_idx = [0, 2, 5, 11, 27, 40, 77, 120, 150, 260]
examples = pd.DataFrame({
    "karyotype": clin_tr["CYTOGENETICS"].iloc[examples_idx].to_numpy(),
    "n_abnormalities": X_train["cyto_n_abn"].iloc[examples_idx].astype("Int64").to_numpy(),
    "ipssr_cyto_category": X_train["cyto_ipssr_cat"].iloc[examples_idx].astype("Int64").to_numpy(),
})
examples.to_csv(config.TABLES / "cyto_parse_examples.csv", index=False)
print("\nParser examples:")
print(examples.to_string(index=False))

lesions = ["cyto_missing", "cyto_normal", "cyto_complex", "cyto_monosomal", "cyto_minus7",
           "cyto_del7q", "cyto_del5q", "cyto_minus5", "cyto_plus8", "cyto_del20q",
           "cyto_minusY", "cyto_17p", "cyto_3q", "tp53_multihit"]
freq = pd.DataFrame({
    "train_pct": (100 * X_train[lesions].mean()).round(1),
    "test_pct": (100 * X_test[lesions].mean()).round(1),
})
freq.to_csv(config.TABLES / "lesion_frequencies.csv")
# .mean() skips NaN, so lesion rates are % of patients WITH a usable karyotype
# (cyto_missing itself is % of all patients). TP53 multi-hit is % of all patients.
print("\nLesion frequencies (% of patients with a usable karyotype; cyto_missing and tp53 = % of all):")
print(freq.to_string())


# ===========================================================================
# 4-5. IPSS-R: Kaplan-Meier curves, and train vs test risk
# ===========================================================================
cat_names = {0: "Very good", 1: "Good", 2: "Intermediate", 3: "Poor", 4: "Very poor"}
# An ordered scale gets an ordered colour ramp (light blue -> dark blue).
ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

X_train["ipssr_group"] = F.ipssr_risk_group(X_train["ipssr_score"])
X_test["ipssr_group"] = F.ipssr_risk_group(X_test["ipssr_score"])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
panels = [
    (axes[0], "cyto_ipssr_cat", [(k, v) for k, v in cat_names.items()], "IPSS-R cytogenetic category"),
    (axes[1], "ipssr_group", [(g, g) for g in F.IPSSR_GROUPS], "IPSS-R risk group (full score)"),
]
km_rows = []
for ax, col, levels, title in panels:
    sub = X_train.dropna(subset=[col])
    res = multivariate_logrank_test(sub["OS_YEARS"], sub[col].astype(str), sub["OS_STATUS"])
    for (value, label), colour in zip(levels, ramp):
        g = sub[sub[col] == value]
        k = KaplanMeierFitter().fit(g["OS_YEARS"], g["OS_STATUS"])
        s = k.survival_function_
        ax.step(s.index, s.iloc[:, 0], where="post", color=colour, linewidth=2,
                label=f"{label} (n={len(g)})")
        km_rows.append({"grouping": title, "group": label, "patients": len(g),
                        "median_years": round(k.median_survival_time_, 2),
                        "S_at_3_years": round(float(k.survival_function_at_times(3).iloc[0]), 3),
                        "logrank_chi2": round(res.test_statistic, 1)})
    ax.set_title(f"{title}\nlog-rank chi2 = {res.test_statistic:.1f} (4 df), p < 0.001", fontsize=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Years since diagnosis")
    ax.set_ylabel("Probability of being alive")
    ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig11_km_ipssr.png")
plt.close(fig)
km_table = pd.DataFrame(km_rows)
km_table.to_csv(config.TABLES / "km_ipssr_groups.csv", index=False)
print("\nKaplan-Meier by IPSS-R groups:")
print(km_table.to_string(index=False))

# Share of patients in each IPSS-R risk group, train vs test.
share = pd.DataFrame({
    "train_pct": (100 * X_train["ipssr_group"].value_counts(normalize=True)).reindex(F.IPSSR_GROUPS),
    "test_pct": (100 * X_test["ipssr_group"].value_counts(normalize=True)).reindex(F.IPSSR_GROUPS),
}).round(1)
print("\nIPSS-R risk groups (% of patients):")
print(share.to_string())
print(f"Median IPSS-R score: train {X_train['ipssr_score'].median():.1f}, "
      f"test {X_test['ipssr_score'].median():.1f}")

fig, ax = plt.subplots(figsize=(7.2, 3.8))
pos = np.arange(len(F.IPSSR_GROUPS))
w = 0.38
ax.bar(pos - w / 2, share["train_pct"], width=w, color=config.COLOR_TRAIN, label="Training (23 centres)")
ax.bar(pos + w / 2, share["test_pct"], width=w, color=config.COLOR_TEST, label="Test (centre KYW)")
for i, (a, b) in enumerate(zip(share["train_pct"], share["test_pct"])):
    ax.text(i - w / 2, a + 0.6, f"{a:.0f}%", ha="center", fontsize=8, color=config.TEXT_SOFT)
    ax.text(i + w / 2, b + 0.6, f"{b:.0f}%", ha="center", fontsize=8, color=config.TEXT_SOFT)
ax.set_xticks(pos)
ax.set_xticklabels(F.IPSSR_GROUPS)
ax.set_ylabel("Patients (%)")
ax.set_title("IPSS-R risk groups: the test centre is higher risk")
ax.set_ylim(0, 48)                       # head-room so the legend clears the bars
ax.legend(loc="upper center", ncol=2)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig12_risk_shift.png")
plt.close(fig)


# ===========================================================================
# 6. Single-feature IPCW C-index (exploratory)
# ===========================================================================
# Each feature is used directly as a risk score (higher value = higher risk).
# C > 0.5: the feature goes with SHORTER survival (risk factor).
# C < 0.5: the feature goes with LONGER survival (protective).
# Missing values get the median here ONLY for this quick look; real models
# fill missing values inside each cross-validation fold (step 7).
y_surv = Surv.from_arrays(event=X_train["OS_STATUS"].astype(bool), time=X_train["OS_YEARS"])
feature_cols = [c for fam in groups.values() for c in fam]
rows = []
for col in feature_cols:
    x = X_train[col].astype(float)
    if x.nunique(dropna=True) < 2:
        continue
    x = x.fillna(x.median())
    c = concordance_index_ipcw(y_surv, y_surv, x.to_numpy(), tau=config.TAU)[0]
    rows.append({"feature": col, "ipcw_C": round(c, 4), "distance_from_0.5": round(abs(c - 0.5), 4),
                 "direction": "risk" if c > 0.5 else "protective"})
feat_c = pd.DataFrame(rows).sort_values("distance_from_0.5", ascending=False)
feat_c.to_csv(config.TABLES / "feature_cindex.csv", index=False)
print("\nTop 15 single features by IPCW C-index (distance from 0.5):")
print(feat_c.head(15).to_string(index=False))

top = feat_c.head(25).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.2, 6.6))
colours = [config.COLOR_TEST if d == "risk" else config.COLOR_TRAIN for d in top["direction"]]
ax.barh(top["feature"], top["ipcw_C"] - 0.5, left=0.5, color=colours, height=0.7)
for k, v in enumerate(top["ipcw_C"]):
    ax.text(v + (0.003 if v > 0.5 else -0.003), k, f"{v:.3f}", va="center",
            ha="left" if v > 0.5 else "right", fontsize=8, color=config.TEXT_SOFT)
ax.axvline(0.5, color=config.TEXT_MAIN, linewidth=1.2)
ax.set_xlim(0.33, 0.72)                  # room on the left for the protective labels
ax.set_xlabel("IPCW C-index of the feature on its own (0.5 = no information)")
ax.set_title("25 most informative single features")
# Legend built from coloured patches (one per meaning).
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=config.COLOR_TEST, label="higher value = shorter survival"),
                   Patch(color=config.COLOR_TRAIN, label="higher value = longer survival")],
          loc="lower right", fontsize=8.5)
ax.tick_params(axis="y", labelsize=8.5)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig13_feature_cindex.png")
plt.close(fig)


# ===========================================================================
# 7. Cross-validation folds grouped by centre
# ===========================================================================
# Rule: all patients of one centre go to the SAME fold, so every validation
# fold contains only centres the model has not seen during training.
#
# How the centres are dealt into 5 folds (like dealing cards):
#   1. put the 23 centres in a RANDOM order,
#   2. take them one by one and give each to the fold that currently has
#      the FEWEST patients (this keeps the folds roughly the same size).
# A different random order per repeat gives a genuinely different split.
#
# (Note, fixed in Stage 4: the first version used sklearn's
#  StratifiedGroupKFold(shuffle=True). With only 23 centres it sorts the
#  centres before shuffling has any effect, so all 3 "repeats" came out
#  IDENTICAL. The check at the end of this block now guards against that.)
N_SPLITS, N_REPEATS = 5, 3


def deal_centres_into_folds(centre_sizes, n_splits, rng):
    """Return {centre: fold number}, dealing centres in a random order."""
    order = rng.permutation(centre_sizes.index.to_numpy())   # random order of centres
    fold_size = np.zeros(n_splits)                           # patients per fold so far
    assignment = {}
    for centre in order:
        k = int(np.argmin(fold_size))                        # the smallest fold so far
        assignment[centre] = k
        fold_size[k] += centre_sizes[centre]
    return assignment


centre_sizes = X_train["CENTER"].value_counts()
folds = pd.DataFrame({"ID": X_train.index, "CENTER": X_train["CENTER"].to_numpy()})
rng = np.random.default_rng(config.SEED)
for r in range(N_REPEATS):
    assignment = deal_centres_into_folds(centre_sizes, N_SPLITS, rng)
    folds[f"repeat{r}"] = folds["CENTER"].map(assignment)
folds.to_csv(config.DATA_PROCESSED / "cv_folds.csv", index=False)

# Check 1: no centre appears in two folds of the same repeat.
for r in range(N_REPEATS):
    assert folds.groupby("CENTER")[f"repeat{r}"].nunique().max() == 1, "a centre is split!"


# Check 2: the repeats really are different splits. A split is described by
# which centres end up together; two repeats are identical if they produce
# the same set of centre groups.
def partition(r):
    return sorted(sorted(c) for c in folds.groupby(f"repeat{r}")["CENTER"].unique())


for r in range(1, N_REPEATS):
    assert partition(r) != partition(0), f"repeat {r} is identical to repeat 0!"
print("\nChecks passed: every centre sits in exactly one fold, and the 3 repeats are different splits.")

summary_rows = []
for r in range(N_REPEATS):
    for k in range(N_SPLITS):
        in_fold = folds[f"repeat{r}"] == k
        sub = X_train[in_fold.to_numpy()]
        summary_rows.append({
            "repeat": r, "fold": k, "patients": len(sub), "centres": sub["CENTER"].nunique(),
            "death_rate": round(sub["OS_STATUS"].mean(), 2),
            "centre_list": " ".join(sorted(sub["CENTER"].unique())),
        })
fold_summary = pd.DataFrame(summary_rows)
fold_summary.to_csv(config.TABLES / "fold_summary.csv", index=False)
print("Folds of all 3 repeats:")
print(fold_summary.to_string(index=False))

# ---- IPSS-R score evaluated fold by fold ------------------------------------
# The IPSS-R score is NOT fitted, so this shows how much the IPCW C-index
# moves between centres purely by chance and case mix. The censoring curve
# G(t) is estimated on the other 4 folds, exactly as Stage 4 will do.
cv_rows = []
for r in range(N_REPEATS):
    for k in range(N_SPLITS):
        valid = (folds[f"repeat{r}"] == k).to_numpy()
        c = concordance_index_ipcw(y_surv[~valid], y_surv[valid],
                                   X_train.loc[valid, "ipssr_score"].to_numpy(), tau=config.TAU)[0]
        cv_rows.append({"repeat": r, "fold": k, "ipcw_C": round(c, 4)})
ipssr_cv = pd.DataFrame(cv_rows)
ipssr_cv.to_csv(config.TABLES / "ipssr_by_fold.csv", index=False)
full_c = concordance_index_ipcw(y_surv, y_surv, X_train["ipssr_score"].to_numpy(), tau=config.TAU)[0]
print(f"\nIPSS-R score IPCW C-index, all training patients: {full_c:.4f}")
print(f"IPSS-R score across the 15 validation folds: mean {ipssr_cv['ipcw_C'].mean():.4f}, "
      f"sd {ipssr_cv['ipcw_C'].std():.4f}, min {ipssr_cv['ipcw_C'].min():.4f}, "
      f"max {ipssr_cv['ipcw_C'].max():.4f}")

# ---- Why fill missing values inside each fold? ------------------------------
# The median used to fill a missing value must be computed from the TRAINING
# part of each fold only. If we used all patients, the validation patients
# would help choose the fill value: a small leak of information.
med_rows = []
for k in range(N_SPLITS):
    valid = (folds["repeat0"] == k).to_numpy()
    med_rows.append({"fold": k,
                     **{c: round(X_train.loc[~valid, c].median(), 3)
                        for c in ["BM_BLAST", "HB", "log_PLT", "log_MONOCYTES"]}})
med_rows.append({"fold": "all patients",
                 **{c: round(X_train[c].median(), 3) for c in ["BM_BLAST", "HB", "log_PLT", "log_MONOCYTES"]}})
fold_medians = pd.DataFrame(med_rows)
fold_medians.to_csv(config.TABLES / "fold_medians.csv", index=False)
print("\nMedians used to fill missing values, computed on each fold's training part:")
print(fold_medians.to_string(index=False))

print("\nSaved 3 figures, 8 tables and 4 files in data/processed/.")
