"""
02_data_audit.py  (Stage 1)
===========================
A careful first look at the data BEFORE any modelling.

Questions this script answers:
  A. Do the files link together correctly through the ID column?
  B. What does the outcome look like (deaths, censoring, follow-up time)?
  C. How much clinical data is missing, and do training and test look alike?
  D. Which clinical centres are in the training and test sets?
  E. What do the mutation (molecular) data look like?
  F. How is the CYTOGENETICS text written?

Outputs:
  figures/fig01_missing_values.png
  figures/fig02_clinical_distributions.png
  figures/fig03_centers.png
  figures/fig04_followup.png
  figures/fig05_gene_frequencies.png
  results/tables/clinical_summary.csv
  results/tables/center_summary.csv
  results/tables/gene_frequencies.csv
  results/tables/audit_summary.txt

Run from the project's top folder:
    python src/02_data_audit.py
"""

import numpy as np                # numbers and arrays
import pandas as pd               # tables
import matplotlib                 # plotting
matplotlib.use("Agg")             # draw figures to files, no pop-up windows
import matplotlib.pyplot as plt

import config
from load_data import load_raw

config.make_folders()

# ---------------------------------------------------------------------------
# Plot style used by every figure (clean, light grid, readable text)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.edgecolor": config.GRID,
    "axes.labelcolor": config.TEXT_SOFT,
    "axes.titlecolor": config.TEXT_MAIN,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": config.GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,          # grid lines behind the bars
    "xtick.color": config.TEXT_SOFT,
    "ytick.color": config.TEXT_SOFT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

# We collect the key facts in this list and save them to a text file at the end.
summary_lines = []


def report(text=""):
    """Print a line AND remember it for audit_summary.txt."""
    print(text)
    summary_lines.append(text)


# ===========================================================================
# 0. Load the five files
# ===========================================================================
data = load_raw()
clin_tr = data["clinical_train"]
clin_te = data["clinical_test"]
mol_tr = data["molecular_train"]
mol_te = data["molecular_test"]
target = data["target_train"]

report("=" * 64)
report("STAGE 1 - DATA AUDIT")
report("=" * 64)

# ===========================================================================
# A. Do the files link together through ID?
# ===========================================================================
report("\nA. LINKING THE FILES")
report(f"  Training patients (clinical): {len(clin_tr)}")
report(f"  Test patients (clinical):     {len(clin_te)}")

# .duplicated() marks repeated IDs; .sum() counts the True values.
report(f"  Repeated IDs in clinical_train: {clin_tr['ID'].duplicated().sum()}")

# Python sets let us compare groups of IDs quickly.
ids_train = set(clin_tr["ID"])
ids_test = set(clin_te["ID"])
ids_target = set(target["ID"])
report(f"  Same IDs in clinical_train and target_train? {ids_train == ids_target}")
report(f"  IDs shared by train and test (should be 0): {len(ids_train & ids_test)}")

# A mutation row must belong to a known patient.
report(f"  Mutation rows with an unknown training ID: {(~mol_tr['ID'].isin(ids_train)).sum()}")
report(f"  Mutation rows with an unknown test ID:     {(~mol_te['ID'].isin(ids_test)).sum()}")

# ===========================================================================
# B. The outcome: OS_YEARS and OS_STATUS
# ===========================================================================
report("\nB. THE OUTCOME (target_train.csv)")

# Some patients have no outcome at all. They cannot be used for training.
no_outcome = target["OS_YEARS"].isna() | target["OS_STATUS"].isna()
report(f"  Patients with NO outcome recorded: {no_outcome.sum()}  -> removed from modelling")

y = target[~no_outcome].copy()   # keep only patients with an outcome
n = len(y)
deaths = int((y["OS_STATUS"] == 1).sum())
censored = int((y["OS_STATUS"] == 0).sum())
report(f"  Patients with an outcome: {n}")
report(f"  Deaths (OS_STATUS = 1):    {deaths}  ({100 * deaths / n:.1f}%)")
report(f"  Censored (OS_STATUS = 0):  {censored}  ({100 * censored / n:.1f}%)")
report(f"  OS_YEARS: min {y['OS_YEARS'].min():.2f}, median {y['OS_YEARS'].median():.2f}, "
       f"max {y['OS_YEARS'].max():.2f} years")
report(f"  Patients with OS_YEARS exactly 0: {(y['OS_YEARS'] == 0).sum()}")
report(f"  Patients followed beyond the 7-year horizon: {(y['OS_YEARS'] > config.TAU).sum()}")

# Which centres do the patients without an outcome come from?
# .merge joins two tables on the ID column (like VLOOKUP in Excel).
missing_by_center = (target[no_outcome]
                     .merge(clin_tr[["ID", "CENTER"]], on="ID")["CENTER"]
                     .value_counts())
report("  Patients without outcome, by centre: " + ", ".join(
    f"{c}={k}" for c, k in missing_by_center.items()))

# ---- Figure 4: follow-up times, split by death / censored -----------------
fig, ax = plt.subplots(figsize=(7.2, 3.8))
bins = np.arange(0, 22.5, 0.5)    # half-year bins from 0 to 22 years
ax.hist(
    [y.loc[y["OS_STATUS"] == 1, "OS_YEARS"], y.loc[y["OS_STATUS"] == 0, "OS_YEARS"]],
    bins=bins, stacked=True,
    color=[config.COLOR_DEATH, config.COLOR_CENSOR],
    edgecolor="white", linewidth=1.0,   # thin white gaps between bars
    label=[f"Died (n={deaths})", f"Censored, alive at last follow-up (n={censored})"],
)
ax.axvline(config.TAU, color=config.TEXT_MAIN, linestyle="--", linewidth=1.2)
ax.text(config.TAU + 0.2, ax.get_ylim()[1] * 0.92, "7-year metric horizon",
        color=config.TEXT_MAIN, fontsize=9)
ax.set_xlabel("Observed time OS_YEARS (years)")
ax.set_ylabel("Number of patients")
ax.set_title("Follow-up time of training patients")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.85))
fig.tight_layout()
fig.savefig(config.FIGURES / "fig04_followup.png")
plt.close(fig)

# ===========================================================================
# C. Clinical variables: missing values and train-vs-test comparison
# ===========================================================================
report("\nC. CLINICAL VARIABLES (training vs test)")
numeric_cols = ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"]
all_cols = numeric_cols + ["CYTOGENETICS"]

rows = []
for col in all_cols:
    row = {
        "variable": col,
        # .isna().mean() = fraction of missing values; x100 gives a percentage.
        "missing_train_pct": round(100 * clin_tr[col].isna().mean(), 1),
        "missing_test_pct": round(100 * clin_te[col].isna().mean(), 1),
    }
    if col in numeric_cols:
        row["median_train"] = round(clin_tr[col].median(), 2)
        row["median_test"] = round(clin_te[col].median(), 2)
    rows.append(row)
clinical_summary = pd.DataFrame(rows)
clinical_summary.to_csv(config.TABLES / "clinical_summary.csv", index=False)
report(clinical_summary.to_string(index=False))

high_blast_tr = 100 * (clin_tr["BM_BLAST"] > 20).mean()
high_blast_te = 100 * (clin_te["BM_BLAST"] > 20).mean()
report(f"  Patients with bone-marrow blasts > 20%: train {high_blast_tr:.1f}%, test {high_blast_te:.1f}%")

# ---- Figure 1: percentage missing, train vs test --------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
positions = np.arange(len(all_cols))   # 0, 1, 2, ... one slot per variable
width = 0.38                           # width of each bar
ax.barh(positions - width / 2, clinical_summary["missing_train_pct"], height=width,
        color=config.COLOR_TRAIN, label="Training (23 centres)")
ax.barh(positions + width / 2, clinical_summary["missing_test_pct"], height=width,
        color=config.COLOR_TEST, label="Test (centre KYW)")
# Write the percentage at the end of each bar.
for i, (a, b) in enumerate(zip(clinical_summary["missing_train_pct"],
                               clinical_summary["missing_test_pct"])):
    ax.text(a + 0.8, i - width / 2, f"{a:.1f}%", va="center", fontsize=8, color=config.TEXT_SOFT)
    ax.text(b + 0.8, i + width / 2, f"{b:.1f}%", va="center", fontsize=8, color=config.TEXT_SOFT)
ax.set_yticks(positions)
ax.set_yticklabels(all_cols)
ax.invert_yaxis()                      # first variable at the top
ax.set_xlabel("Missing values (% of patients)")
ax.set_xlim(0, 85)
ax.set_title("Missing clinical data: training vs test")
ax.legend(loc="lower right")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig01_missing_values.png")
plt.close(fig)

# ---- Figure 2: distributions of the 6 numeric variables -------------------
# Blood counts are very skewed (a few huge values), so we plot them on a
# log10 scale. log10(10) = 1, log10(100) = 2, so equal steps = x10 changes.
log_cols = {"WBC", "ANC", "MONOCYTES", "PLT"}
fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6))
for ax, col in zip(axes.ravel(), numeric_cols):
    tr = clin_tr[col].dropna()
    te = clin_te[col].dropna()
    if col in log_cols:
        # Adding 0.01 avoids log10(0), which is undefined.
        tr, te = np.log10(tr + 0.01), np.log10(te + 0.01)
        label = f"log10({col})"
    else:
        label = col
    # Shared bins so both histograms use the same intervals.
    edges = np.linspace(min(tr.min(), te.min()), max(tr.max(), te.max()), 36)
    # density=True rescales each histogram to area 1, so training (3,323
    # patients) and test (1,193 patients) can be compared fairly.
    ax.hist(tr, bins=edges, density=True, histtype="step", linewidth=2,
            color=config.COLOR_TRAIN, label="Training")
    ax.hist(te, bins=edges, density=True, histtype="step", linewidth=2,
            color=config.COLOR_TEST, label="Test (KYW)")
    ax.set_title(col)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
axes[0, 0].legend(loc="upper right")
fig.suptitle("Clinical variables: training vs test", fontweight="bold", color=config.TEXT_MAIN)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig02_clinical_distributions.png")
plt.close(fig)

# ===========================================================================
# D. Clinical centres
# ===========================================================================
report("\nD. CLINICAL CENTRES")
centres_train = sorted(clin_tr["CENTER"].unique())
centres_test = sorted(clin_te["CENTER"].unique())
report(f"  Number of training centres: {len(centres_train)}")
report(f"  Test centres: {centres_test}")
report(f"  Test centres also seen in training: {sorted(set(centres_test) & set(centres_train))}")

# Per-centre table: number of patients, deaths, median follow-up.
merged = clin_tr[["ID", "CENTER"]].merge(y, on="ID")   # only patients with outcome
center_summary = (merged.groupby("CENTER")
                  .agg(patients_with_outcome=("ID", "size"),
                       death_rate=("OS_STATUS", "mean"),
                       median_years=("OS_YEARS", "median"))
                  .round(2))
center_summary.insert(0, "patients_total", clin_tr["CENTER"].value_counts())
center_summary = center_summary.sort_values("patients_total", ascending=False)
center_summary.to_csv(config.TABLES / "center_summary.csv")
report(center_summary.to_string())

# ---- Figure 3: patients per centre ----------------------------------------
counts = clin_tr["CENTER"].value_counts()
counts_test = clin_te["CENTER"].value_counts()
fig, ax = plt.subplots(figsize=(7.2, 6.0))
labels = list(counts.index) + [f"{c} (test)" for c in counts_test.index]
values = list(counts.values) + list(counts_test.values)
colors = [config.COLOR_TRAIN] * len(counts) + [config.COLOR_TEST] * len(counts_test)
positions = np.arange(len(labels))
ax.barh(positions, values, color=colors, height=0.72)
for i, v in enumerate(values):
    ax.text(v + 10, i, str(v), va="center", fontsize=8, color=config.TEXT_SOFT)
ax.set_yticks(positions)
ax.set_yticklabels(labels, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("Number of patients")
ax.set_title("Patients per clinical centre (blue = training, orange = test)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig03_centers.png")
plt.close(fig)

# ===========================================================================
# E. Molecular (mutation) data
# ===========================================================================
report("\nE. MOLECULAR DATA (one row = one mutation)")
report(f"  Mutation rows: train {len(mol_tr)}, test {len(mol_te)}")
report("  Extra column not in the challenge description: DEPTH (sequencing read depth)")

# Mutations per patient. Patients with no row get 0 (no mutation found),
# which is information, NOT missing data.
per_patient_tr = mol_tr.groupby("ID").size().reindex(clin_tr["ID"], fill_value=0)
per_patient_te = mol_te.groupby("ID").size().reindex(clin_te["ID"], fill_value=0)
report(f"  Mean mutations per patient: train {per_patient_tr.mean():.2f}, test {per_patient_te.mean():.2f}")
report(f"  Patients with no mutation: train {(per_patient_tr == 0).sum()}, test {(per_patient_te == 0).sum()}")

# Rows with no chromosome position: these are special mutation types.
no_pos = mol_tr[mol_tr["CHR"].isna()]
report(f"  Training rows without CHR/START/END: {len(no_pos)} "
       f"(types: {no_pos['EFFECT'].value_counts().to_dict()}, genes: {no_pos['GENE'].value_counts().to_dict()})")

# Same gene mutated more than once in a patient (e.g. two TP53 hits).
multi = (mol_tr.groupby(["ID", "GENE"]).size() > 1).sum()
report(f"  (patient, gene) pairs with 2+ mutations in training: {multi}")

# Share of patients with at least one mutation in each gene.
# nunique() counts each patient once even if the gene is hit twice.
gene_tr = mol_tr.groupby("GENE")["ID"].nunique() / len(clin_tr)
gene_te = mol_te.groupby("GENE")["ID"].nunique() / len(clin_te)
genes = (pd.concat([gene_tr, gene_te], axis=1, keys=["pct_train", "pct_test"])
         .fillna(0) * 100).round(1)
genes = genes.sort_values("pct_train", ascending=False)
genes.to_csv(config.TABLES / "gene_frequencies.csv")
report(f"  Distinct genes: train {mol_tr['GENE'].nunique()}, test {mol_te['GENE'].nunique()}")
only_test = sorted(set(mol_te["GENE"]) - set(mol_tr["GENE"]))
report(f"  Genes found ONLY in test ({len(only_test)}): {', '.join(only_test)}")
report("  Top 10 genes (% of patients mutated):")
report(genes.head(10).to_string())

# ---- Figure 5: top 20 genes ------------------------------------------------
top = genes.head(20)
fig, ax = plt.subplots(figsize=(7.2, 6.0))
positions = np.arange(len(top))
width = 0.38
ax.barh(positions - width / 2, top["pct_train"], height=width, color=config.COLOR_TRAIN, label="Training")
ax.barh(positions + width / 2, top["pct_test"], height=width, color=config.COLOR_TEST, label="Test (KYW)")
ax.set_yticks(positions)
ax.set_yticklabels(top.index, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("Patients with at least one mutation in the gene (%)")
ax.set_title("20 most frequently mutated genes in training")
ax.legend(loc="lower right")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig05_gene_frequencies.png")
plt.close(fig)

# ===========================================================================
# F. How is CYTOGENETICS written?
# ===========================================================================
report("\nF. CYTOGENETICS TEXT FORMAT")


def describe_cyto(series, name):
    """Count the writing styles found in a column of karyotype strings."""
    s = series.dropna()
    n_missing = series.isna().sum()
    n_word_normal = (s.str.strip().str.lower() == "normal").sum()
    n_upper = s.str.contains("XX|XY").sum()        # upper-case sex chromosomes
    n_lower = s.str.contains("xx|xy").sum()        # lower-case sex chromosomes
    report(f"  {name}: missing {n_missing}, the word 'Normal' {n_word_normal}, "
           f"upper-case XX/XY {n_upper}, lower-case xx/xy {n_lower}")


describe_cyto(clin_tr["CYTOGENETICS"], "Training")
describe_cyto(clin_te["CYTOGENETICS"], "Test    ")
report("  -> In Stage 3 we must convert text to lower case and treat 'Normal' as a normal karyotype.")

# ===========================================================================
# Save the summary
# ===========================================================================
with open(config.TABLES / "audit_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\nSaved 5 figures to figures/ and 4 tables to results/tables/.")
