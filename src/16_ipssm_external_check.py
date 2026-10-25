"""
16_ipssm_external_check.py  (Stage 9 add-on)
=============================================
Why did the published IPSS-M fall so far behind our models at the test centre?

What we saw:
  cross-validation : ensemble - IPSS-M = +0.005   (15 centre-grouped folds)
  public score     : ensemble - IPSS-M = +0.030   (0.7552 - 0.7255, centre KYW)

The gap at the test centre is six times the cross-validated gap. Before
writing any explanation in the paper we TEST three candidate explanations on
the training data, where we know the outcomes:

  1. Case mix.  The test centre is sicker (33% IPSS-R very high vs 11%).
     Does the gap grow if we make the validation folds look like the test
     centre?  -> resample each validation fold to the test centre's mix of
     IPSS-R risk groups and recompute the gap.

  2. Missing clinical values.  Blood counts / blasts are missing for about
     9.5% of test patients but only 2.5% of training patients. Both models
     fill a missing value with a typical value (IPSS-M: its reference mean;
     our models: the training median). Does losing these values hurt the
     IPSS-M more than our model?  -> blank them for an extra 7% of each
     validation fold and see how much each model's C-index drops.

  3. Centre-to-centre variation.  Maybe the gap simply varies a lot from
     hospital to hospital and KYW happens to be at the high end.
     -> score the gap separately inside each large training centre.

Outputs:
  figures/fig29_ipssm_gap_by_centre.png
  results/tables/ipssm_gap_by_centre.csv
  results/tables/ipssm_external_checks.csv

Run from the project's top folder, AFTER 12_ipssm_benchmark.py
(about 1 minute: check 2 refits the elastic-net 15 times):
    python src/16_ipssm_external_check.py
"""

import json
import warnings

import numpy as np
import pandas as pd

import config
import cv_utils as U
import features as F
import ipssm as M

warnings.filterwarnings("ignore")      # silence harmless convergence warnings
plt = config.set_plot_style()
config.make_folders()

train, test, groups, folds = U.load_stage3()
y_all = U.survival_target(train)

# Public leaderboard scores at the test centre (typed in from Challenge Data).
PUBLIC_ENSEMBLE = 0.7552
PUBLIC_IPSSM = 0.7255
PUBLIC_GAP = PUBLIC_ENSEMBLE - PUBLIC_IPSSM

# ---------------------------------------------------------------------------
# Risk scores we need, for every training patient and every CV repeat
# ---------------------------------------------------------------------------
# IPSS-M: no fitting, so one score per patient (the same in every repeat).
ipssm_train = M.ipssm_score(train)[0]

# Out-of-fold elastic-net (D) and XGBoost (G) scores saved by scripts 06 and 08.
long = pd.concat([pd.read_csv(config.DATA_PROCESSED / "oof_cox.csv"),
                  pd.read_csv(config.DATA_PROCESSED / "oof_trees.csv")], ignore_index=True)
long["code"] = long["model"].str[0]          # "D. Elastic-net ..." -> "D"
pred = {r: long[long["repeat"] == f"repeat{r}"].pivot(index="ID", columns="code", values="risk")
            .loc[train.index] for r in range(3)}


def rank01(x):
    """Ranks between 0 and 1 (same rule as 09_ensemble.py)."""
    return pd.Series(x).rank(method="average").to_numpy() / len(x)


def ensemble_scores(r, valid):
    """Ensemble = average rank of elastic-net and XGBoost inside one validation fold."""
    p = pred[r].loc[valid]
    return pd.Series((rank01(p["D"].to_numpy()) + rank01(p["G"].to_numpy())) / 2, index=p.index)


rows_summary = []

# ===========================================================================
# Check 1: case mix. Resample validation folds to the test centre's risk mix.
# ===========================================================================
# IPSS-R risk groups (Greenberg 2012 cut-points).
def risk_group(t):
    return pd.cut(t["ipssr_score"], [-1, 1.5, 3, 4.5, 6, 99],
                  labels=["very low", "low", "intermediate", "high", "very high"])


g_train, g_test = risk_group(train), risk_group(test)
target = g_test.value_counts(normalize=True)          # share of each group at KYW
print("IPSS-R risk groups (share of patients):")
print(pd.DataFrame({"train": g_train.value_counts(normalize=True),
                    "test (KYW)": target}).round(3).to_string())

rng = np.random.default_rng(config.SEED)
gaps_casemix = []
for r in range(3):
    for k in range(5):
        valid = (folds[f"repeat{r}"] == k).to_numpy()
        ids = train.index[valid]
        ens = ensemble_scores(r, ids)
        g = g_train.loc[ids]
        # Largest sample we can draw WITHOUT replacement while matching the
        # test shares: limited by the group that is scarcest relative to its target.
        n = int(min(g.value_counts()[c] / target[c] for c in target.index))
        fold_gaps = []
        for b in range(20):                                # 20 resamples per fold
            pick = np.concatenate([rng.choice(g.index[g == c], int(round(n * target[c])),
                                              replace=False) for c in target.index])
            sub = train.index.isin(pick)
            c_ens = U.ipcw_c(y_all[~valid], y_all[sub], ens.loc[train.index[sub]])
            c_ipssm = U.ipcw_c(y_all[~valid], y_all[sub], ipssm_train[sub])
            fold_gaps.append(c_ens - c_ipssm)
        gaps_casemix.append(np.mean(fold_gaps))
gaps_casemix = np.array(gaps_casemix)
rows_summary.append({"check": "1. Validation folds resampled to the test centre's IPSS-R mix",
                     "ensemble_minus_ipssm": round(gaps_casemix.mean(), 4),
                     "detail": f"ensemble wins {(gaps_casemix > 0).sum()}/15 folds"})

# ===========================================================================
# Check 2: missing clinical values. Blank them for an extra 7% of patients.
# ===========================================================================
CLIN = ["BM_BLAST", "HB", "log_WBC", "log_ANC", "log_MONOCYTES", "log_PLT"]
FULL = [c for fam in ["clinical", "cytogenetic", "molecular_summary", "genes", "scores"]
        for c in groups[fam] if c != "log_MONOCYTES"]          # elastic-net's 84 features
with open(config.TABLES / "coxnet_selected.json", encoding="utf-8") as f:
    sel = json.load(f)                                         # final l1_ratio and alpha


def clinical_frame(t):
    """Raw-scale clinical values in the layout F.ipssr_points expects."""
    return pd.DataFrame({"ID": t.index, "BM_BLAST": t["BM_BLAST"].to_numpy(),
                         "HB": t["HB"].to_numpy(), "PLT": np.expm1(t["log_PLT"]).to_numpy(),
                         "ANC": np.expm1(t["log_ANC"]).to_numpy()})


# IPSS-R points for a missing component = training median points (as in script 05).
fill_points = F.ipssr_points(clinical_frame(train), train["cyto_ipssr_cat"]).median()


def blank(t, ids):
    """Copy of t with the clinical values of patients `ids` removed, and their
    IPSS-R score recomputed the way script 05 does for missing values."""
    t = t.copy()
    t.loc[ids, CLIN] = np.nan
    pts = F.ipssr_points(clinical_frame(t.loc[ids]), t.loc[ids, "cyto_ipssr_cat"])
    t.loc[ids, "ipssr_score"] = pts.fillna(fill_points).sum(axis=1).to_numpy()
    return t


drops_enet, drops_ipssm = [], []
for r in range(3):
    for k in range(5):
        valid = (folds[f"repeat{r}"] == k).to_numpy()
        va = train.index[valid]
        # Elastic-net with the final settings, fitted on the training folds only.
        model = U._coxnet(sel["l1_ratio"], alphas=[sel["alpha"]])
        model.fit(train.loc[~valid, FULL], y_all[~valid])
        c_enet0 = U.ipcw_c(y_all[~valid], y_all[valid], model.predict(train.loc[va, FULL]))
        c_ipssm0 = U.ipcw_c(y_all[~valid], y_all[valid], M.ipssm_score(train.loc[va])[0])
        for b in range(3):                                     # 3 random blankings
            have = va[train.loc[va, "HB"].notna()]             # only patients who HAVE values
            ids = rng.choice(have, int(round(0.07 * len(va))), replace=False)
            tb = blank(train.loc[va], ids)
            drops_enet.append(c_enet0 - U.ipcw_c(y_all[~valid], y_all[valid], model.predict(tb[FULL])))
            drops_ipssm.append(c_ipssm0 - U.ipcw_c(y_all[~valid], y_all[valid], M.ipssm_score(tb)[0]))
d_enet, d_ipssm = np.mean(drops_enet), np.mean(drops_ipssm)
rows_summary.append({"check": "2. Extra 7% of clinical values blanked: widening of the elastic-net vs IPSS-M gap",
                     "ensemble_minus_ipssm": round(d_ipssm - d_enet, 4),
                     "detail": f"C-index drop: elastic-net {d_enet:.4f}, IPSS-M {d_ipssm:.4f}"})

# ===========================================================================
# Check 3: the gap inside each large training centre
# ===========================================================================
# Each centre sits in exactly one validation fold per repeat, so its
# out-of-fold scores come from models that never saw that centre.
rows = []
for centre, n_c in train["CENTER"].value_counts().items():
    if n_c < 100:                                  # too small to score reliably
        continue
    in_c = (train["CENTER"] == centre).to_numpy()
    c_ens, c_ipssm = [], []
    for r in range(3):
        k = folds.loc[in_c, f"repeat{r}"].iloc[0]  # the fold holding this centre
        valid = (folds[f"repeat{r}"] == k).to_numpy()
        ens = ensemble_scores(r, train.index[valid])
        c_ens.append(U.ipcw_c(y_all[~valid], y_all[in_c], ens.loc[train.index[in_c]]))
        c_ipssm.append(U.ipcw_c(y_all[~valid], y_all[in_c], ipssm_train[in_c]))
    rows.append({"centre": centre, "patients": int(n_c), "C_ipssm": round(np.mean(c_ipssm), 4),
                 "C_ensemble": round(np.mean(c_ens), 4),
                 "ensemble_minus_ipssm": round(np.mean(c_ens) - np.mean(c_ipssm), 4)})
by_centre = pd.DataFrame(rows).sort_values("ensemble_minus_ipssm")
by_centre.to_csv(config.TABLES / "ipssm_gap_by_centre.csv", index=False)
print("\nEnsemble minus IPSS-M inside each training centre (>= 100 patients):")
print(by_centre.to_string(index=False))

gap_sd = by_centre["ensemble_minus_ipssm"].std()
rows_summary.append({"check": "3. Spread of the gap across training centres (sd)",
                     "ensemble_minus_ipssm": round(gap_sd, 4),
                     "detail": f"range {by_centre['ensemble_minus_ipssm'].min():+.4f} to "
                               f"{by_centre['ensemble_minus_ipssm'].max():+.4f}"})
rows_summary.append({"check": "Test centre KYW (public leaderboard)",
                     "ensemble_minus_ipssm": round(PUBLIC_GAP, 4),
                     "detail": f"{PUBLIC_ENSEMBLE} - {PUBLIC_IPSSM}"})
summary = pd.DataFrame(rows_summary)
summary.to_csv(config.TABLES / "ipssm_external_checks.csv", index=False)
print("\nSummary of the three checks:")
print(summary.to_string(index=False))

# ---- Figure 29: gap by centre, with the test centre added ------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
labels = [f"{c} ({n})" for c, n in zip(by_centre["centre"], by_centre["patients"])] + ["KYW test (public)"]
values = list(by_centre["ensemble_minus_ipssm"]) + [PUBLIC_GAP]
colours = [config.COLOR_TRAIN] * len(by_centre) + [config.COLOR_TEST]
ax.barh(labels, values, color=colours, height=0.6)
ax.axvline(0, color=config.TEXT_SOFT, linewidth=1)
ax.axvline(0.0047, color=config.COLOR_GROUP3, linewidth=1.5, linestyle="--",
           label="cross-validated gap (+0.005)")
for i, v in enumerate(values):
    ax.annotate(f"{v:+.3f}", (v, i), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=config.TEXT_MAIN)
ax.set_xlim(-0.02, 0.04)
ax.set_xlabel("Ensemble minus IPSS-M (IPCW C-index)")
ax.set_title("The IPSS-M gap varies from centre to centre")
ax.legend(loc="lower right", fontsize=8.5)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig29_ipssm_gap_by_centre.png")
plt.close(fig)

print("\nDone: 1 figure, 2 tables.")
