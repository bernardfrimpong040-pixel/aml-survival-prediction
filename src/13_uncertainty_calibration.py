"""
13_uncertainty_calibration.py  (Stage 8, part 2)
================================================
What a journal reviewer asks for beyond the C-index:

  1. UNCERTAINTY - 95% confidence intervals and p-values for each model's
     cross-validated C-index and for the key paired differences, using the
     corrected resampled t-test (Nadeau & Bengio 2003).
  2. TIME-DEPENDENT AUC at 1, 3 and 5 years: "of the patients who died by
     year t, how well does the score separate them from those still alive?"
  3. BRIER SCORE and CALIBRATION: do the predicted survival PROBABILITIES
     match what actually happens? (The C-index only checks the ranking.)

Why the correction in (1)? The 15 validation folds reuse the same patients
(3 repeats), and training sets overlap, so the 15 scores are not independent.
The usual standard error s / sqrt(15) is then too small. Nadeau & Bengio
inflate the variance:
        corrected variance = (1/J + n_valid/n_train) * s^2,   J = 15 folds
and use a t distribution with J - 1 = 14 degrees of freedom.

Outputs:
  figures/fig26_paired_ci.png
  figures/fig27_time_auc.png
  figures/fig28_calibration.png
  results/tables/cv_confidence_intervals.csv
  results/tables/time_dependent_auc.csv
  results/tables/brier_scores.csv
  results/tables/calibration_3y.csv

Run from the project's top folder, AFTER 12_ipssm_benchmark.py (about 1-2 minutes):
    python src/13_uncertainty_calibration.py
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import KaplanMeierFitter
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score, brier_score
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

import config
import cv_utils as U

warnings.filterwarnings("ignore")
plt = config.set_plot_style()
config.make_folders()

train, test, groups, folds = U.load_stage3()
y_all = U.survival_target(train)
FULL = [c for fam in ["clinical", "cytogenetic", "molecular_summary", "genes", "scores"]
        for c in groups[fam] if c != "log_MONOCYTES"]
ipssm = pd.read_csv(config.DATA_PROCESSED / "ipssm_scores.csv", index_col=0)
ipssm_tr = ipssm.loc[train.index, "ipssm_score"]

# ===========================================================================
# 1. Confidence intervals (corrected resampled t-test)
# ===========================================================================
ens = pd.read_csv(config.TABLES / "ensemble_cv_folds.csv")
cox = pd.read_csv(config.TABLES / "cox_cv_folds.csv")
S = {code: ens[ens["code"] == code].set_index(["repeat", "fold"])["ipcw_C"] for code in ["A", "D", "G", "E1"]}
S["M"] = pd.read_csv(config.TABLES / "ipssm_cv.csv").set_index(["repeat", "fold"])["ipcw_C"]
S["B"] = cox[cox["model"].str.startswith("B.")].set_index(["repeat", "fold"])["ipcw_C"]
S["C"] = cox[cox["model"].str.startswith("C.")].set_index(["repeat", "fold"])["ipcw_C"]
LABEL = {"A": "IPSS-R", "M": "IPSS-M", "B": "Benchmark-style Cox", "C": "IPSS-M-style Cox",
         "D": "Elastic-net Cox", "G": "XGBoost Cox", "E1": "Ensemble (final)"}

# Ratio of validation to training patients, averaged over the 15 folds.
ratios = []
for r in range(3):
    for k in range(5):
        n_va = int((folds[f"repeat{r}"] == k).sum())
        ratios.append(n_va / (len(train) - n_va))
ratio = float(np.mean(ratios))
J = 15
t_crit = stats.t.ppf(0.975, df=J - 1)
print(f"Average n_valid / n_train = {ratio:.3f};  correction factor 1/J + ratio = {1 / J + ratio:.3f}")
print(f"(the naive factor would be 1/J = {1 / J:.3f})")


def corrected(values):
    """mean, corrected 95% CI, p-value (vs 0), and the naive SE for comparison."""
    v = np.asarray(values, dtype=float)
    m, s = v.mean(), v.std(ddof=1)
    se = np.sqrt((1 / J + ratio) * s ** 2)
    p = 2 * stats.t.sf(abs(m) / se, df=J - 1) if se > 0 else np.nan
    return m, m - t_crit * se, m + t_crit * se, p, s / np.sqrt(J)


rows = []
for code in ["A", "M", "B", "C", "D", "G", "E1"]:
    m, lo, hi, _, naive = corrected(S[code])
    rows.append({"quantity": LABEL[code], "type": "C-index", "estimate": m, "ci_low": lo, "ci_high": hi,
                 "p_value": np.nan, "naive_se": naive})
PAIRS = [("M", "A", "IPSS-M vs IPSS-R"), ("C", "B", "IPSS-M-style Cox vs benchmark"),
         ("D", "A", "Elastic-net vs IPSS-R"), ("D", "M", "Elastic-net vs IPSS-M"),
         ("G", "D", "XGBoost vs elastic-net"), ("E1", "D", "Ensemble vs elastic-net"),
         ("E1", "M", "Ensemble vs IPSS-M")]
for a, b, label in PAIRS:
    m, lo, hi, p, naive = corrected(S[a] - S[b])
    rows.append({"quantity": label, "type": "paired difference", "estimate": m, "ci_low": lo,
                 "ci_high": hi, "p_value": p, "naive_se": naive})
ci = pd.DataFrame(rows)
ci_out = ci.copy()
for c in ["estimate", "ci_low", "ci_high", "naive_se"]:
    ci_out[c] = ci_out[c].round(4)
ci_out["p_value"] = ci_out["p_value"].map(lambda p: "" if pd.isna(p) else f"{p:.3g}")
ci_out.to_csv(config.TABLES / "cv_confidence_intervals.csv", index=False)
print("\nCross-validated C-index and paired differences, corrected 95% CI:")
print(ci_out.to_string(index=False))

# ---- Figure 26: paired differences with their corrected 95% CIs -----------
pairs = ci[ci["type"] == "paired difference"].iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.4, 4.0))
for i, row in pairs.iterrows():
    sig = row["ci_low"] > 0 or row["ci_high"] < 0
    colour = config.COLOR_TRAIN if sig else config.TEXT_SOFT
    ax.plot([row["ci_low"], row["ci_high"]], [i, i], color=colour, linewidth=2.2)
    ax.plot(row["estimate"], i, "o", color=colour, markersize=7, markeredgecolor="white", markeredgewidth=1.2)
    ax.text(1.02, i, f"{row['estimate']:+.4f}  [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]",
            transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=config.TEXT_SOFT)
ax.axvline(0, color=config.TEXT_MAIN, linewidth=1)
ax.set_yticks(range(len(pairs)))
ax.set_yticklabels(pairs["quantity"], fontsize=8.5)
ax.set_xlabel("Difference in IPCW C-index (corrected 95% CI)")
ax.set_title("Paired comparisons over 15 centre-grouped folds\n(blue = CI excludes 0, grey = includes 0)", fontsize=10)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.subplots_adjust(right=0.68)
fig.savefig(config.FIGURES / "fig26_paired_ci.png")
plt.close(fig)

# ===========================================================================
# 2. Time-dependent AUC at 1, 3 and 5 years (all 15 folds)
# ===========================================================================
long = pd.concat([pd.read_csv(config.DATA_PROCESSED / "oof_cox.csv"),
                  pd.read_csv(config.DATA_PROCESSED / "oof_trees.csv")], ignore_index=True)
long["code"] = long["model"].str[0]
pred = {r: long[long["repeat"] == f"repeat{r}"].pivot(index="ID", columns="code", values="risk")
            .loc[train.index] for r in range(3)}


def rank01(x):
    return pd.Series(x).rank().to_numpy() / len(x)


def censor_at(y, horizon=6.5):
    """
    Administrative censoring: follow-up beyond `horizon` years is cut at the
    horizon (and a death after it counts as 'alive at the horizon').
    Needed because a validation patient may be followed longer than anyone in
    the training part, where the censoring curve G(t) reaches 0 and the IPCW
    weight 1/G(t) cannot be computed. No result below uses times beyond 6
    years, so nothing that is evaluated changes.
    """
    y = y.copy()
    late = y["time"] > horizon
    y["event"] = y["event"] & ~late
    y["time"] = np.minimum(y["time"], horizon)
    return y


TIMES = np.array([1.0, 3.0, 5.0])
auc_rows = []
for r in range(3):
    for k in range(5):
        valid = (folds[f"repeat{r}"] == k).to_numpy()
        p = pred[r][valid]
        scores = {"A": p["A"].to_numpy(), "M": ipssm_tr[valid].to_numpy(), "D": p["D"].to_numpy(),
                  "G": p["G"].to_numpy(),
                  "E1": (rank01(p["D"].to_numpy()) + rank01(p["G"].to_numpy())) / 2}
        for code, risk in scores.items():
            auc, _ = cumulative_dynamic_auc(y_all[~valid], censor_at(y_all[valid]), risk, TIMES)
            for t, a in zip(TIMES, auc):
                auc_rows.append({"model": LABEL[code], "years": t, "repeat": r, "fold": k, "auc": a})
auc = pd.DataFrame(auc_rows)
auc_summary = auc.groupby(["model", "years"])["auc"].agg(["mean", "std"]).round(4).reset_index()
auc_summary.to_csv(config.TABLES / "time_dependent_auc.csv", index=False)
print("\nTime-dependent AUC (mean over 15 folds):")
print(auc_summary.pivot(index="model", columns="years", values="mean").to_string())

fig, ax = plt.subplots(figsize=(7.2, 4.0))
style = {"IPSS-R": (config.TEXT_SOFT, "--"), "IPSS-M": (config.COLOR_GROUP3, "--"),
         "Elastic-net Cox": (config.COLOR_TRAIN, "-"), "XGBoost Cox": ("#6250d6", "-"),
         "Ensemble (final)": (config.COLOR_TEST, "-")}
for name, (colour, ls) in style.items():
    h = auc_summary[auc_summary["model"] == name]
    ax.plot(h["years"], h["mean"], color=colour, linestyle=ls, linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1.5, label=name)
ax.set_xticks(TIMES)
ax.set_xlabel("Years since diagnosis (t)")
ax.set_ylabel("Time-dependent AUC (mean of 15 folds)")
ax.set_title("Separating deaths by year t from survivors past year t")
ax.legend(loc="lower left", fontsize=8.5)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig27_time_auc.png")
plt.close(fig)

# ===========================================================================
# 3. Brier score and calibration (repeat 0, out-of-fold probabilities)
# ===========================================================================
# Three models that give survival PROBABILITIES:
#   * Kaplan-Meier "null" model: the same curve for everyone (no features),
#   * IPSS-M recalibrated: a Cox model with the IPSS-M score as its only input,
#     fitted on the training part of the fold (the weights stay published;
#     only the link from score to probability is estimated),
#   * Elastic-net Cox with its baseline survival curve.
GRID = np.linspace(0.25, 6.0, 24)            # years at which probabilities are compared
brier_rows, s3 = [], pd.DataFrame(index=train.index, columns=["Elastic-net Cox", "IPSS-M (recalibrated)"],
                                  dtype=float)
for k in range(5):
    valid = (folds["repeat0"] == k).to_numpy()
    tr_part, va_part = train[~valid], train[valid]
    y_tr, y_va = y_all[~valid], censor_at(y_all[valid])
    times = GRID[GRID < y_va["time"].max()]           # Brier needs times inside the fold's follow-up

    t_km, s_km = kaplan_meier_estimator(y_tr["event"], y_tr["time"])
    surv_km = np.tile(np.interp(times, t_km, s_km), (valid.sum(), 1))

    cox_m = CoxPHSurvivalAnalysis().fit(ipssm_tr[~valid].to_frame(), y_tr)
    surv_m = np.vstack([fn(times) for fn in cox_m.predict_survival_function(ipssm_tr[valid].to_frame())])

    enet, alpha, _, _ = U.fit_coxnet_nested(tr_part[FULL], y_tr, tr_part["CENTER"].to_numpy(), fit_baseline=True)
    surv_d = np.vstack([fn(times) for fn in enet.predict_survival_function(va_part[FULL], alpha=alpha)])

    for name, surv in [("Kaplan-Meier (no features)", surv_km), ("IPSS-M (recalibrated)", surv_m),
                       ("Elastic-net Cox", surv_d)]:
        ibs = integrated_brier_score(y_tr, y_va, surv, times)
        i3 = int(np.argmin(np.abs(times - 3.0)))
        b3 = brier_score(y_tr, y_va, surv[:, i3], times[i3])[1][0]
        brier_rows.append({"model": name, "fold": k, "integrated_brier": ibs, "brier_3y": b3})
    i3 = int(np.argmin(np.abs(times - 3.0)))
    s3.loc[valid, "Elastic-net Cox"] = surv_d[:, i3]
    s3.loc[valid, "IPSS-M (recalibrated)"] = surv_m[:, i3]
    print(f"  fold {k} done")

brier = pd.DataFrame(brier_rows).groupby("model")[["integrated_brier", "brier_3y"]].mean().round(4)
brier = brier.reindex(["Kaplan-Meier (no features)", "IPSS-M (recalibrated)", "Elastic-net Cox"])
brier.to_csv(config.TABLES / "brier_scores.csv")
print("\nBrier scores (lower = better; repeat 0, mean of 5 folds):")
print(brier.to_string())

# ---- Calibration at 3 years: predicted vs observed, by tenths of risk -------
cal_rows = []
for name in s3.columns:
    decile = pd.qcut(s3[name], 10, labels=False, duplicates="drop")
    for d in sorted(decile.unique()):
        m = decile == d
        kmf = KaplanMeierFitter().fit(train.loc[m, "OS_YEARS"], train.loc[m, "OS_STATUS"])
        obs = float(kmf.survival_function_at_times(3.0).iloc[0])
        ci_ = kmf.confidence_interval_survival_function_
        row_ci = ci_[ci_.index <= 3.0].iloc[-1]
        cal_rows.append({"model": name, "group": int(d) + 1, "patients": int(m.sum()),
                         "predicted_S3": float(s3.loc[m, name].mean()), "observed_S3": obs,
                         "obs_low": float(row_ci.iloc[0]), "obs_high": float(row_ci.iloc[1])})
cal = pd.DataFrame(cal_rows)
cal["abs_error"] = (cal["predicted_S3"] - cal["observed_S3"]).abs()
cal.round(4).to_csv(config.TABLES / "calibration_3y.csv", index=False)
print("\nMean absolute calibration error at 3 years (10 risk groups):")
print(cal.groupby("model")["abs_error"].mean().round(4).to_string())

fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.plot([0, 1], [0, 1], color=config.TEXT_SOFT, linestyle="--", linewidth=1, label="perfect calibration")
for name, colour, marker in [("Elastic-net Cox", config.COLOR_TRAIN, "o"),
                             ("IPSS-M (recalibrated)", config.COLOR_TEST, "s")]:
    c = cal[cal["model"] == name]
    ax.errorbar(c["predicted_S3"], c["observed_S3"],
                yerr=[c["observed_S3"] - c["obs_low"], c["obs_high"] - c["observed_S3"]],
                fmt=marker, color=colour, ecolor=colour, elinewidth=1, capsize=2, markersize=6,
                markeredgecolor="white", label=name)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.set_xlabel("Predicted 3-year survival (mean of risk group)")
ax.set_ylabel("Observed 3-year survival (Kaplan-Meier, 95% CI)")
ax.set_title("Calibration at 3 years\n(out-of-fold, 10 risk groups)", fontsize=10)
ax.legend(loc="upper left", fontsize=8.5)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig28_calibration.png")
plt.close(fig)

print("\nDone: 3 figures, 4 tables.")
