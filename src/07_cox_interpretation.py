"""
07_cox_interpretation.py  (Stage 4, part 2)
===========================================
What do the Cox models say, and are their assumptions reasonable?

  1. Hazard ratios of the IPSS-M-style Cox model (24 variables), with 95%
     confidence intervals, fitted with lifelines on all training patients.
  2. The proportional-hazards (PH) check: Schoenfeld-residual tests and plots.
  3. The elastic-net Cox coefficients chosen in 06_cox_models.py.
  4. Kaplan-Meier curves of OUT-OF-FOLD risk groups: patients are grouped by
     a risk score from a model that never saw their centre.

Outputs:
  figures/fig15_hazard_ratios.png
  figures/fig16_schoenfeld.png
  figures/fig17_elasticnet_coefs.png
  figures/fig18_km_oof_quartiles.png
  results/tables/cox_hazard_ratios.csv
  results/tables/ph_test.csv
  results/tables/km_oof_quartiles.csv

Run from the project's top folder (needs 06_cox_models.py to have run first):
    python src/07_cox_interpretation.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.ticker
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test, multivariate_logrank_test

import config
import cv_utils as U

plt = config.set_plot_style()
config.make_folders()

train, test, groups, folds = U.load_stage3()
train = U.add_ipssm_residual(train)

# ===========================================================================
# 1. Hazard ratios of the IPSS-M-style Cox model
# ===========================================================================
CONTINUOUS = ["BM_BLAST", "HB", "log_WBC", "log_ANC", "log_PLT"]   # -> per standard deviation
COUNTS = ["cyto_ipssr_cat", "ipssm_residual"]                       # -> per one step
BINARY = ["cyto_missing"] + U.IPSSM_MAIN                            # -> mutated vs not
cols = CONTINUOUS + COUNTS + BINARY

df = train[cols].copy()
df = df.fillna(df.median())                    # one-off fit on all data, so all-data medians are fine here
sd = df[CONTINUOUS].std()
df[CONTINUOUS] = (df[CONTINUOUS] - df[CONTINUOUS].mean()) / sd      # z-scores
df["OS_YEARS"] = train["OS_YEARS"]
df["OS_STATUS"] = train["OS_STATUS"]

# A tiny penalty keeps the fit stable for rare 0/1 genes (e.g. MLL-PTD).
cph = CoxPHFitter(penalizer=0.001)
cph.fit(df, duration_col="OS_YEARS", event_col="OS_STATUS")

# Hazard ratio HR = exp(coefficient). HR = 2 means "twice the instantaneous
# risk of death", holding the other variables fixed.
s = cph.summary
hr = pd.DataFrame({
    "variable": s.index,
    "unit": ["per 1 SD" if v in CONTINUOUS else "per step" if v in COUNTS else "yes vs no" for v in s.index],
    "HR": s["exp(coef)"].round(3).to_numpy(),
    "lower_95": s["exp(coef) lower 95%"].round(3).to_numpy(),
    "upper_95": s["exp(coef) upper 95%"].round(3).to_numpy(),
    "p_value": s["p"].map(lambda p: f"{p:.1e}").to_numpy(),
})
hr.to_csv(config.TABLES / "cox_hazard_ratios.csv", index=False)
print("Hazard ratios (IPSS-M-style Cox, all training patients):")
print(hr.to_string(index=False))
print("\nOne standard deviation of each continuous variable:")
print(sd.round(3).to_string())
print(f"\nConcordance on the training data itself (optimistic): {cph.concordance_index_:.4f}")

# ---- Figure 15: forest plot ------------------------------------------------
order = hr.sort_values("HR").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.2, 7.2))
for i, row in order.iterrows():
    significant = not (row["lower_95"] <= 1 <= row["upper_95"])
    colour = (config.COLOR_TEST if row["HR"] > 1 else config.COLOR_TRAIN) if significant else config.TEXT_SOFT
    ax.plot([row["lower_95"], row["upper_95"]], [i, i], color=colour, linewidth=2)
    ax.plot(row["HR"], i, "o", color=colour, markersize=7, markeredgecolor="white", markeredgewidth=1.2)
    # Text just outside the right edge of the plot (x = 1.02 in axes units).
    ax.text(1.02, i, f"{row['HR']:.2f} ({row['lower_95']:.2f}-{row['upper_95']:.2f})",
            va="center", fontsize=7.5, color=config.TEXT_SOFT,
            transform=ax.get_yaxis_transform())
ax.axvline(1, color=config.TEXT_MAIN, linewidth=1)
ax.set_xscale("log")                       # ratios are symmetric on a log scale: 0.5 and 2 are equally far from 1
ax.set_xlim(0.4, 4.5)
ax.set_xticks([0.5, 1, 2, 4])
ax.set_xticklabels(["0.5", "1", "2", "4"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())   # no labels on small ticks
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{v} ({u})" for v, u in zip(order["variable"], order["unit"])], fontsize=8)
ax.set_xlabel("Hazard ratio with 95% confidence interval (log scale)")
ax.set_title("IPSS-M-style Cox model\norange = higher risk, blue = lower risk, grey = not significant")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.subplots_adjust(right=0.80)            # leave room for the HR text on the right
fig.savefig(config.FIGURES / "fig15_hazard_ratios.png")
plt.close(fig)

# ===========================================================================
# 2. Proportional-hazards check
# ===========================================================================
# The Cox model assumes each hazard ratio is CONSTANT over time. The test
# looks at "scaled Schoenfeld residuals": if a variable's effect fades or
# grows with time, its residuals show a trend against time. A small p-value
# = evidence the effect changes over time. With 3,173 patients even tiny,
# harmless changes become "significant", so we also LOOK at the size of the trend.
ph = proportional_hazard_test(cph, df, time_transform="rank")
ph_table = ph.summary[["test_statistic", "p"]].sort_values("test_statistic", ascending=False)
ph_table["test_statistic"] = ph_table["test_statistic"].round(2)
ph_table["p"] = ph_table["p"].map(lambda p: f"{p:.1e}")
ph_table.index.name = "variable"
ph_table.to_csv(config.TABLES / "ph_test.csv")
print("\nProportional-hazards test (largest statistics first):")
print(ph_table.head(10).to_string())
n_viol = (ph.summary["p"] < 0.05).sum()
print(f"Variables with p < 0.05: {n_viol} of {len(ph.summary)}")

# ---- Figure 16: Schoenfeld residuals -----------------------------------------
# Left: the strongest violator overall. Right: the strongest CONTINUOUS
# violator (residuals of a 0/1 variable form two bands and are hard to read).
resid = cph.compute_residuals(df, kind="scaled_schoenfeld")   # one row per death
death_times = df.loc[resid.index, "OS_YEARS"]
ranked = list(ph.summary["test_statistic"].sort_values(ascending=False).index)
top_continuous = [v for v in ranked if v in CONTINUOUS][0]
top2 = [ranked[0], top_continuous] if ranked[0] != top_continuous else ranked[:2]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
for ax, var in zip(axes, top2):
    # Grambsch & Therneau: (scaled residual + coefficient) estimates the effect
    # beta(t) at the time of each death. lifelines does NOT add the coefficient
    # (R does), so we add it here. A flat trend at the dashed line = PH holds.
    beta = cph.params_[var]
    r = resid[var] + beta
    ordr = np.argsort(death_times.to_numpy())
    t_sorted = death_times.to_numpy()[ordr]
    r_sorted = r.to_numpy()[ordr]
    ax.scatter(t_sorted, r_sorted, s=6, color=config.COLOR_TRAIN, alpha=0.25, linewidth=0)
    # Moving average of 150 neighbouring deaths shows the trend.
    trend = pd.Series(r_sorted).rolling(150, center=True, min_periods=50).mean()
    ax.plot(t_sorted, trend, color=config.TEXT_MAIN, linewidth=2.2, label="moving average = beta(t)")
    ax.axhline(beta, color=config.COLOR_TEST, linestyle="--", linewidth=1.5,
               label=f"constant effect, beta = {beta:.2f}")
    ax.axhline(0, color=config.TEXT_SOFT, linewidth=0.8)
    lo, hi = np.nanpercentile(r_sorted, [2, 98])
    ax.set_ylim(lo, hi)
    ax.set_xlim(0, 8)
    ax.set_xlabel("Time of death (years)")
    ax.set_ylabel("Estimated log hazard ratio beta(t)")
    stat = ph.summary.loc[var, "test_statistic"]
    ax.set_title(f"{var}: PH test statistic {stat:.1f}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig16_schoenfeld.png")
plt.close(fig)

# ===========================================================================
# 3. Elastic-net Cox coefficients (from 06_cox_models.py)
# ===========================================================================
with open(config.TABLES / "coxnet_selected.json", encoding="utf-8") as f:
    enet = json.load(f)
coef = pd.Series(enet["coefficients"])
top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(20).sort_values()
print(f"\nElastic-net: l1_ratio {enet['l1_ratio']}, alpha {enet['alpha']:.4f}, "
      f"{enet['n_nonzero_coefficients']} non-zero coefficients. Largest 20 (per 1 SD):")
print(top.round(4).to_string())

fig, ax = plt.subplots(figsize=(7.2, 5.8))
colours = [config.COLOR_TEST if v > 0 else config.COLOR_TRAIN for v in top]
ax.barh(top.index, top.to_numpy(), color=colours, height=0.7)
for i, v in enumerate(top):
    ax.text(v + (0.004 if v > 0 else -0.004), i, f"{v:+.3f}", va="center",
            ha="left" if v > 0 else "right", fontsize=7.5, color=config.TEXT_SOFT)
ax.axvline(0, color=config.TEXT_MAIN, linewidth=1)
lim = top.abs().max() * 1.35
ax.set_xlim(-lim, lim)
ax.set_xlabel("Coefficient per 1 standard deviation (log hazard ratio)")
ax.set_title(f"Elastic-net Cox: 20 largest of {enet['n_nonzero_coefficients']} non-zero coefficients")
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig17_elasticnet_coefs.png")
plt.close(fig)

# ===========================================================================
# 4. Out-of-fold risk quartiles
# ===========================================================================
# Each patient's score comes from the model trained WITHOUT their centre.
# Scores from different folds come from different models, so we form the
# quartiles WITHIN each fold (rank inside the fold), then pool the folds.
oof = pd.read_csv(config.DATA_PROCESSED / "oof_repeat0.csv", index_col="ID")
model_col = [c for c in oof.columns if c.startswith("D.")][0]
fold0 = folds.loc[oof.index, "repeat0"]
quartile = oof.groupby(fold0)[model_col].transform(lambda x: pd.qcut(x, 4, labels=False))
labels = ["Q1 (lowest risk)", "Q2", "Q3", "Q4 (highest risk)"]
ramp = ["#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"]

res = multivariate_logrank_test(oof["OS_YEARS"], quartile, oof["OS_STATUS"])
fig, ax = plt.subplots(figsize=(7.2, 4.2))
rows = []
for q, (lab, colour) in enumerate(zip(labels, ramp)):
    m = quartile == q
    k = KaplanMeierFitter().fit(oof.loc[m, "OS_YEARS"], oof.loc[m, "OS_STATUS"])
    sf = k.survival_function_
    ax.step(sf.index, sf.iloc[:, 0], where="post", color=colour, linewidth=2, label=f"{lab} (n={m.sum()})")
    rows.append({"quartile": lab, "patients": int(m.sum()),
                 "median_years": round(k.median_survival_time_, 2),
                 "S_at_3_years": round(float(k.survival_function_at_times(3).iloc[0]), 3),
                 "S_at_7_years": round(float(k.survival_function_at_times(7).iloc[0]), 3)})
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Years since diagnosis")
ax.set_ylabel("Probability of being alive")
ax.set_title(f"Elastic-net Cox, out-of-fold risk quartiles (unseen centres)\n"
             f"log-rank chi2 = {res.test_statistic:.1f} (3 df), p < 0.001", fontsize=10)
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig18_km_oof_quartiles.png")
plt.close(fig)
km_q = pd.DataFrame(rows)
km_q.to_csv(config.TABLES / "km_oof_quartiles.csv", index=False)
print("\nOut-of-fold risk quartiles (elastic-net Cox):")
print(km_q.to_string(index=False))

print("\nDone: 4 figures, 3 tables.")
