"""
03_kaplan_meier.py  (Stage 2, part 1)
=====================================
Kaplan-Meier survival curves and log-rank tests.

What this script does:
  1. Computes the Kaplan-Meier (KM) curve BY HAND and checks it against
     the lifelines library.
  2. Draws the overall survival curve with a 95% confidence band.
  3. Compares survival between groups of patients (blast level, TP53,
     SF3B1, monosomy 7) and tests the differences with log-rank tests.
  4. Computes one log-rank test BY HAND (TP53) and checks it against lifelines.
  5. Estimates the CENSORING curve G(t) ("reverse Kaplan-Meier"), which is
     exactly what the challenge metric uses for its weights.

Outputs:
  figures/fig06_km_overall.png
  figures/fig07_km_groups.png
  figures/fig08_censoring_curve.png
  results/tables/km_first_steps.csv
  results/tables/km_overall_at_years.csv
  results/tables/logrank_tests.csv
  results/tables/logrank_tp53_by_hand.txt

Run from the project's top folder:
    python src/03_kaplan_meier.py
"""

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy.stats import chi2          # chi-square distribution, for p-values

import config
from load_data import load_train_with_outcome

plt = config.set_plot_style()
config.make_folders()

# ===========================================================================
# 0. Data: training patients WITH an outcome (3,173 patients)
# ===========================================================================
train, mol = load_train_with_outcome()
T = train["OS_YEARS"].to_numpy()      # observed time (years)
E = train["OS_STATUS"].to_numpy()     # 1 = died, 0 = censored
print(f"Patients: {len(train)}  deaths: {E.sum()}  censored: {(E == 0).sum()}")


# ===========================================================================
# 1. Kaplan-Meier BY HAND
# ===========================================================================
# The idea: survival is a chain of "conditional survivals".
# At each time t_k when at least one death happens:
#     n_k = number of patients still "at risk" just before t_k
#           (not yet died and not yet censored)
#     d_k = number of deaths at t_k
#     probability of surviving past t_k, given alive just before it = 1 - d_k / n_k
# S(t) = product of (1 - d_k / n_k) over all death times t_k <= t.
# Censored patients never cause a drop; they simply leave the risk set.

def kaplan_meier_by_hand(times, events):
    """Return a table with one row per distinct death time."""
    table = pd.DataFrame({"time": times, "event": events})
    death_times = np.sort(table.loc[table["event"] == 1, "time"].unique())

    rows = []
    survival = 1.0
    for t in death_times:
        n_at_risk = (table["time"] >= t).sum()                                  # still in the study at t
        n_deaths = ((table["time"] == t) & (table["event"] == 1)).sum()         # deaths exactly at t
        factor = 1 - n_deaths / n_at_risk                                       # conditional survival
        survival = survival * factor                                            # chain rule
        rows.append({"time": t, "n_at_risk": n_at_risk, "deaths": n_deaths,
                     "factor_1_minus_d_over_n": factor, "S_t": survival})
    return pd.DataFrame(rows)


km_hand = kaplan_meier_by_hand(T, E)

# Show the first 6 steps with full working.
print("\nKaplan-Meier by hand - first 6 death times:")
first = km_hand.head(6).copy()
for _, r in first.iterrows():
    print(f"  t = {r['time']:.4f} y: n = {int(r['n_at_risk'])}, d = {int(r['deaths'])}, "
          f"1 - {int(r['deaths'])}/{int(r['n_at_risk'])} = {r['factor_1_minus_d_over_n']:.5f}, "
          f"S(t) = {r['S_t']:.5f}")
first.round(5).to_csv(config.TABLES / "km_first_steps.csv", index=False)

# The same curve from lifelines.
kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E, label="All patients")

# Compare at every death time: largest difference should be ~0 (rounding only).
lifelines_values = kmf.survival_function_at_times(km_hand["time"]).to_numpy()
max_diff = np.abs(lifelines_values - km_hand["S_t"].to_numpy()).max()
print(f"Largest difference, hand vs lifelines: {max_diff:.2e}  (0 means identical)")

# Survival at 1, 2, 3, 5 and 7 years, with 95% confidence intervals.
years = [1, 2, 3, 5, 7]
ci = kmf.confidence_interval_survival_function_
rows = []
for t in years:
    s = float(kmf.survival_function_at_times(t).iloc[0])
    # The confidence interval table is indexed by time; take the last row at or before t.
    ci_row = ci[ci.index <= t].iloc[-1]
    rows.append({"years": t, "S_t": round(s, 3),
                 "lower_95": round(ci_row.iloc[0], 3), "upper_95": round(ci_row.iloc[1], 3)})
at_years = pd.DataFrame(rows)
at_years.to_csv(config.TABLES / "km_overall_at_years.csv", index=False)
print("\nOverall survival at chosen years:")
print(at_years.to_string(index=False))
print(f"Median overall survival: {kmf.median_survival_time_:.2f} years")

# ---- Figure 6: overall KM curve --------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
sf = kmf.survival_function_
ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], step="post",
                color=config.COLOR_TRAIN, alpha=0.18, linewidth=0, label="95% confidence band")
ax.step(sf.index, sf.iloc[:, 0], where="post", color=config.COLOR_TRAIN, linewidth=2,
        label="Kaplan-Meier estimate S(t)")
median = kmf.median_survival_time_
ax.plot([0, median], [0.5, 0.5], color=config.TEXT_SOFT, linestyle=":", linewidth=1.2)
ax.plot([median, median], [0, 0.5], color=config.TEXT_SOFT, linestyle=":", linewidth=1.2)
ax.text(median + 0.25, 0.52, f"median = {median:.2f} years", color=config.TEXT_MAIN, fontsize=9)
ax.axvline(config.TAU, color=config.TEXT_MAIN, linestyle="--", linewidth=1.2)
ax.text(config.TAU + 0.2, 0.9, "7-year horizon", color=config.TEXT_MAIN, fontsize=9)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Years since diagnosis")
ax.set_ylabel("Probability of being alive")
ax.set_title("Overall survival of training patients (Kaplan-Meier)")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82))
fig.tight_layout()
fig.savefig(config.FIGURES / "fig06_km_overall.png")
plt.close(fig)


# ===========================================================================
# 2. Groups of patients
# ===========================================================================
# (a) Bone-marrow blasts: <5%, 5-20%, >20% (common clinical cut-offs).
train["blast_group"] = pd.cut(train["BM_BLAST"], bins=[-0.1, 5, 20, 100],
                              labels=["<5%", "5-20%", ">20%"])

# (b) Gene mutations: True if the patient has at least one mutation in the gene.
train["TP53_mut"] = train["ID"].isin(mol.loc[mol["GENE"] == "TP53", "ID"])
train["SF3B1_mut"] = train["ID"].isin(mol.loc[mol["GENE"] == "SF3B1", "ID"])

# (c) Monosomy 7 = loss of a whole chromosome 7, written "-7" in the karyotype.
#     We lower-case the text, then look for "-7" right after the start, a comma
#     or a slash, and NOT followed by another digit (so "-17" or "-7x" style
#     typos are not counted as monosomy 7).
cyto = train["CYTOGENETICS"].str.lower()
train["monosomy7"] = cyto.str.contains(r"(?:^|[,/])\s*-7(?!\d)", regex=True, na=False)

# Each group comparison: (column, title, list of (value, label, colour)).
comparisons = [
    ("blast_group", "Bone-marrow blasts",
     [("<5%", "Blasts < 5%", config.COLOR_TRAIN),
      ("5-20%", "Blasts 5-20%", config.COLOR_TEST),
      (">20%", "Blasts > 20%", config.COLOR_GROUP3)]),
    ("TP53_mut", "TP53 mutation",
     [(False, "TP53 not mutated", config.COLOR_TRAIN),
      (True, "TP53 mutated", config.COLOR_TEST)]),
    ("SF3B1_mut", "SF3B1 mutation",
     [(False, "SF3B1 not mutated", config.COLOR_TRAIN),
      (True, "SF3B1 mutated", config.COLOR_TEST)]),
    ("monosomy7", "Monosomy 7 (-7) in karyotype",
     [(False, "No monosomy 7", config.COLOR_TRAIN),
      (True, "Monosomy 7", config.COLOR_TEST)]),
]

test_rows = []
fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
for ax, (col, title, groups) in zip(axes.ravel(), comparisons):
    sub = train.dropna(subset=[col]) if col == "blast_group" else train

    # Log-rank test: are the survival curves of the groups different?
    if len(groups) == 2:
        a = sub[sub[col] == groups[0][0]]
        b = sub[sub[col] == groups[1][0]]
        res = logrank_test(a["OS_YEARS"], b["OS_YEARS"], a["OS_STATUS"], b["OS_STATUS"])
    else:
        res = multivariate_logrank_test(sub["OS_YEARS"], sub[col].astype(str), sub["OS_STATUS"])
    stat, p = res.test_statistic, res.p_value

    for value, label, colour in groups:
        g = sub[sub[col] == value]
        k = KaplanMeierFitter().fit(g["OS_YEARS"], g["OS_STATUS"])
        s = k.survival_function_
        ax.step(s.index, s.iloc[:, 0], where="post", color=colour, linewidth=2,
                label=f"{label} (n={len(g)})")
        test_rows.append({
            "comparison": title, "group": label, "patients": len(g),
            "deaths": int(g["OS_STATUS"].sum()),
            "median_survival_years": round(k.median_survival_time_, 2),
            "S_at_3_years": round(float(k.survival_function_at_times(3).iloc[0]), 3),
            "logrank_chi2": round(stat, 1), "degrees_of_freedom": len(groups) - 1,
            "p_value": f"{p:.1e}",
        })

    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.set_title(f"{title}\nlog-rank chi2 = {stat:.1f}, {p_text}", fontsize=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Years since diagnosis")
    ax.set_ylabel("Probability of being alive")
    ax.legend(loc="upper right", fontsize=8.5)

fig.tight_layout()
fig.savefig(config.FIGURES / "fig07_km_groups.png")
plt.close(fig)

tests = pd.DataFrame(test_rows)
tests.to_csv(config.TABLES / "logrank_tests.csv", index=False)
print("\nGroup comparisons:")
print(tests.drop(columns=["degrees_of_freedom"]).to_string(index=False))


# ===========================================================================
# 3. One log-rank test BY HAND: TP53 mutated vs not mutated
# ===========================================================================
# At every distinct death time t_k:
#   n  = patients at risk (both groups),  n1 = at risk in group 1 (TP53 mutated)
#   d  = deaths (both groups),            d1 = deaths in group 1
#   If TP53 made no difference, group 1 would get its fair share of the deaths:
#       expected E1_k = d * n1 / n
#   Variance of d1 (hypergeometric):
#       V_k = n1 * n2 * d * (n - d) / (n^2 * (n - 1))
# Then   chi2 = (sum of (d1 - E1_k))^2 / (sum of V_k),  with 1 degree of freedom.
g1 = train["TP53_mut"].to_numpy()
death_times = np.sort(np.unique(T[E == 1]))
O1 = E1 = Vsum = 0.0
for t in death_times:
    at_risk = T >= t
    n = at_risk.sum()
    n1 = (at_risk & g1).sum()
    n2 = n - n1
    died = (T == t) & (E == 1)
    d = died.sum()
    d1 = (died & g1).sum()
    O1 += d1
    E1 += d * n1 / n
    if n > 1:
        Vsum += n1 * n2 * d * (n - d) / (n ** 2 * (n - 1))
chi2_hand = (O1 - E1) ** 2 / Vsum
p_hand = chi2.sf(chi2_hand, df=1)     # sf = 1 - CDF = upper-tail probability
lifelines_chi2 = tests.loc[tests["comparison"] == "TP53 mutation", "logrank_chi2"].iloc[0]

lines = [
    "Log-rank test by hand: TP53 mutated (group 1) vs not mutated",
    f"  Observed deaths in TP53 group,   O1 = {O1:.0f}",
    f"  Expected deaths if no difference, E1 = {E1:.2f}",
    f"  O1 - E1 = {O1 - E1:.2f}   (more deaths than expected -> higher risk)",
    f"  Sum of variances V = {Vsum:.2f}",
    f"  chi2 = (O1 - E1)^2 / V = ({O1 - E1:.2f})^2 / {Vsum:.2f} = {chi2_hand:.2f}",
    f"  p-value (1 degree of freedom) = {p_hand:.1e}",
    f"  lifelines chi2 (rounded) = {lifelines_chi2}",
]
print("\n" + "\n".join(lines))
with open(config.TABLES / "logrank_tp53_by_hand.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


# ===========================================================================
# 4. The censoring curve G(t) ("reverse Kaplan-Meier")
# ===========================================================================
# Swap the roles: treat "being censored" as the event (1 - E).
# G(t) = probability that a patient is still uncensored (still followed) at t.
# The challenge metric weights each death at time T_i by 1 / G(T_i)^2.
kmf_cens = KaplanMeierFitter()
kmf_cens.fit(T, event_observed=1 - E, label="Censoring curve G(t)")
print("\nCensoring curve G(t) and the weight 1/G(t)^2 used by the metric:")
for t in [1, 3, 5, 7]:
    g = float(kmf_cens.survival_function_at_times(t).iloc[0])
    print(f"  t = {t} y: G(t) = {g:.3f},  1/G(t)^2 = 1/{g:.3f}^2 = {1 / g ** 2:.2f}")

# ---- Figure 8: survival curve vs censoring curve --------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
s_surv = kmf.survival_function_
s_cens = kmf_cens.survival_function_
ax.step(s_surv.index, s_surv.iloc[:, 0], where="post", color=config.COLOR_DEATH,
        linewidth=2, label="S(t): probability still alive")
ax.step(s_cens.index, s_cens.iloc[:, 0], where="post", color=config.COLOR_CENSOR,
        linewidth=2, label="G(t): probability still followed (not censored)")
g7 = float(kmf_cens.survival_function_at_times(config.TAU).iloc[0])
ax.axvline(config.TAU, color=config.TEXT_MAIN, linestyle="--", linewidth=1.2)
ax.plot([config.TAU], [g7], marker="o", markersize=8, color=config.COLOR_CENSOR,
        markeredgecolor="white", markeredgewidth=2)
# Label placed below both curves so it does not cover them.
ax.text(config.TAU + 0.3, 0.02, f"G(7) = {g7:.3f}\nweight 1/G^2 = {1 / g7 ** 2:.1f}",
        color=config.TEXT_MAIN, fontsize=9, va="bottom")
ax.set_xlim(0, 15)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Years since diagnosis")
ax.set_ylabel("Probability")
ax.set_title("Survival curve and censoring curve")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(config.FIGURES / "fig08_censoring_curve.png")
plt.close(fig)

print("\nSaved 3 figures to figures/ and 4 tables to results/tables/.")
