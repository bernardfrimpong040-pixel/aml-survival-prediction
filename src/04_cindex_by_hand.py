"""
04_cindex_by_hand.py  (Stage 2, part 2)
=======================================
The challenge metric, computed by hand and then with scikit-survival.

Part A - a toy example with 6 patients:
  * list every comparable pair and say whether it is concordant,
  * Harrell's C-index by hand,
  * the censoring curve G(t) by hand,
  * the IPCW C-index (Uno's C, truncated at 7 years) by hand,
  * check both numbers against scikit-survival.

Part B - the real training data:
  * score patients with ONE variable at a time (no model fitted yet),
  * compute Harrell's C and the IPCW C-index for each,
  * this gives baseline numbers that every later model must beat.

Outputs:
  figures/fig09_toy_example.png
  figures/fig10_single_variable_cindex.png
  results/tables/toy_pairs.csv
  results/tables/cindex_single_variables.csv

Run from the project's top folder:
    python src/04_cindex_by_hand.py
"""

import numpy as np
import pandas as pd
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

import config
from load_data import load_train_with_outcome

plt = config.set_plot_style()
config.make_folders()

# ===========================================================================
# PART A - toy example
# ===========================================================================
print("=" * 64)
print("PART A - TOY EXAMPLE (6 patients)")
print("=" * 64)

toy = pd.DataFrame({
    "patient": ["A", "B", "C", "D", "E", "F"],
    "time":    [1.0, 2.0, 3.0, 4.0, 5.0, 8.0],   # observed time (years)
    "event":   [1,   0,   1,   0,   1,   1],     # 1 = died, 0 = censored
    "risk":    [0.90, 0.30, 0.60, 0.80, 0.05, 0.10],  # model's risk score
})
print(toy.to_string(index=False))

# ---------------------------------------------------------------------------
# A1. Comparable pairs and Harrell's C
# ---------------------------------------------------------------------------
# A pair (i, j) is comparable when patient i DIED and did so before patient j's
# observed time (T_i < T_j). Then we know for sure that i died first.
# It is concordant when the model agrees: risk_i > risk_j.
pairs = []
for _, pi in toy.iterrows():
    if pi["event"] != 1:
        continue                         # i must have died
    for _, pj in toy.iterrows():
        if pi["time"] < pj["time"]:      # i died before j's time
            pairs.append({
                "i": pi["patient"], "j": pj["patient"],
                "T_i": pi["time"], "T_j": pj["time"],
                "risk_i": pi["risk"], "risk_j": pj["risk"],
                "concordant": int(pi["risk"] > pj["risk"]),
            })
pairs = pd.DataFrame(pairs)

n_comparable = len(pairs)
n_concordant = pairs["concordant"].sum()
harrell_hand = n_concordant / n_comparable
print("\nComparable pairs (i died first):")
print(pairs.to_string(index=False))
print(f"\nHarrell's C by hand = concordant / comparable = {n_concordant} / {n_comparable} = {harrell_hand:.4f}")

# scikit-survival version: concordance_index_censored(event, time, risk)
harrell_sksurv = concordance_index_censored(toy["event"].astype(bool), toy["time"], toy["risk"])[0]
print(f"Harrell's C from scikit-survival             = {harrell_sksurv:.4f}")

# ---------------------------------------------------------------------------
# A2. The censoring curve G(t) by hand
# ---------------------------------------------------------------------------
# Same Kaplan-Meier recipe, but the "event" is now being CENSORED.
# Censorings happen at t = 2 (patient B) and t = 4 (patient D).
#   At t = 2: at risk = B, C, D, E, F = 5 patients, 1 censored  -> factor 1 - 1/5 = 0.8
#             G(2) = 1 x 0.8 = 0.8
#   At t = 4: at risk = D, E, F = 3 patients, 1 censored        -> factor 1 - 1/3 = 0.6667
#             G(4) = 0.8 x 0.6667 = 0.5333
# G stays flat between censoring times.
def G_toy(t):
    """Censoring curve of the toy example, worked out above."""
    g = 1.0
    if t >= 2:
        g *= 1 - 1 / 5
    if t >= 4:
        g *= 1 - 1 / 3
    return g


# ---------------------------------------------------------------------------
# A3. IPCW C-index (Uno's C) with tau = 7 by hand
# ---------------------------------------------------------------------------
# Every comparable pair whose first death T_i is before tau gets weight 1 / G(T_i)^2.
# Pairs with T_i >= tau are dropped (weight 0).
tau = config.TAU
pairs["G_Ti"] = pairs["T_i"].apply(G_toy)
pairs["weight"] = np.where(pairs["T_i"] < tau, 1 / pairs["G_Ti"] ** 2, 0.0)
pairs["weighted_concordant"] = pairs["weight"] * pairs["concordant"]
pairs.round(4).to_csv(config.TABLES / "toy_pairs.csv", index=False)

print("\nWeights of each death (only deaths before tau = 7 count):")
for name in ["A", "C", "E"]:
    t = toy.loc[toy["patient"] == name, "time"].iloc[0]
    g = G_toy(t)
    print(f"  patient {name}: T = {t:.0f}, G(T) = {g:.4f}, weight 1/G^2 = 1/{g:.4f}^2 = {1 / g ** 2:.4f}")

numerator = pairs["weighted_concordant"].sum()
denominator = pairs["weight"].sum()
uno_hand = numerator / denominator
print("\nNumerator   = sum of weights of concordant pairs")
for name in ["A", "C", "E"]:
    rows = pairs[pairs["i"] == name]
    print(f"  from {name}: {rows['concordant'].sum()} concordant x {rows['weight'].iloc[0]:.4f} "
          f"= {rows['weighted_concordant'].sum():.4f}")
print(f"  total numerator = {numerator:.4f}")
print("Denominator = sum of weights of all comparable pairs")
for name in ["A", "C", "E"]:
    rows = pairs[pairs["i"] == name]
    print(f"  from {name}: {len(rows)} pairs x {rows['weight'].iloc[0]:.4f} = {rows['weight'].sum():.4f}")
print(f"  total denominator = {denominator:.4f}")
print(f"\nIPCW C by hand = {numerator:.4f} / {denominator:.4f} = {uno_hand:.4f}")

# scikit-survival version. survival_train is used to estimate G(t); here we
# use the same 6 patients for both, exactly as in our hand calculation.
y_toy = Surv.from_arrays(event=toy["event"].astype(bool), time=toy["time"])
uno_sksurv = concordance_index_ipcw(y_toy, y_toy, toy["risk"], tau=tau)[0]
print(f"IPCW C from scikit-survival = {uno_sksurv:.4f}")
print("\nWhy is IPCW C lower? The two wrong pairs, (C, D) and (E, F), start from")
print("LATER deaths, which are rarer to observe and so carry bigger weights.")

# ---- Figure 9: the toy example drawn as timelines -------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for k, row in toy.iloc[::-1].reset_index(drop=True).iterrows():
    y = k
    ax.plot([0, row["time"]], [y, y], color=config.TEXT_SOFT, linewidth=2)
    if row["event"] == 1:
        ax.plot(row["time"], y, marker="X", markersize=11, color=config.COLOR_DEATH,
                markeredgecolor="white", markeredgewidth=1.5, linestyle="none")
    else:
        ax.plot(row["time"], y, marker="o", markersize=10, color="white",
                markeredgecolor=config.COLOR_CENSOR, markeredgewidth=2.5, linestyle="none")
    ax.text(row["time"] + 0.25, y, f"risk {row['risk']:.2f}", va="center", fontsize=9,
            color=config.TEXT_MAIN)
ax.set_yticks(range(len(toy)))
ax.set_yticklabels([f"Patient {p}" for p in toy["patient"][::-1]])
ax.axvline(tau, color=config.TEXT_MAIN, linestyle="--", linewidth=1.2)
ax.text(tau + 0.1, len(toy) - 0.45, "tau = 7", fontsize=9, color=config.TEXT_MAIN)
# Legend entries drawn with empty data, just to explain the two markers.
ax.plot([], [], marker="X", markersize=10, color=config.COLOR_DEATH, linestyle="none", label="died")
ax.plot([], [], marker="o", markersize=9, color="white", markeredgecolor=config.COLOR_CENSOR,
        markeredgewidth=2.5, linestyle="none", label="censored")
ax.legend(loc="upper right")         # empty corner, away from the labels
ax.set_xlim(0, 10)
ax.set_ylim(-0.6, len(toy) - 0.2)
ax.set_xlabel("Years since diagnosis")
ax.set_title(f"Toy example: Harrell C = {harrell_hand:.3f}, IPCW C = {uno_hand:.3f}")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig09_toy_example.png")
plt.close(fig)


# ===========================================================================
# PART B - single-variable risk scores on the real training data
# ===========================================================================
print("\n" + "=" * 64)
print("PART B - ONE VARIABLE AT A TIME (3,173 training patients)")
print("=" * 64)

train, mol = load_train_with_outcome()
y_train = Surv.from_arrays(event=train["OS_STATUS"].astype(bool), time=train["OS_YEARS"])

# Build simple scores. The DIRECTION comes from clinical knowledge, not from
# fitting: e.g. more blasts = higher risk, so score = +BM_BLAST; higher
# haemoglobin = healthier, so score = -HB. Missing values get the median
# (a neutral middle value). Nothing is learned from the outcome here.
cyto = train["CYTOGENETICS"].str.lower()
n_mut = mol.groupby("ID").size().reindex(train["ID"], fill_value=0).to_numpy()
rng = np.random.default_rng(config.SEED)

scores = {
    "Random numbers (reference)": rng.random(len(train)),
    "+ BM_BLAST": train["BM_BLAST"].fillna(train["BM_BLAST"].median()),
    "- HB": -train["HB"].fillna(train["HB"].median()),
    "- PLT": -train["PLT"].fillna(train["PLT"].median()),
    "+ TP53 mutated (0/1)": train["ID"].isin(mol.loc[mol["GENE"] == "TP53", "ID"]).astype(int),
    "+ Monosomy 7 (0/1)": cyto.str.contains(r"(?:^|[,/])\s*-7(?!\d)", regex=True, na=False).astype(int),
    "- SF3B1 mutated (0/1)": -train["ID"].isin(mol.loc[mol["GENE"] == "SF3B1", "ID"]).astype(int),
    "+ Number of mutations": n_mut,
    "- BM_BLAST (wrong direction)": -train["BM_BLAST"].fillna(train["BM_BLAST"].median()),
}

rows = []
for name, s in scores.items():
    s = np.asarray(s, dtype=float)
    harrell = concordance_index_censored(y_train["event"], y_train["time"], s)[0]
    uno = concordance_index_ipcw(y_train, y_train, s, tau=config.TAU)[0]
    rows.append({"score": name, "harrell_C": round(harrell, 4), "ipcw_C_tau7": round(uno, 4)})
single = pd.DataFrame(rows)
single.to_csv(config.TABLES / "cindex_single_variables.csv", index=False)
print(single.to_string(index=False))
print("\nNotes:")
print(" - Random numbers give about 0.5, as expected.")
print(" - Flipping the sign of a score gives about 1 - C (wrong direction).")
print(" - 0/1 scores create many TIED pairs, which count as 0.5, so their C stays nearer 0.5.")

# ---- Figure 10: IPCW C of each single-variable score ----------------------
plot_df = single[~single["score"].str.contains("wrong")].sort_values("ipcw_C_tau7")
fig, ax = plt.subplots(figsize=(7.2, 4.0))
colours = [config.TEXT_SOFT if "Random" in s else config.COLOR_TRAIN for s in plot_df["score"]]
ax.barh(plot_df["score"], plot_df["ipcw_C_tau7"] - 0.5, left=0.5, color=colours, height=0.65)
for k, v in enumerate(plot_df["ipcw_C_tau7"]):
    ax.text(max(v, 0.5) + 0.004, k, f"{v:.3f}", va="center", fontsize=8.5, color=config.TEXT_SOFT)
ax.axvline(0.5, color=config.TEXT_MAIN, linewidth=1.2)
ax.set_xlim(0.45, 0.75)
ax.set_xlabel("IPCW C-index, tau = 7 years (0.5 = random)")
ax.set_title("How well does each variable rank patients on its own?")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(config.FIGURES / "fig10_single_variable_cindex.png")
plt.close(fig)

print("\nSaved 2 figures to figures/ and 2 tables to results/tables/.")
