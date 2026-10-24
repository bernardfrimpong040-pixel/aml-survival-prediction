"""
14_table1.py  (Stage 8, part 3)
===============================
"Table 1": the standard description of the patients in a clinical paper,
training cohort (3,173 patients with an outcome) next to the test centre
(1,193 patients).

Numbers are shown as  median (Q1-Q3)  for measurements and  n (%)  for
categories. Sex is not in the data; it is read from the karyotype
(XX = female, XY or loss of Y = male) and reported as "inferred".

Outputs:
  results/tables/table1.csv

Run from the project's top folder, AFTER 12_ipssm_benchmark.py (a few seconds):
    python src/14_table1.py
"""

import re
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

import config
import cv_utils as U
import ipssm as M
import features as F
from load_data import load_raw

config.make_folders()
train, test, groups, folds = U.load_stage3()
raw = load_raw()
cyto_raw = pd.concat([raw["clinical_train"], raw["clinical_test"]]).set_index("ID")["CYTOGENETICS"]


def inferred_sex(text):
    """'female', 'male' or 'unknown' from the sex-chromosome part of a karyotype."""
    if not isinstance(text, str):
        return "unknown"
    s = text.lower().replace(" ", "")
    m = re.match(r"^\d+(?:-\d+)?,([xy]+)", s)          # e.g. "46,xy" -> "xy"
    token = m.group(1) if m else ""
    if "y" in token or "-y" in s:
        return "male"
    if token.startswith("xx") or "-x" in s:
        return "female"
    return "unknown"


def med_iqr(x, digits=1):
    x = x.dropna()
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.{digits}f} ({q1:.{digits}f}-{q3:.{digits}f})"


def n_pct(mask, n_total):
    n = int(mask.sum())
    return f"{n} ({100 * n / n_total:.1f})"


def describe(t, is_train):
    n = len(t)
    out = {}
    out["Patients"] = str(n)
    out["Clinical centres"] = str(t["CENTER"].nunique())
    sex = t.index.to_series().map(lambda i: inferred_sex(cyto_raw.get(i)))
    out["Sex (inferred): female, n (%)"] = n_pct(sex == "female", n)
    out["Sex (inferred): male, n (%)"] = n_pct(sex == "male", n)
    out["Sex (inferred): unknown, n (%)"] = n_pct(sex == "unknown", n)
    if is_train:
        out["Deaths, n (%)"] = n_pct(t["OS_STATUS"] == 1, n)
        km = KaplanMeierFitter().fit(t["OS_YEARS"], t["OS_STATUS"])
        out["Median overall survival, years"] = f"{km.median_survival_time_:.2f}"
        # Median follow-up by the reverse Kaplan-Meier method (censoring as the event)
        rkm = KaplanMeierFitter().fit(t["OS_YEARS"], 1 - t["OS_STATUS"])
        out["Median follow-up (reverse KM), years"] = f"{rkm.median_survival_time_:.2f}"
    else:
        out["Deaths, n (%)"] = "not available"
        out["Median overall survival, years"] = "not available"
        out["Median follow-up (reverse KM), years"] = "not available"
    plt = np.expm1(t["log_PLT"])
    wbc = np.expm1(t["log_WBC"])
    anc = np.expm1(t["log_ANC"])
    out["Bone-marrow blasts, %"] = med_iqr(t["BM_BLAST"])
    out["Haemoglobin, g/dL"] = med_iqr(t["HB"])
    out["Platelets, G/L"] = med_iqr(plt, 0)
    out["White blood cells, G/L"] = med_iqr(wbc)
    out["Neutrophils (ANC), G/L"] = med_iqr(anc, 2)
    out["Any clinical value missing, n (%)"] = n_pct(t[["BM_BLAST", "HB", "log_PLT", "log_WBC", "log_ANC"]]
                                                     .isna().any(axis=1), n)
    out["Karyotype missing or unusable, n (%)"] = n_pct(t["cyto_missing"] == 1, n)
    out["Normal karyotype, n (%)"] = n_pct(t["cyto_normal"] == 1, n)
    out["Complex karyotype, n (%)"] = n_pct(t["cyto_complex"] == 1, n)
    for k, name in {0: "very good", 1: "good", 2: "intermediate", 3: "poor", 4: "very poor"}.items():
        out[f"IPSS-R cytogenetics {name}, n (%)"] = n_pct(t["cyto_ipssr_cat"] == k, n)
    out["Mutated genes per patient (shared panel)"] = med_iqr(t["mol_n_genes"], 0)
    out["No mutation detected, n (%)"] = n_pct(t["mol_n_mut"] == 0, n)
    out["TP53 multi-hit (IPSS-M definition), n (%)"] = n_pct(M.ipssm_inputs(t)["TP53multi"] == 1, n)
    for g in ["TET2", "ASXL1", "SF3B1", "SRSF2", "DNMT3A", "RUNX1", "TP53"]:
        out[f"{g} mutated, n (%)"] = n_pct(t["mut_" + g] == 1, n)
    for grp in F.IPSSR_GROUPS:
        out[f"IPSS-R risk {grp.lower()}, n (%)"] = n_pct(F.ipssr_risk_group(t["ipssr_score"]) == grp, n)
    score, _ = M.ipssm_score(t)
    cat = M.ipssm_category(score)
    for c in M.CATEGORIES:
        out[f"IPSS-M risk {c.lower()}, n (%)"] = n_pct(cat == c, n)
    return out


table1 = pd.DataFrame({"Training (23 centres)": describe(train, True),
                       "Test (centre KYW)": describe(test, False)})
table1.index.name = "Characteristic"
table1.to_csv(config.TABLES / "table1.csv")
with pd.option_context("display.max_rows", 100, "display.width", 140):
    print(table1.to_string())
print("\nDone: 1 table.")
