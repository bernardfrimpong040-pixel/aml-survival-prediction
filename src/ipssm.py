"""
ipssm.py  (Stage 8)
===================
The Molecular International Prognostic Scoring System (IPSS-M) for
myelodysplastic syndromes, computed from its PUBLISHED weights - no fitting.

Source of the weights and rules: the official IPSS-M calculator
(npm package "ipssm" 1.0.4, Papaemmanuil Lab, MIT licence), which implements
Bernard et al., NEJM Evidence 2022.

The score:
    IPSS-M = sum over variables of  coefficient x (value - reference mean) / ln 2
A missing input gets the reference mean (the calculator's "means" scenario),
i.e. it contributes 0. Only published constants are used, so nothing is
learned from our data and the score is a fair, leakage-free benchmark.

Differences from the official calculator (stated in the write-up):
  * TP53 loss of heterozygosity (LOH) is known here only through VAF > 0.55
    or a 17p loss in the karyotype; copy-neutral LOH data are not available,
    so one TP53 mutation without those is treated as single-hit.
  * Genes are treated as sequenced when absent from the mutation file
    (all 31 IPSS-M genes are on the panel shared by both centres).
"""

import numpy as np
import pandas as pd

# name: (coefficient, reference mean)  - copied from utils/betasRiskScore.js
BETAS = {
    "CYTOVEC": (0.287, 1.39),
    "BLAST5": (0.352, 0.922),
    "TRANSF_PLT100": (-0.222, 1.41),
    "HB1": (-0.171, 9.87),
    "SF3B1_alpha": (-0.0794, 0.186),
    "SF3B1_5q": (0.504, 0.0166),
    "ASXL1": (0.213, 0.252),
    "SRSF2": (0.239, 0.158),
    "DNMT3A": (0.221, 0.161),
    "RUNX1": (0.423, 0.126),
    "U2AF1": (0.247, 0.0866),
    "EZH2": (0.27, 0.0588),
    "CBL": (0.295, 0.0473),
    "NRAS": (0.417, 0.0362),
    "IDH2": (0.379, 0.0429),
    "KRAS": (0.202, 0.0271),
    "MLL_PTD": (0.798, 0.0247),
    "ETV6": (0.391, 0.0216),
    "NPM1": (0.43, 0.0112),
    "TP53multi": (1.18, 0.071),
    "FLT3": (0.798, 0.0108),
    "Nres2": (0.231, 0.388),
}
RESIDUAL_GENES = ["BCOR", "STAG2", "NF1", "CEBPA", "SETBP1", "PTPN11", "GATA2", "PRPF8",
                  "BCORL1", "ETNK1", "PPM1D", "PHF6", "WT1", "IDH1", "GNB1"]
MAIN_GENES = ["ASXL1", "SRSF2", "DNMT3A", "RUNX1", "U2AF1", "EZH2", "CBL", "NRAS",
              "IDH2", "KRAS", "ETV6", "NPM1"]
CATEGORIES = ["Very low", "Low", "Moderate low", "Moderate high", "High", "Very high"]
CUTPOINTS = [-np.inf, -1.5, -0.5, 0, 0.5, 1.5, np.inf]   # closed on the right


def ipssm_inputs(table):
    """
    Build the IPSS-M model variables from our Stage 3 feature table.
    Missing values are left as NaN here (they become the reference mean later).
    """
    t = table
    v = pd.DataFrame(index=t.index)
    mut = lambda g: t["mut_" + g].astype(float)

    # Clinical variables, transformed exactly as in the calculator.
    v["HB1"] = t["HB"]
    v["BLAST5"] = np.minimum(t["BM_BLAST"], 20) / 5                       # capped at 20%
    plt = np.expm1(t["log_PLT"])                                          # undo log(1 + x)
    v["TRANSF_PLT100"] = np.minimum(plt, 250) / 100                       # capped at 250
    v["CYTOVEC"] = t["cyto_ipssr_cat"]                                    # 0 = very good ... 4 = very poor

    # Karyotype flags needed for the SF3B1 and TP53 rules (missing karyotype -> 0).
    del5q = t["cyto_del5q"].fillna(0)
    del7 = ((t["cyto_minus7"].fillna(0) + t["cyto_del7q"].fillna(0)) > 0).astype(float)
    complex_ = t["cyto_complex"].fillna(0)
    del17 = t["cyto_17p"].fillna(0)

    # SF3B1 with isolated del(5q)
    v["SF3B1_5q"] = ((mut("SF3B1") == 1) & (del5q == 1) & (del7 == 0) & (complex_ == 0)).astype(float)
    # SF3B1 "alpha": SF3B1 without del(5q) and without co-mutations that cancel its benefit
    others = mut("SRSF2") + mut("STAG2") + mut("BCOR") + mut("BCORL1") + mut("RUNX1") + mut("NRAS")
    v["SF3B1_alpha"] = ((mut("SF3B1") == 1) & (v["SF3B1_5q"] == 0) & (others == 0)).astype(float)

    # TP53 multi-hit: 2+ mutations, or 1 mutation plus loss of the other copy
    loh = (t["tp53_max_vaf"] > 0.55) | (del17 == 1)
    v["TP53multi"] = ((t["tp53_n_mut"] >= 2) | ((t["tp53_n_mut"] == 1) & loh)).astype(float)

    for g in MAIN_GENES:
        v[g] = mut(g)
    v["MLL_PTD"] = t["mol_MLL_PTD"].astype(float)
    v["FLT3"] = mut("FLT3")                                               # ITD or TKD
    v["Nres2"] = np.minimum(sum(mut(g) for g in RESIDUAL_GENES), 2)       # capped at 2
    return v


def ipssm_score(table):
    """IPSS-M risk score (higher = higher risk) and its contributions."""
    v = ipssm_inputs(table)
    contrib = pd.DataFrame(index=v.index)
    for name, (coef, mean) in BETAS.items():
        value = v[name].fillna(mean)                       # missing -> reference mean
        contrib[name] = (value - mean) * coef / np.log(2)
    return contrib.sum(axis=1), contrib


def ipssm_category(score):
    return pd.cut(score, bins=CUTPOINTS, labels=CATEGORIES, right=True)
