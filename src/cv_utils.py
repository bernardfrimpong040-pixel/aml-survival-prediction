"""
cv_utils.py  (Stage 4, reused in Stage 5)
=========================================
Helpers for honest model evaluation:
  * loading the Stage 3 feature tables and folds,
  * the challenge metric (IPCW C-index, tau = 7 years),
  * a cross-validation loop over the 15 centre-grouped folds,
  * an elastic-net Cox model whose penalty is chosen by an INNER
    centre-grouped cross-validation (nested cross-validation).
"""

import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

import config

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_stage3():
    """Return (train table, test table, feature groups dict, folds table)."""
    train = pd.read_csv(config.DATA_PROCESSED / "features_train.csv", index_col="ID")
    test = pd.read_csv(config.DATA_PROCESSED / "features_test.csv", index_col="ID")
    with open(config.DATA_PROCESSED / "feature_groups.json", encoding="utf-8") as f:
        groups = json.load(f)
    folds = pd.read_csv(config.DATA_PROCESSED / "cv_folds.csv", index_col="ID").loc[train.index]
    return train, test, groups, folds


def survival_target(table):
    """scikit-survival wants (event, time) packed into one structured array."""
    return Surv.from_arrays(event=table["OS_STATUS"].astype(bool), time=table["OS_YEARS"])


# ---------------------------------------------------------------------------
# IPSS-M style features (no fitting: a fixed list from the literature)
# ---------------------------------------------------------------------------
# The IPSS-M (Bernard et al., NEJM Evidence 2022) uses 16 "main effect" genes
# and counts how many of 15 "residual" genes are mutated (capped at 2).
IPSSM_MAIN = ["mut_ASXL1", "mut_CBL", "mut_DNMT3A", "mut_ETV6", "mut_EZH2", "mut_FLT3",
              "mut_IDH2", "mut_KRAS", "mol_MLL_PTD", "mut_NPM1", "mut_NRAS", "mut_RUNX1",
              "mut_SF3B1", "mut_SRSF2", "tp53_multihit", "mut_U2AF1"]
IPSSM_RESIDUAL = ["mut_BCOR", "mut_BCORL1", "mut_CEBPA", "mut_ETNK1", "mut_GATA2", "mut_GNB1",
                  "mut_IDH1", "mut_NF1", "mut_PHF6", "mut_PPM1D", "mut_PRPF8", "mut_PTPN11",
                  "mut_SETBP1", "mut_STAG2", "mut_WT1"]


def add_ipssm_residual(table):
    """Number of mutated residual genes, capped at 2 (0, 1 or 2)."""
    table = table.copy()
    table["ipssm_residual"] = table[IPSSM_RESIDUAL].sum(axis=1).clip(upper=2)
    return table


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------
def ipcw_c(y_train, y_valid, risk):
    """Challenge metric. G(t) is estimated from y_train, as in the benchmark."""
    return concordance_index_ipcw(y_train, y_valid, np.asarray(risk, dtype=float), tau=config.TAU)[0]


def harrell_c(y_valid, risk):
    return concordance_index_censored(y_valid["event"], y_valid["time"], np.asarray(risk, dtype=float))[0]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_cox(ridge=0.1):
    """
    Plain Cox model inside a pipeline:
      1. SimpleImputer(median): fills missing values with medians learned on
         the TRAINING data it is fitted on (so inside each fold - no leakage),
      2. StandardScaler: puts every column on the same scale (mean 0, sd 1),
      3. Cox model with a tiny ridge penalty for numerical stability.
    """
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         CoxPHSurvivalAnalysis(alpha=ridge))


def _coxnet(l1_ratio, alphas=None):
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alphas, alpha_min_ratio=0.01,
                               n_alphas=30, max_iter=100000))


def fit_coxnet_nested(X, y, centres, l1_ratios=(0.1, 0.5, 1.0), n_inner=4):
    """
    Elastic-net Cox with its two settings chosen by an inner grouped CV:
      l1_ratio : 1 = lasso (many coefficients exactly 0), small = close to ridge
      alpha    : overall penalty strength (bigger = simpler model)
    Steps, for each l1_ratio:
      a. fit once on all of X to get a grid of 30 alpha values,
      b. split X into n_inner parts BY CENTRE (GroupKFold), fit on 3 parts,
         score every alpha on the 4th, repeat for each part,
      c. average the inner scores.
    The (l1_ratio, alpha) with the best average inner score is refitted on all
    of X. The outer validation fold is never seen here.
    Returns: fitted pipeline, chosen alpha, chosen l1_ratio, inner score.
    """
    inner = GroupKFold(n_splits=n_inner)
    best = (-np.inf, None, None, None)
    for l1 in l1_ratios:
        grid = _coxnet(l1).fit(X, y)[-1].alphas_
        scores = np.zeros(len(grid))
        for tr, va in inner.split(X, groups=centres):
            model = _coxnet(l1, alphas=grid).fit(X.iloc[tr], y[tr])
            for k, a in enumerate(grid):
                risk = model.predict(X.iloc[va], alpha=a)
                scores[k] += ipcw_c(y[tr], y[va], risk) / n_inner
        k_best = int(np.argmax(scores))
        if scores[k_best] > best[0]:
            best = (scores[k_best], l1, grid, grid[k_best])
    inner_score, l1, grid, alpha = best
    model = _coxnet(l1, alphas=grid).fit(X, y)
    return model, alpha, l1, inner_score


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------
def cross_validate(fit_predict, train, folds, n_repeats=3, n_splits=5, verbose_name=None):
    """
    fit_predict(train_part, valid_part) must return risk scores for valid_part.
    Returns a table: repeat, fold, ipcw_C, harrell_C (15 rows), plus the
    out-of-fold predictions of repeat 0 (one risk score per training patient).
    """
    y_all = survival_target(train)
    rows = []
    oof = pd.Series(np.nan, index=train.index)
    for r in range(n_repeats):
        for k in range(n_splits):
            valid = (folds[f"repeat{r}"] == k).to_numpy()
            risk = fit_predict(train[~valid], train[valid])
            rows.append({"repeat": r, "fold": k,
                         "ipcw_C": ipcw_c(y_all[~valid], y_all[valid], risk),
                         "harrell_C": harrell_c(y_all[valid], risk)})
            if r == 0:
                oof[valid] = np.asarray(risk, dtype=float)
        if verbose_name:
            print(f"  {verbose_name}: repeat {r} done")
    return pd.DataFrame(rows), oof
