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
def ipcw_c(y_train, y_valid, risk, tau=config.TAU):
    """Challenge metric. G(t) is estimated from y_train, as in the benchmark.
    tau = horizon in years (the challenge uses 7; Stage 5 also looks at 1-5)."""
    return concordance_index_ipcw(y_train, y_valid, np.asarray(risk, dtype=float), tau=tau)[0]


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


def _coxnet(l1_ratio, alphas=None, fit_baseline=False):
    """fit_baseline=True also stores the baseline survival curve S0(t), so the
    model can predict survival PROBABILITIES (needed for calibration, Stage 8)."""
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alphas, alpha_min_ratio=0.01,
                               n_alphas=30, max_iter=100000, fit_baseline_model=fit_baseline))


def fit_coxnet_nested(X, y, centres, l1_ratios=(0.1, 0.5, 1.0), n_inner=4, fit_baseline=False):
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
    model = _coxnet(l1, alphas=grid, fit_baseline=fit_baseline).fit(X, y)
    return model, alpha, l1, inner_score


# ---------------------------------------------------------------------------
# Tree models (added in Stage 5)
# ---------------------------------------------------------------------------
def make_rsf(seed=config.SEED):
    """
    Random survival forest: 300 survival trees, each grown on a bootstrap
    sample of patients, trying a random subset of features (sqrt of the number
    of features) at every split. A leaf must hold at least 15 patients.
    The forest's risk score is the predicted cumulative hazard summed over
    time: a bigger value means more expected deaths, i.e. higher risk.
    Trees do not need scaling, but this implementation needs no missing
    values, so the median imputer (learned inside the fold) stays.
    """
    from sksurv.ensemble import RandomSurvivalForest
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomSurvivalForest(n_estimators=300, min_samples_leaf=15,
                                              max_features="sqrt", n_jobs=-1, random_state=seed))


def xgb_label(table):
    """
    XGBoost's Cox loss reads the outcome from ONE number per patient:
      +time if the patient died, -time if censored.
    A censored time of exactly 0 would read as +0, so it becomes -0.000001.
    """
    t = table["OS_YEARS"].to_numpy(dtype=float)
    lab = np.where(table["OS_STATUS"].to_numpy() == 1, t, -t)
    lab[(table["OS_STATUS"].to_numpy() == 0) & (t == 0)] = -1e-6
    return lab


XGB_DEPTHS = (2, 3, 4)                 # how many splits deep each tree may go
XGB_TREES = (50, 100, 150, 200, 300, 450, 600)  # numbers of trees tried
# (50 and 150 were added after the first run: 100, then the smallest value,
#  was chosen in 7 of 15 folds. A choice at the edge of the grid means the
#  best value might lie beyond it, so the grid was widened.)


XGB_SEEDS = (42, 43, 44, 45, 46)       # the final model averages 5 random seeds (Stage 6)


def _xgb(depth, n_trees, seed=config.SEED):
    """
    Gradient boosting with the Cox partial likelihood as loss: trees are added
    one at a time, each correcting the errors of the sum of the previous ones.
      learning_rate 0.03 : each tree only takes a small step (slow, steady learning)
      subsample 0.8      : each tree sees a random 80% of patients
      colsample 0.5      : ... and a random half of the features
      min_child_weight 10, reg_lambda 1 : keep leaves from fitting a few patients
    XGBoost handles missing values itself (it learns which branch they go to).

    tree_method="exact" with ONE thread (changed in Stage 6): the faster
    "hist" method gave slightly different results on Windows and Linux
    (CV 0.7238 vs 0.7227). "exact" tries every split point, so it is slower
    but does not depend on the computer.
    """
    import xgboost as xgb
    return xgb.XGBRegressor(objective="survival:cox", learning_rate=0.03, n_estimators=n_trees,
                            max_depth=depth, subsample=0.8, colsample_bytree=0.5,
                            min_child_weight=10, reg_lambda=1.0, tree_method="exact",
                            n_jobs=1, random_state=seed)


def xgb_risk(models, X, n_trees):
    """
    Log relative risk using only the first n_trees trees.
    `models` may be one model or a list of models (then their scores are averaged).
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    return np.mean([m.predict(X, output_margin=True, iteration_range=(0, n_trees)) for m in models], axis=0)


def fit_xgb_nested(table, cols, n_inner=4, seeds=XGB_SEEDS):
    """
    Choose tree depth and number of trees by an inner centre-grouped CV,
    then refit on all of `table` once per seed (5 models, averaged when
    predicting: different random subsamples smooth each other out).
    For each depth, one model with the largest number of trees is fitted
    per inner split, and every smaller number of trees is scored from the
    same model (the first k trees), which is cheap.
    Returns: list of fitted models, chosen depth, chosen number of trees, inner score.
    """
    X, y = table[cols], survival_target(table)
    centres = table["CENTER"].to_numpy()
    inner = GroupKFold(n_splits=n_inner)
    best = (-np.inf, None, None)
    for depth in XGB_DEPTHS:
        scores = np.zeros(len(XGB_TREES))
        for tr, va in inner.split(X, groups=centres):
            model = _xgb(depth, max(XGB_TREES)).fit(X.iloc[tr], xgb_label(table.iloc[tr]))
            for k, n in enumerate(XGB_TREES):
                scores[k] += ipcw_c(y[tr], y[va], xgb_risk(model, X.iloc[va], n)) / n_inner
        k_best = int(np.argmax(scores))
        if scores[k_best] > best[0]:
            best = (scores[k_best], depth, XGB_TREES[k_best])
    inner_score, depth, n_trees = best
    models = [_xgb(depth, n_trees, seed=s).fit(X, xgb_label(table)) for s in seeds]
    return models, depth, n_trees, inner_score


# ---------------------------------------------------------------------------
# Cross-validation loop
# ---------------------------------------------------------------------------
def cross_validate(fit_predict, train, folds, n_repeats=3, n_splits=5, verbose_name=None):
    """
    fit_predict(train_part, valid_part) must return risk scores for valid_part.
    Returns:
      1. a table: repeat, fold, ipcw_C, harrell_C (15 rows),
      2. the out-of-fold predictions: one column per repeat ("repeat0",
         "repeat1", "repeat2"), one risk score per training patient. Each
         score comes from a model that never saw that patient's centre.
         (Stage 6 combines these saved predictions into ensembles.)
    """
    y_all = survival_target(train)
    rows = []
    oof = pd.DataFrame(np.nan, index=train.index, columns=[f"repeat{r}" for r in range(n_repeats)])
    for r in range(n_repeats):
        for k in range(n_splits):
            valid = (folds[f"repeat{r}"] == k).to_numpy()
            risk = fit_predict(train[~valid], train[valid])
            rows.append({"repeat": r, "fold": k,
                         "ipcw_C": ipcw_c(y_all[~valid], y_all[valid], risk),
                         "harrell_C": harrell_c(y_all[valid], risk)})
            oof.loc[valid, f"repeat{r}"] = np.asarray(risk, dtype=float)
        if verbose_name:
            print(f"  {verbose_name}: repeat {r} done")
    return pd.DataFrame(rows), oof


def save_oof(oof_by_model, path):
    """Save out-of-fold predictions in 'long' form: ID, model, repeat, risk."""
    parts = []
    for name, table in oof_by_model.items():
        long = table.reset_index().melt(id_vars="ID", var_name="repeat", value_name="risk")
        long.insert(1, "model", name)
        parts.append(long)
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)
