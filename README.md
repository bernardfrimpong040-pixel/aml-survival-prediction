# Overall Survival Prediction in Myeloid Leukemia

Predicting how long patients with adult myeloid leukemia survive, from blood counts,
chromosome (cytogenetic) reports and gene mutations, using survival analysis:
Kaplan–Meier curves, Cox regression, and machine-learning survival models.

This is my entry to the **QRT × Institut Gustave Roussy data challenge** on
[Challenge Data (ENS Paris)](https://challengedata.ens.fr/challenges/162).

**Author:** Bernard Frimpong — M.S. Applied Mathematics (University of Akron, 2026)

---

## Results at a glance

IPCW C-index, τ = 7 years, on 15 centre-grouped validation folds (each fold holds out whole
hospitals), plus the public leaderboard score where submitted.

| Model | Cross-validation | Beats IPSS-R on | Public score |
|---|---|---|---|
| IPSS-R clinical score (no fitting) | 0.669 ± 0.029 | — | — |
| Benchmark-style Cox (4 variables) | 0.688 ± 0.029 | 13/15 folds | — |
| IPSS-M-style Cox (24 variables) | 0.716 ± 0.023 | 15/15 folds | 0.7374 |
| **Elastic-net Cox (84 features)** | **0.722 ± 0.021** | **15/15 folds** | **0.7543** |
| Random survival forest (84 features) | 0.719 ± 0.021 | 15/15 folds | — |
| XGBoost Cox (84 features, nested tuning) | 0.723 ± 0.021 | 15/15 folds | *pending* |

![Tree models vs elastic-net](figures/fig19_tree_cv.png)

---

## The problem in one paragraph

Each patient has an observed time `OS_YEARS` and a status `OS_STATUS`
(1 = died, 0 = still alive at last follow-up, called *censored*). For censored
patients we only know they survived *at least* `OS_YEARS`. The task is to give
every test patient a **risk score**: a higher score means we expect them to die
sooner. Only the **ranking** of the scores matters.

## How the challenge scores a submission

The metric is the **IPCW concordance index** (Uno's C-index), truncated at
**7 years**, computed with `sksurv.metrics.concordance_index_ipcw`.

- Look at every pair of patients where we know who died first.
- Count the share of those pairs where the model gave the higher risk to the patient who died first.
- 0.5 = random guessing, 1.0 = perfect ranking.
- "IPCW" (inverse probability of censoring weighting) gives each such pair the weight
  1 / G(T)², where G(T) is the probability of still being followed at the time T of
  the first death. Late deaths are rarely observed, so they count more.

## Data

| File | Rows | What it contains |
|---|---|---|
| `clinical_train.csv` | 3,323 | one row per patient: centre, blood counts, cytogenetics text |
| `clinical_test.csv` | 1,193 | same, for the test patients |
| `molecular_train.csv` | 10,935 | one row per somatic mutation (gene, effect, VAF, depth) |
| `molecular_test.csv` | 3,089 | same, for the test patients |
| `target_train.csv` | 3,323 | `OS_YEARS`, `OS_STATUS` |

The data are **not** included in this repository (Challenge Data terms of use).
To reproduce the work, register for the challenge and place the five files in `data/raw/`.

## Stage 1 — data audit: key findings

1. **The test set is one new hospital.** All 1,193 test patients come from centre
   `KYW`, which never appears in the 23 training centres. So I validate by holding out
   *whole centres* (grouped cross-validation), imitating the real test situation.
2. **The test patients are sicker.** More bone-marrow blasts above 20% (17.6% vs 4.2%),
   lower platelets and haemoglobin, and more than twice the rate of **TP53** mutations
   (25.7% vs 11.4%).
3. **150 training patients have no outcome** and are removed, leaving 3,173 patients:
   1,600 deaths (50.4%) and 1,573 censored (49.6%).
4. **Monocyte counts are missing for 74% of test patients** (18% in training).
5. **The cytogenetics text is written differently** in the two sets (`46,xy` vs `46,XY`
   or `Normal`).

## Stage 2 — Kaplan–Meier, log-rank tests and the metric by hand

- Kaplan–Meier computed by hand matches `lifelines` to within 2×10⁻¹⁴.
  Median overall survival: **3.15 years**.
- Log-rank tests (all p < 0.001): blasts (χ² = 272.1), TP53 (244.6, computed by hand too),
  monosomy 7 (139.5), SF3B1 (51.8, protective).
- The censoring curve G(t) falls to 0.198 at 7 years, so a death just before 7 years
  weighs 1/0.198² ≈ 25.6 in the metric.
- On a 6-patient toy example: Harrell's C = 7/9 = 0.778 and IPCW C = 0.615, both
  matching `scikit-survival` exactly.
- **Lesson:** TP53 has a huge log-rank χ² but a C-index of only 0.553, because about 75%
  of comparable pairs are tied on a 0/1 score. Significance is not ranking power.

## Stage 3 — feature engineering

- **91 features**: clinical (log blood counts), missing flags, cytogenetic (a forgiving
  karyotype parser tested on 25 tricky strings), molecular summaries, 45 gene indicators,
  and the published **IPSS-R** score.
- **Different gene panels:** only 64 of 124 training genes appear in the test file, so all
  mutation features use the 64 shared genes. Test splice-site mutations with no label were
  relabelled, and 9 test mutations below training's 2% VAF detection floor were dropped.
- **The test centre is higher risk:** 33% IPSS-R "very high" against 11% in training.
- **Centre-grouped CV:** 23 centres dealt into 5 folds in a random order, 3 different
  repeats = 15 validation folds. **Bug found and fixed:** `StratifiedGroupKFold(shuffle=True)`
  produced the *same* split for all 3 repeats with only 23 groups. The code now deals the
  centres itself and checks that the repeats differ.

![IPSS-R shift](figures/fig12_risk_shift.png)

## Stage 4 — Cox regression

- **Model:** h(t | x) = h₀(t) · exp(βᵀx). exp(βₖ) is a hazard ratio, and βᵀx is the
  submitted risk score. The elastic net adds the penalty α(ρ·Σ|βₖ| + (1−ρ)/2·Σβₖ²).
- **Honest evaluation:** imputation and scaling inside a pipeline, the penalty chosen by
  an **inner** centre-grouped CV (nested), and models compared **fold by fold**.
- **Findings:** the big gain comes from the feature set (IPSS-M-style vs benchmark:
  +0.028, 14/15 folds). Monocytes add nothing. TP53 multi-hit HR **3.21**. SF3B1 is
  protective alone but neutral once adjusted (confounding). NPM1 and platelets violate
  proportional hazards.

![Hazard ratios](figures/fig15_hazard_ratios.png)

**Leaderboard lessons.**
- Both public scores beat cross-validation. The test centre's wide spread of risk makes
  ranking easier, so the score reflects the centre as much as the model.
- The gap between the 84-feature and 24-feature models **tripled** at the test centre
  (+0.006 in CV, +0.017 public). The extra features describe the high-risk end that
  dominates KYW.

## Stage 5 — machine-learning survival models

- **Random survival forest:** 300 survival trees, splits chosen by the log-rank
  statistic, risk = predicted cumulative hazard.
- **XGBoost with a Cox loss:** small trees added one at a time, each fitted to the
  gradient of the Cox partial likelihood. Tree depth and number of trees were chosen by
  nested centre-grouped CV. When the first grid's smallest value was picked too often,
  the grid was widened and the results confirmed.

| Paired comparison (15 folds) | Mean difference | Folds won |
|---|---|---|
| XGBoost − elastic-net | +0.001 | 10/15 |
| Random forest − elastic-net | −0.003 | 4/15 |

**An honest null result:** the trees **tie** the elastic-net. The inner CV kept choosing
small models, a sign that little structure is left beyond what the engineered features
(IPSS-R score, cytogenetic categories, TP53 multi-hit) already encode. The three models
rank test patients very similarly (Spearman 0.96–0.97). All models rank early deaths
better than late ones, and XGBoost's only edge is at the 1-year horizon.

![C-index by horizon](figures/fig20_horizon.png)

## Project structure

```
aml-survival-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/            <- the 5 challenge CSV files go here (not on GitHub)
│   └── processed/      <- features, CV folds, out-of-fold predictions (not on GitHub)
├── src/
│   ├── config.py                 paths, 7-year horizon, random seed, colours, plot style
│   ├── load_data.py              functions that read the files
│   ├── features.py               karyotype parser, clinical/molecular features, IPSS-R
│   ├── cv_utils.py               metric, CV loop, nested elastic-net and XGBoost, RSF
│   ├── 01_check_setup.py         Stage 0: checks Python, packages and data files
│   ├── 02_data_audit.py          Stage 1: data audit
│   ├── 03_kaplan_meier.py        Stage 2: KM by hand, log-rank, censoring curve
│   ├── 04_cindex_by_hand.py      Stage 2: C-index and IPCW C-index by hand
│   ├── 05_build_features.py      Stage 3: features, IPSS-R, centre-grouped CV folds
│   ├── 06_cox_models.py          Stage 4: Cox models in CV + submission files
│   ├── 07_cox_interpretation.py  Stage 4: hazard ratios, PH check, coefficients
│   └── 08_tree_models.py         Stage 5: random survival forest, XGBoost Cox
├── figures/            <- PNG figures (used by the README and the paper)
├── results/tables/     <- CSV/TXT/JSON summaries
├── submissions/        <- files uploaded to Challenge Data (kept local)
└── paper/
    └── main.tex        <- LaTeX write-up, updated at every stage
```

## How to run (Windows, VS Code terminal)

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps scikit-survival
python src/01_check_setup.py
python src/02_data_audit.py
python src/03_kaplan_meier.py
python src/04_cindex_by_hand.py
python src/05_build_features.py
python src/06_cox_models.py
python src/07_cox_interpretation.py
python src/08_tree_models.py
```

`06` and `08` each take a few minutes (nested cross-validation).

**Why `--no-deps` for scikit-survival?** It lists `ecos` as a dependency, but `ecos`
has no ready-made Windows package for Python 3.13 and would need Microsoft C++ Build
Tools to compile. `ecos` is only used by one survival-SVM function that this project
does not use, so scikit-survival is installed without it.
`xgboost-cpu` is the CPU-only build of XGBoost, a much smaller download than `xgboost`.

## Stage plan and progress

| Stage | Content | Status |
|---|---|---|
| 0 | Setup: repository, packages, data check | ✅ done |
| 1 | Data audit: linking, outcome, missing values, centres, mutations, cytogenetics | ✅ done |
| 2 | Kaplan–Meier curves, log-rank tests, C-index and IPCW C-index computed by hand | ✅ done |
| 3 | Feature engineering, IPSS-R benchmark, centre-grouped CV folds | ✅ done |
| 4 | Cox regression: benchmark, IPSS-M-style, elastic-net (nested CV), PH check | ✅ done |
| 5 | Random survival forest and XGBoost Cox (nested CV), C-index by horizon | ✅ done |
| 6 | Ensemble, permutation importance, final model choice and submission | ⏳ next |
| 7 | Final write-up | ⬜ |

## Leaderboard submissions

| Date | File | Model | CV IPCW C-index | Public score |
|---|---|---|---|---|
| 23 Oct 2026 | `stage4_elasticnet_cox.csv` | Elastic-net Cox (84 features) | 0.722 ± 0.021 | **0.7543** |
| 23 Oct 2026 | `stage4_ipssm_cox.csv` | IPSS-M-style Cox (24 variables) | 0.716 ± 0.023 | 0.7374 |
| — | `stage5_xgboost_cox.csv` | XGBoost Cox (84 features) | 0.723 ± 0.021 | *pending* |

## References

- Kaplan E. L. and Meier P. (1958). Nonparametric estimation from incomplete
  observations. *Journal of the American Statistical Association* 53(282).
- Mantel N. (1966). Evaluation of survival data and two new rank order statistics
  arising in its consideration. *Cancer Chemotherapy Reports* 50(3).
- Cox D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical
  Society, Series B* 34(2).
- Harrell F. E. *et al.* (1982). Evaluating the yield of medical tests. *JAMA* 247(18).
- Grambsch P. M. and Therneau T. M. (1994). Proportional hazards tests and diagnostics
  based on weighted residuals. *Biometrika* 81(3).
- Friedman J. H. (2001). Greedy function approximation: a gradient boosting machine.
  *The Annals of Statistics* 29(5).
- Zou H. and Hastie T. (2005). Regularization and variable selection via the elastic net.
  *Journal of the Royal Statistical Society, Series B* 67(2).
- Ishwaran H. *et al.* (2008). Random survival forests. *The Annals of Applied
  Statistics* 2(3).
- Simon N. *et al.* (2011). Regularization paths for Cox's proportional hazards model via
  coordinate descent. *Journal of Statistical Software* 39(5).
- Uno H. *et al.* (2011). On the C-statistics for evaluating overall adequacy of risk
  prediction procedures with censored survival data. *Statistics in Medicine* 30(10).
- Greenberg P. L. *et al.* (2012). Revised International Prognostic Scoring System for
  myelodysplastic syndromes. *Blood* 120(12).
- Schanz J. *et al.* (2012). New comprehensive cytogenetic scoring system for primary
  myelodysplastic syndromes. *Journal of Clinical Oncology* 30(8).
- Chen T. and Guestrin C. (2016). XGBoost: a scalable tree boosting system.
  *Proceedings of KDD 2016*.
- Noren D. P. *et al.* (2016). A crowdsourcing approach to developing and assessing
  prediction algorithms for AML prognosis. *PLOS Computational Biology* 12(6): e1004890.
- Bernard E. *et al.* (2022). Molecular International Prognostic Scoring System for
  myelodysplastic syndromes. *NEJM Evidence* 1(7).
- Pölsterl S. (2020). scikit-survival: a library for time-to-event analysis built on
  top of scikit-learn. *Journal of Machine Learning Research* 21(212).
- Davidson-Pilon C. (2019). lifelines: survival analysis in Python. *Journal of Open
  Source Software* 4(40).
