"""
features.py  (Stage 3)
======================
Functions that turn the raw tables into model-ready features.
Scripts from Stage 3 on import this file.

Three families of features:
  1. CLINICAL     - blood counts (log-transformed) and blasts, plus
                    "was missing" flags. Missing values are LEFT AS NaN here;
                    they are filled in later INSIDE each cross-validation fold.
  2. CYTOGENETIC  - read from the free-text karyotype: number of
                    abnormalities, specific lesions (-7, del(5q), +8, ...),
                    and the IPSS-R cytogenetic risk category.
  3. MOLECULAR    - gene mutation flags, mutation counts, VAF, TP53 details
                    and pathway counts, using ONLY genes sequenced in both
                    the training and the test centre.

Plus the IPSS-R clinical risk score, a published score that doctors use for
myelodysplastic syndromes (Greenberg et al., Blood 2012). It needs no model
fitting, so it is a fair "clinical benchmark".
"""

import re
import numpy as np
import pandas as pd


# ===========================================================================
# 1. CLINICAL FEATURES
# ===========================================================================
# Blood counts are very skewed (a few huge values), so we use log(1 + x).
# The "1 +" avoids log(0), because some counts are recorded as exactly 0.
LOG_COLUMNS = ["WBC", "ANC", "MONOCYTES", "PLT"]
RAW_COLUMNS = ["BM_BLAST", "HB"]          # already on a sensible scale


def clinical_features(clinical):
    """Return a table with one row per patient (index = ID)."""
    out = pd.DataFrame(index=clinical["ID"])
    for col in RAW_COLUMNS:
        out[col] = clinical[col].to_numpy()
    for col in LOG_COLUMNS:
        # np.log1p(x) = log(1 + x)
        out["log_" + col] = np.log1p(clinical[col].to_numpy())
    # Missing-value flags: 1 if the value was missing, else 0.
    for col in RAW_COLUMNS + LOG_COLUMNS:
        out["missing_" + col] = clinical[col].isna().astype(int).to_numpy()
    return out


# ===========================================================================
# 2. CYTOGENETIC FEATURES (parsing the karyotype text)
# ===========================================================================
# A karyotype in ISCN notation looks like:
#     46,xy,del(20)(q12)[2]/46,xy[18]
#     |  |  |              |  \_ a second cell line (clone): normal, 18 cells
#     |  |  |              \_ number of cells in this clone
#     |  |  \_ an abnormality: deletion on the long arm (q) of chromosome 20
#     |  \_ sex chromosomes
#     \_ number of chromosomes
# Clones are separated by "/". Abnormalities are separated by ",".
# Numerical changes: "-7" = loss of chromosome 7 (monosomy 7), "+8" = extra 8.
# Structural changes: del( ) deletion, t( ) translocation, inv( ) inversion,
# add( ) extra material, der( ) derivative chromosome, i( ) isochromosome,
# "mar" = an unidentified marker chromosome.
#
# Real-world text is messy (spaces, "{16}" instead of "[16]", the word
# "Normal", "complex", unbalanced brackets), so the parser is deliberately
# forgiving: it LOOKS FOR patterns instead of fully splitting the text.

# Free-text answers that mean "no usable karyotype".
NO_RESULT_WORDS = ["nomitosis", "notevaluated", "notexamined", "keinezell", "nucish", "failed"]

# Structural abnormality types, longest names first so "idic" is not read as "i".
STRUCTURAL = r"(idic|ider|del|der|add|ins|inv|dup|dic|trp|hsr|i|r|t)"

# Common harmless variant: inv(9)(p12q13) is a normal population variant,
# not a cancer abnormality, so it is NOT counted.
IGNORED_KEYS = {"inv(9)"}


def _clean_text(text):
    """Lower-case, remove spaces, and fix the most common typing variants."""
    s = str(text).lower().strip()
    s = s.replace(" ", "")
    s = s.replace("{", "[").replace("}", "]")      # {16} -> [16]
    s = re.sub(r"\)\.(?=[a-z+\-])", "),", s)        # "...).add(" -> "...),add("
    return s


def parse_karyotype(text):
    """
    Read one karyotype string and return a dictionary of features.
    Returns None when the karyotype is missing or not usable.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    s = _clean_text(text)
    if s == "" or any(word in s for word in NO_RESULT_WORDS):
        return None

    # --- Short free-text descriptions (mostly in the test file) -------------
    forced_count = None
    if s.startswith("normal"):
        s = "46,xx"                                   # a normal karyotype
    elif "complex" in s or ">3" in s:
        # "complex" means 3 or more abnormalities; ">3" means more than 3.
        forced_count = 4 if ">3" in s else 3
    elif s.startswith("abnormal"):
        forced_count = 1                              # abnormal, details unknown
    elif not s[0].isdigit():
        # Short-hand like "del7", "tris8", "dely", "+8(fish)", "del(20)".
        s = (s.replace("tris8", "+8").replace("dely", "-y")
              .replace("del7", "del(7)(q)").replace("del20", "del(20)(q)"))
        s = "," + s                                   # so "+8" is found as a token

    # --- Remove cell counts like [20], [3/20], [cp3] -----------------------
    no_counts = re.sub(r"\[[^\]]*\]", "", s)

    # --- Numerical changes: +8, -7, -y, +x ----------------------------------
    # A sign at the start of a token (after start, comma, ")" or "/"),
    # followed by 1-2 digits or x/y, and NOT followed by another digit,
    # by p/q (that would be an arm, e.g. "7q-"), by a letter (e.g. "+mar"),
    # or by "-digit" (e.g. "+1-3mar" = 1 to 3 markers).
    numeric = set(re.findall(r"(?:^|(?<=[,)/]))([+\-](?:\d{1,2}|x|y))(?![\dpqa-z])(?!-\d)", no_counts))

    # --- Structural changes: del(5), t(3;3), inv(3), der(7) ... -------------
    # Skip a type that directly follows ")" as in "der(1)t(1;7)": the t(...)
    # there only describes how der(1) was formed, it is not a new abnormality.
    structural = set()
    for kind, inside in re.findall(STRUCTURAL + r"\(([^()]*)\)", no_counts):
        structural.add(f"{kind}({inside})")
    for kind, inside in re.findall(r"(?<=\))" + STRUCTURAL + r"\(([^()]*)\)", no_counts):
        structural.discard(f"{kind}({inside})")
    # Old notation such as "5q-" or "7q-" (deletion of a long arm).
    for chrom, arm in re.findall(r"(?<![\d(;])(\d{1,2})([pq])-", no_counts):
        structural.add(f"del({chrom}{arm})")
    structural -= IGNORED_KEYS

    # --- Marker chromosomes: +mar, +2mar, +1-3mar ---------------------------
    has_marker = bool(re.search(r"\+\d*(?:-\d+)?mar", no_counts))

    n_abn = len(numeric) + len(structural) + int(has_marker)
    if forced_count is not None:
        n_abn = max(n_abn, forced_count)

    # --- Specific lesions known to matter for prognosis ---------------------
    t = no_counts
    f = {}
    f["cyto_n_abn"] = n_abn
    f["cyto_normal"] = int(n_abn == 0)
    f["cyto_minus7"] = int("-7" in numeric)
    f["cyto_del7q"] = int(bool(re.search(r"del\(7\)\(q|del\(7q|(?<![\d(;])7q-", t)))
    f["cyto_minus5"] = int("-5" in numeric)
    f["cyto_del5q"] = int(bool(re.search(r"del\(5\)\(q|del\(5q|(?<![\d(;])5q-", t)))
    f["cyto_plus8"] = int("+8" in numeric)
    f["cyto_del20q"] = int(bool(re.search(r"del\(20\)", t)))
    f["cyto_minusY"] = int("-y" in numeric)
    f["cyto_del11q"] = int(bool(re.search(r"del\(11\)\(q|del\(11q", t)))
    f["cyto_del12p"] = int(bool(re.search(r"del\(12\)\(p|del\(12p", t)))
    f["cyto_17p"] = int("-17" in numeric or bool(re.search(r"del\(17\)\(p|del\(17p|i\(17\)\(q|i\(17q", t)))
    f["cyto_3q"] = int(bool(re.search(r"inv\(3\)|t\(3;3\)|del\(3\)\(q", t)))
    f["cyto_plus19"] = int("+19" in numeric)
    # Autosomal monosomies = losses of chromosomes other than X and Y.
    n_monosomies = sum(1 for x in numeric if x.startswith("-") and x not in ("-x", "-y"))
    f["cyto_monosomal"] = int(n_monosomies >= 2 or (n_monosomies >= 1 and len(structural) >= 1))
    f["cyto_complex"] = int(n_abn >= 3)
    f["cyto_ipssr_cat"] = ipssr_cyto_category(f)
    return f


def ipssr_cyto_category(f):
    """
    IPSS-R cytogenetic risk category (Schanz et al., J Clin Oncol 2012):
        0 = very good : -Y alone, del(11q) alone
        1 = good      : normal, del(5q)/del(12p)/del(20q) alone, double incl. del(5q)
        2 = intermediate : del(7q) alone, +8, +19, i(17q), other single or double
        3 = poor      : -7, inv(3)/t(3q)/del(3q), double incl. -7/del(7q), 3 abnormalities
        4 = very poor : more than 3 abnormalities
    """
    n = f["cyto_n_abn"]
    if n == 0:
        return 1
    if n > 3:
        return 4
    if n == 3:
        return 3
    if f["cyto_minus7"] or f["cyto_3q"]:
        return 3
    if n == 2 and f["cyto_del7q"]:
        return 3
    if n == 1:
        if f["cyto_minusY"] or f["cyto_del11q"]:
            return 0
        if f["cyto_del5q"] or f["cyto_del12p"] or f["cyto_del20q"]:
            return 1
        return 2
    # n == 2
    if f["cyto_del5q"]:
        return 1
    return 2


CYTO_COLUMNS = ["cyto_n_abn", "cyto_normal", "cyto_minus7", "cyto_del7q", "cyto_minus5",
                "cyto_del5q", "cyto_plus8", "cyto_del20q", "cyto_minusY", "cyto_del11q",
                "cyto_del12p", "cyto_17p", "cyto_3q", "cyto_plus19", "cyto_monosomal",
                "cyto_complex", "cyto_ipssr_cat"]


def cytogenetic_features(clinical):
    """Parse every karyotype; missing karyotypes get NaN plus cyto_missing = 1."""
    rows = []
    for text in clinical["CYTOGENETICS"]:
        parsed = parse_karyotype(text)
        if parsed is None:
            row = {c: np.nan for c in CYTO_COLUMNS}
            row["cyto_missing"] = 1
        else:
            row = parsed
            row["cyto_missing"] = 0
        rows.append(row)
    return pd.DataFrame(rows, index=clinical["ID"])[CYTO_COLUMNS + ["cyto_missing"]]


# ===========================================================================
# 3. MOLECULAR FEATURES
# ===========================================================================
# Biological pathways. A gene only counts if it is in the shared gene panel.
PATHWAYS = {
    "path_splicing": ["SF3B1", "SRSF2", "U2AF1", "U2AF2", "ZRSR2", "PRPF8", "LUC7L2"],
    "path_dna_methylation": ["TET2", "DNMT3A", "IDH1", "IDH2"],
    "path_chromatin": ["ASXL1", "ASXL2", "EZH2", "BCOR", "BCORL1", "KDM6A", "ARID2",
                       "KMT2C", "KMT2D", "EP300", "CREBBP", "SUZ12", "EED", "PHF6"],
    "path_signalling": ["NRAS", "KRAS", "CBL", "PTPN11", "NF1", "FLT3", "JAK2", "MPL",
                        "KIT", "SH2B3", "CSF3R", "BRAF", "GNB1", "GNAS", "CALR"],
    "path_cohesin": ["STAG2", "RAD21", "SMC1A", "SMC3"],
    "path_transcription": ["RUNX1", "CEBPA", "ETV6", "GATA2", "WT1", "NPM1", "CUX1"],
}
TRUNCATING = {"frameshift_variant", "stop_gained", "splice_site_variant"}


def harmonise_mutations(molecular):
    """
    Make the training and test mutation tables use the same labels.
    In the test file, splice-site mutations have an empty EFFECT (written as
    c.307+1G>A etc.), while training labels them "splice_site_variant".
    """
    m = molecular.copy()
    m["EFFECT"] = m["EFFECT"].fillna("splice_site_variant")
    return m


def molecular_features(molecular, ids, gene_list, panel_genes):
    """
    molecular   : mutation table (one row per mutation)
    ids         : the patient IDs, in order (patients with no mutation get 0s)
    gene_list   : genes that get their own 0/1 column
    panel_genes : genes sequenced in BOTH centres; all counts use only these
    """
    m = harmonise_mutations(molecular)
    m = m[m["GENE"].isin(panel_genes)]
    out = pd.DataFrame(index=pd.Index(ids, name="ID"))

    # Counts per patient. reindex(..., fill_value=0) gives 0 to patients with
    # no mutation row: "no mutation found" is information, not missing data.
    grouped = m.groupby("ID")
    out["mol_n_mut"] = grouped.size().reindex(ids, fill_value=0).to_numpy()
    out["mol_n_genes"] = grouped["GENE"].nunique().reindex(ids, fill_value=0).to_numpy()
    out["mol_max_vaf"] = grouped["VAF"].max().reindex(ids).fillna(0).to_numpy()
    trunc = m[m["EFFECT"].isin(TRUNCATING)].groupby("ID").size()
    out["mol_n_truncating"] = trunc.reindex(ids, fill_value=0).to_numpy()

    # TP53 details: number of TP53 mutations and the highest TP53 VAF.
    tp = m[m["GENE"] == "TP53"].groupby("ID")
    out["tp53_n_mut"] = tp.size().reindex(ids, fill_value=0).to_numpy()
    out["tp53_max_vaf"] = tp["VAF"].max().reindex(ids).fillna(0).to_numpy()

    # Special mutation types.
    itd = m[(m["GENE"] == "FLT3") & (m["EFFECT"] == "ITD")]["ID"].unique()
    ptd = m[(m["GENE"] == "MLL") & (m["EFFECT"] == "PTD")]["ID"].unique()
    out["mol_FLT3_ITD"] = out.index.isin(itd).astype(int)
    out["mol_MLL_PTD"] = out.index.isin(ptd).astype(int)

    # Pathway counts: how many genes of each pathway are mutated.
    for name, genes in PATHWAYS.items():
        sub = m[m["GENE"].isin(genes)].groupby("ID")["GENE"].nunique()
        out[name] = sub.reindex(ids, fill_value=0).to_numpy()

    # One 0/1 column per selected gene.
    for gene in gene_list:
        mutated = m.loc[m["GENE"] == gene, "ID"].unique()
        out["mut_" + gene] = out.index.isin(mutated).astype(int)
    return out


# ===========================================================================
# 4. IPSS-R CLINICAL RISK SCORE (no fitting needed)
# ===========================================================================
def ipssr_points(clinical, cyto_cat):
    """
    Points for each IPSS-R component (Greenberg et al., Blood 2012).
    Returns a table of component points; NaN where the input is missing.
    """
    pts = pd.DataFrame(index=clinical["ID"])
    pts["cyto"] = cyto_cat.to_numpy()                      # 0 to 4 points

    blast = clinical["BM_BLAST"].to_numpy()                # <=2: 0, (2,5): 1, [5,10]: 2, >10: 3
    pts["blast"] = np.select([blast <= 2, blast < 5, blast <= 10, blast > 10],
                             [0, 1, 2, 3], default=np.nan)

    hb = clinical["HB"].to_numpy()                         # >=10: 0, [8,10): 1, <8: 1.5
    pts["hb"] = np.select([hb >= 10, hb >= 8, hb < 8], [0, 1, 1.5], default=np.nan)

    plt_ = clinical["PLT"].to_numpy()                      # >=100: 0, [50,100): 0.5, <50: 1
    pts["plt"] = np.select([plt_ >= 100, plt_ >= 50, plt_ < 50], [0, 0.5, 1], default=np.nan)

    anc = clinical["ANC"].to_numpy()                       # >=0.8: 0, <0.8: 0.5
    pts["anc"] = np.select([anc >= 0.8, anc < 0.8], [0, 0.5], default=np.nan)
    return pts


IPSSR_GROUPS = ["Very low", "Low", "Intermediate", "High", "Very high"]


def ipssr_risk_group(score):
    """Very low <=1.5, Low (1.5,3], Intermediate (3,4.5], High (4.5,6], Very high >6."""
    return pd.cut(score, bins=[-0.1, 1.5, 3, 4.5, 6, 10.1], labels=IPSSR_GROUPS)
