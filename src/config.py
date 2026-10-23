"""
config.py
=========
One place for every setting the project uses: folder locations, file names,
the random seed and the challenge's 7-year horizon.

Every other script does `import config` and reads from here, so if a path
ever changes we only edit this one file.
"""

# `Path` lets us build file paths that work on Windows AND Mac/Linux
# (Windows uses back-slashes "\", others use "/"; Path handles both).
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Folder locations
# ---------------------------------------------------------------------------
# __file__ is the location of THIS file (src/config.py).
# .resolve() turns it into a full absolute path.
# .parent is the folder it sits in (src/), and .parent again is the
# project's top folder (aml-survival-prediction/).
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"              # the 5 CSV files from the challenge
DATA_PROCESSED = ROOT / "data" / "processed"  # cleaned tables we create later
FIGURES = ROOT / "figures"                    # PNG figures (also used by the paper)
TABLES = ROOT / "results" / "tables"          # CSV/TXT result tables
SUBMISSIONS = ROOT / "submissions"            # files to upload to Challenge Data

# ---------------------------------------------------------------------------
# 2. File names exactly as downloaded from the challenge
# ---------------------------------------------------------------------------
FILES = {
    "clinical_train": "clinical_train.csv",    # one row per training patient
    "clinical_test": "clinical_test.csv",      # one row per test patient
    "molecular_train": "molecular_train.csv",  # one row per mutation (training)
    "molecular_test": "molecular_test.csv",    # one row per mutation (test)
    "target_train": "target_train.csv",        # OS_YEARS and OS_STATUS (training only)
}

# ---------------------------------------------------------------------------
# 3. Challenge constants
# ---------------------------------------------------------------------------
# The metric (IPCW C-index) only looks at the first 7 years of follow-up.
TAU = 7.0

# A fixed random seed makes every random step (CV splits, models) repeatable.
SEED = 42

# ---------------------------------------------------------------------------
# 4. Plot colours (one fixed meaning per colour, used in every figure)
# ---------------------------------------------------------------------------
COLOR_TRAIN = "#2a78d6"   # blue   = training data
COLOR_TEST = "#eb6834"    # orange = test data (centre KYW)
COLOR_DEATH = "#2a78d6"   # blue   = patient died (event)
COLOR_CENSOR = "#eb6834"  # orange = patient alive at last follow-up (censored)
TEXT_MAIN = "#0b0b0b"     # main text colour
TEXT_SOFT = "#52514e"     # axis labels / secondary text
GRID = "#e4e3df"          # light grid lines


def make_folders():
    """Create every output folder if it does not exist yet."""
    for folder in [DATA_PROCESSED, FIGURES, TABLES, SUBMISSIONS]:
        # parents=True also creates missing parent folders;
        # exist_ok=True means "no error if it is already there".
        folder.mkdir(parents=True, exist_ok=True)
