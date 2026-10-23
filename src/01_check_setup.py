"""
01_check_setup.py  (Stage 0)
============================
Checks that your computer is ready for the project:
  1. the Python version,
  2. that every package in requirements.txt can be imported,
  3. that the 5 data files are in data/raw/ and have the expected sizes.

Run from the project's top folder:
    python src/01_check_setup.py
"""

import sys          # gives the Python version
import importlib    # lets us import a package from its name (a string)

import config
from load_data import check_files_exist, load_raw

print("=" * 60)
print("STAGE 0 - SETUP CHECK")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------
# sys.version_info holds (major, minor, micro), e.g. (3, 13, 1).
major, minor = sys.version_info[0], sys.version_info[1]
print(f"\nPython version: {major}.{minor}")
if (major, minor) < (3, 10):
    print("  WARNING: please use Python 3.10 or newer (you planned 3.13).")
else:
    print("  OK")

# ---------------------------------------------------------------------------
# 2. Packages
# ---------------------------------------------------------------------------
# Left: the name we import in Python. Right: the name used by pip.
packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "lifelines": "lifelines",
    "sksurv": "scikit-survival",
}

print("\nPackages:")
all_ok = True
for import_name, pip_name in packages.items():
    try:
        module = importlib.import_module(import_name)
        # Most packages store their version in __version__.
        version = getattr(module, "__version__", "unknown")
        print(f"  OK       {pip_name:<16} version {version}")
    except ImportError:
        print(f"  MISSING  {pip_name:<16} -> run: pip install -r requirements.txt")
        all_ok = False

# ---------------------------------------------------------------------------
# 3. Data files
# ---------------------------------------------------------------------------
print("\nData files in data/raw/:")
missing = check_files_exist()
if missing:
    for name in missing:
        print(f"  MISSING  {name}")
    all_ok = False
else:
    data = load_raw()
    # Expected (rows, columns) for each file, from the challenge description.
    expected = {
        "clinical_train": (3323, 9),
        "clinical_test": (1193, 9),
        "molecular_train": (10935, 11),
        "molecular_test": (3089, 11),
        "target_train": (3323, 3),
    }
    for key, table in data.items():
        status = "OK      " if table.shape == expected[key] else "CHECK   "
        print(f"  {status} {config.FILES[key]:<22} rows={table.shape[0]:>6}  columns={table.shape[1]}")

# ---------------------------------------------------------------------------
# 4. Make output folders and give the verdict
# ---------------------------------------------------------------------------
config.make_folders()
print("\nOutput folders are ready (figures/, results/tables/, submissions/).")
print("\nRESULT:", "everything is ready." if all_ok else "fix the items marked above, then run again.")
