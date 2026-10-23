"""
load_data.py
============
Small helper functions to read the five challenge files.
Later stages import these instead of repeating the reading code.
"""

import pandas as pd   # pandas = the standard Python library for tables

import config         # our settings file (folder paths and file names)


def check_files_exist():
    """
    Look for each of the 5 CSV files in data/raw/.
    Returns a list of the file names that are MISSING (empty list = all good).
    """
    missing = []
    for file_name in config.FILES.values():
        path = config.DATA_RAW / file_name
        if not path.exists():
            missing.append(file_name)
    return missing


def load_raw():
    """
    Read all five CSV files and return them in a dictionary, e.g.
        data["clinical_train"]  -> clinical table for training patients.

    Stops with a clear message if any file is missing.
    """
    missing = check_files_exist()
    if missing:
        # `raise` stops the program and prints our message.
        raise FileNotFoundError(
            "These files are missing from data/raw/: " + ", ".join(missing)
            + "\nDownload them from the challenge page and copy them there."
        )

    data = {}
    for key, file_name in config.FILES.items():
        # pd.read_csv reads a CSV file into a DataFrame (a table).
        data[key] = pd.read_csv(config.DATA_RAW / file_name)
    return data


def load_train_with_outcome():
    """
    Training clinical table joined with the outcome, keeping ONLY patients
    whose outcome is recorded (the 150 patients without one are dropped).
    Also returns the training mutation table.
    """
    data = load_raw()
    target = data["target_train"].dropna(subset=["OS_YEARS", "OS_STATUS"])
    # .merge joins the two tables on ID; "inner" keeps IDs present in both.
    train = data["clinical_train"].merge(target, on="ID", how="inner")
    # OS_STATUS is stored as 0.0 / 1.0; make it a whole number 0 / 1.
    train["OS_STATUS"] = train["OS_STATUS"].astype(int)
    return train, data["molecular_train"]
