from typing import Union

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

CLASSIFIER = Union[RandomForestClassifier, XGBClassifier]
SITE_CONVERSION_CODEBOOK = {
    1: "Amsterdam",
    2: "London",
    3: "Oxford",
    4: "Stockholm",
    5: "Thessaloniki",
    6: "Bucharest",
    7: "Ljubljana",
    8: "Lisbon",
    9: "Brescia",
    10: "Geneva",
    11: "Lile",
    12: "Mannheim",
    13: "Barcelona",
    14: "Stavanger",
}
SITE_CONVERSION = {v: k for k, v in SITE_CONVERSION_CODEBOOK.items()}


def site_encoder(vec):
    enc = np.zeros((len(vec), len(SITE_CONVERSION)))
    for i, x in enumerate(vec):
        j = SITE_CONVERSION[x] - 1
        enc[i, j] = 1
    assert np.sum(enc) == len(vec)
    df = pd.DataFrame(enc, columns=SITE_CONVERSION.keys())
    return df


SEASON_CONVERSION_REV = {1: "summer", 2: "autumn", 3: "winter", 4: "spring"}
SEASON_CONVERSION = {v: k for k, v in SEASON_CONVERSION_REV.items()}


def season_encoder(vec):
    enc = np.zeros((len(vec), len(SEASON_CONVERSION)))
    for i, x in enumerate(vec):
        j = SEASON_CONVERSION[x] - 1
        enc[i, j] = 1
    assert np.sum(enc) == len(vec)
    df = pd.DataFrame(enc, columns=SEASON_CONVERSION.keys())
    return df


NEW_GROUPS = {
    1: "HC",
    2: "PreAD",
    3: "ProAD",
    4: "MildAD",
    0: "AD Spectrum",
    5: "ProAD+MildAD",
}
NEW_GROUPS_INVERSE = {v: float(k) for k, v in NEW_GROUPS.items()}
CONFOUNDERS = {"site", "age", "sex", "education_years"}
POT_CONFOUDNERS = {"sex", "age", "education_years", "bmi", "site", "season"}
SCALE_EXCLUDE = {
    "sex",
    "site",
    "season",
    "autumn",
    "spring",
    "summer",
    "winter",
    "Stavanger",
    "Amsterdam",
    "London",
    "Oxford",
    "Stockholm",
    "Thessaloniki",
    "Bucharest",
    "Ljubljana",
    "Lisbon",
    "Brescia",
    "Geneva",
    "Lille",
    "Mannheim",
    "Barcelona",
    "Lile",
}
COMPARISON_NAMES = [
    "HC\n&\nPreAD",
    "HC\n&\nProAD",
    "HC\n&\nMildAD",
    "PreAD\n&\nProAD",
    "ProAD\n&\nMildAD",
    "PreAD\n&\nMildAD",
    "HC\n&\AD Spectrum",
    "HC\n&\nProAD+MildAD",
]
ALTOIDA_COMPARISON_NAMES = [
    "HC\n&\nPreAD",
    "HC\n&\nProAD",
    "PreAD\n&\nProAD",
]
DEVICE_NAMES = {
    "altoida": "Altoida (CDS)",
    "altoidaDns": "Altoida (DNS)",
    "banking": "Banking",
    "iadl": "A-iADL",
    "fitbit": "Fitbit",
    "gait_dual": "Physilog (Dual)",
    "gait_tug": "Physilog (TUG)",
    "axivity": "Axivity",
    "mezurio": "Mezurio",
    "gs": "FDS",
    "altoida_gs": "Altoida (CDS) (+QS)",
    "banking_gs": "Banking (+QS)",
    "fitbitReduced_gs": "Fitbit (+QS)",
    "fitbit_gs": "Fitbit (+hourly) (+QS)",
    "gait_dual_gs": "Physilog (Dual) (+QS)",
    "gait_tug_gs": "Physilog (TUG) (+QS)",
    "axivity_gs": "Axivity (+QS)",
    "clinical_gs": "In-Clinic Assessment (Questionnaire) (+QS)",
    "mezurio_gs": "Mezurio (+QS)",
}
BASE_MODELS = ["lr_base", "rf_base", "xgb_base", "cart_base"]
BASE_PRETTY = ["LR (Base)", "RF (Base)", "XGBoost (Base)", "CART (Base)"]
COMPARISONS = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3), (0, -1), (0, 4)]
