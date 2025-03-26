from collections import Counter
from copy import copy
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer

from src.features import (
    ALTODIA_DNS,
    MALTOIDA,
    MAXIVITY,
    MBANKING,
    MEZURIO_FT,
    MFITBIT,
    MFITCOMPLETE,
    MGAIT_DUAL,
    MGAIT_TUG,
)
from src.definitions import (
    CONFOUNDERS,
    NEW_GROUPS,
    POT_CONFOUDNERS,
    site_encoder,
    SITE_CONVERSION,
    NEW_GROUPS_INVERSE,
    SEASON_CONVERSION,
    season_encoder,
)


app = typer.Typer()
np.random.seed(42)


def prepare_df(
    dataframe: pd.DataFrame,
    variables: List[str],
    targets: List[str],
    ref: pd.DataFrame,
    extra: List[str] = ["group", "record_id", "mmse_total"] + list(CONFOUNDERS),
    merge_var: str = None,
) -> pd.DataFrame:
    data = dataframe.copy()
    if merge_var is not None:
        for var in ["group"]:
            try:
                data.drop(var, axis=1, inplace=True)
            except KeyError:
                pass
        if merge_var == "record_id":
            data = pd.merge(
                data,
                ref.loc[:, targets + extra],
                on="record_id",
            )
        else:
            data = pd.merge(
                data,
                ref.loc[:, targets + extra],
                left_on=merge_var,
                right_on="record_id",
            )
    colnames = data.columns
    data.columns = [f"xvar_{x}" if x in variables else x for x in colnames]

    res = data.loc[:, [f"xvar_{var}" for var in variables] + targets + extra]
    if "site" in list(CONFOUNDERS):
        site_enc = site_encoder(res.pop("site").tolist())
        site_enc.index = res.index
        res = pd.concat([res, site_enc], axis=1)

    if "season" in list(CONFOUNDERS):
        season_enc = season_encoder(res.pop("season").tolist())
        season_enc.index = res.index
        res = pd.concat([res, season_enc], axis=1)

    return res


def clean_dataframe(
    data: pd.DataFrame,
    target_columns: List[Any] = [],
    row_threshold: float = 0.75,
    column_threshold: float = 0.9,
    keepna: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    data.set_index("record_id", inplace=True)

    # cleaning
    if not keepna:
        print(f"Before: {data.shape}")
        print(
            f"Rows with many na values: {np.sum(data.notna().sum(1) / data.shape[1] < row_threshold)})"
        )
        data = data.loc[data.notna().sum(1) / data.shape[1] > row_threshold, :]
        print(f"After step 1: {data.shape}")
        print(
            f"Cols with many na values: {data.columns[data.notna().sum() / data.shape[0] < column_threshold]})"
        )
        data = data.loc[:, data.notna().sum() / data.shape[0] > column_threshold]
        print(f"After step 2: {data.shape}")

        # check specifically for the predictors
        print(data.columns)
        predictor_subset = data.loc[
            :,
            [f"xvar_{x}" for x in target_columns if f"xvar_{x}" in data.columns],
        ]
        # check how many nan values per row
        nan_ratio = predictor_subset.isna().sum(1) / predictor_subset.shape[1]
        # exclude rows with more than 50% nan values
        data = data.loc[nan_ratio < 0.5, :]
        print(f"After step 3: {data.shape}")

        # check nan ratio per row
        nan_ratio = max(data.isna().sum(1) / data.shape[1])
        if nan_ratio > 0.4:
            raise Exception(f"At least one row has {nan_ratio * 100}% nan values")

    else:
        # remove at least columns which are never filled
        print(f"Before: {data.shape}")
        col_boolean = data.notna().sum() / data.shape[0]
        print(f"Cols with many na values: {data.columns[col_boolean < 0.5]})")
        data = data.loc[:, col_boolean > 0.5]
        print(f"After: {data.shape}")

    # separation
    x_idx = [
        i
        for i, x in enumerate(data.columns)
        if x.startswith("xvar")
        or x in POT_CONFOUDNERS
        or x in SITE_CONVERSION.keys()
        or x in SEASON_CONVERSION.keys()
    ]
    x_data = data.iloc[:, x_idx]
    x_data.columns = list(map(lambda x: x.replace("xvar_", ""), x_data.columns))

    if target_columns is not None and len(target_columns) > 0:
        target_idx = [i for i, x in enumerate(data.columns) if x in target_columns]
        target_data = data.iloc[:, target_idx]
    else:
        target_data = None

    group = data.group

    return x_data, group, target_data


def average_z_scores(questionnaires):
    # create a list to store the z-scores for each questionnaire
    z_scores_list = []

    # iterate through each questionnaire
    for questionnaire in questionnaires:
        # convert questionnaire scores to a numpy array
        scores = np.array(questionnaire)

        # calculate mean and standard deviation while ignoring nan values
        mean = np.nanmean(scores)
        std_dev = np.nanstd(scores)

        # calculate z-scores and add to list
        z_scores = (scores - mean) / std_dev
        z_scores_list.append(z_scores)

    # Convert list of lists to 2D numpy array
    z_scores_array = np.array(z_scores_list)

    # Calculate mean across questionnaires (column-wise) while ignoring nan values
    average_z_scores = np.nanmean(z_scores_array, axis=0)

    return average_z_scores


@app.command()
def process_data(
    input_file: Path,
    output_xdata: Path,
    output_group: Path,
    output_features: Path,
    input_type: str = typer.Argument(
        "...", help="altoida|data|gait_(dual/tug)|banking|fitbit(_reduced)"
    ),
    output_targets: Path = typer.Option(None),
    target_col: str = None,
    addition: Optional[Path] = None,
    keepna: bool = False,
):
    if str(input_file).endswith("csv"):
        merged_data = pd.read_csv(input_file, index_col="record_id")
    else:
        merged_data = pd.read_excel(input_file, index_col="record_id")
        merged_data.rename({"AIADL_SV_tscore": "tscore"}, axis=1, inplace=True)
    target_col = (
        [target_col.replace("-", "_")] if isinstance(target_col, str) else target_col
    )
    predictor_cols: List[str] | None = None
    confounders = list(CONFOUNDERS)

    if input_type == "altoida":
        predictor_cols = MALTOIDA
    elif input_type == "altoidaDns":
        predictor_cols = ALTODIA_DNS
    elif input_type == "banking":
        predictor_cols = MBANKING
    elif input_type == "axivity":
        predictor_cols = MAXIVITY
        confounders.append("bmi")
        confounders.append("season")
    elif input_type == "gait_dual":
        predictor_cols = MGAIT_DUAL
        confounders.append("bmi")
    elif input_type == "gait_tug":
        predictor_cols = MGAIT_TUG
        confounders.append("bmi")
    elif input_type == "fitbit":
        confounders.append("bmi")
        confounders.append("season")
        predictor_cols = MFITCOMPLETE
    elif input_type == "fitbitReduced":
        predictor_cols = MFITBIT
        confounders.append("bmi")
        confounders.append("season")
    elif input_type == "mezurio":
        predictor_cols = MEZURIO_FT
    else:
        raise Exception(f"{input_type} not found")

    xdata, ydata, groups = prepare_data(
        merged_data,
        predictor_cols,
        target_col,
        confounder=confounders,
        keepna=keepna,
    )

    # Note: Altoida data should not be used for mildAD participants and hence should be removed
    if "altoida" in input_type:
        group_indicator = NEW_GROUPS_INVERSE.get("MildAD")
        xdata.loc[groups == group_indicator, predictor_cols] = np.nan
        print(
            f"Set predictors to NA for group {group_indicator} in dataset with input type {input_type}"
        )
        print(
            f"Shape: {xdata.shape=}, {groups.shape=}, {ydata.shape if ydata is not None else None}"
        )

    xdata.to_csv(output_xdata)
    groups.to_csv(output_group)
    if ydata is not None:
        ydata.to_csv(output_targets)

    with output_features.open("w+") as f:
        for col in list(xdata.columns):
            f.write(f"{col}\n")

    print(Counter(map(lambda x: NEW_GROUPS[x], groups)))


def prepare_data(
    data: pd.DataFrame,
    preds: List[str],
    targets: Optional[List[str]],
    confounder: List[str],
    row_threshold: float = 0.5,
    column_threshold: float = 0.5,
    keepna: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    # build subset
    orig_preds = preds
    preds = []
    for pred in orig_preds:
        if pred in data.columns:
            preds.append(pred)
        else:
            print(f"{pred} not found in data.")

    cols_to_keep = preds + confounder + ["group", "num_range"]
    if targets is not None:
        targets = [x if x in data.columns else x.replace("-", "_") for x in targets]
        cols_to_keep += targets
    data.insert(0, "num_range", list(range(data.shape[0])))
    subset = data.loc[:, cols_to_keep]

    sess_time = None
    if "SessionUploadTime" in subset.columns:
        sess_time = subset["SessionUploadTime"]

    if not keepna:
        # remove missing values
        print(f"Before: {subset.shape}")
        row_boolean = subset.notna().sum(1) / subset.shape[1]
        print(f"Rows with many na values: {np.sum(row_boolean < row_threshold)})")
        subset = subset.loc[row_boolean > row_threshold, :]
        print(f"After step 1: {subset.shape}")
        col_boolean = subset.notna().sum() / subset.shape[0]
        for x, v in zip(subset.columns, col_boolean):
            print(f"{x}: {v}")
        print(
            f"Cols with many na values: {subset.columns[col_boolean < column_threshold]})"
        )
        subset = subset.loc[:, col_boolean > column_threshold]
        print(f"After step 2: {subset.shape}")

        # check specifically for the predictors
        predictor_subset = subset.loc[:, [x for x in preds if x in subset.columns]]
        # check how many nan values per row
        nan_ratio = predictor_subset.isna().sum(1) / predictor_subset.shape[1]
        # exclude rows with more than 50% nan values
        subset = subset.loc[nan_ratio < 0.5, :]
        print(f"After step 3: {subset.shape}")

        # check nan ratio per row
        nan_ratio = max(subset.isna().sum(1) / subset.shape[1])
        if nan_ratio > 0.4:
            raise Exception(f"At least one row has {nan_ratio * 100}% nan values")
    else:
        # remove at least columns which are never filled
        print(f"Before: {subset.shape}")
        col_boolean = subset.notna().sum() / subset.shape[0]
        print(f"Cols with many na values: {subset.columns[col_boolean < 0.5]})")
        subset = subset.loc[:, col_boolean > 0.5]
        print(f"After: {subset.shape}")

    if targets is not None:
        if len(targets) > 1:
            raise NotImplementedError
        if any([x not in subset.columns.tolist() for x in targets]):
            subset = pd.merge(
                subset,
                data.loc[:, targets + ["num_range"]],
                on="num_range",
                how="left",
            )
            subset = subset.loc[subset[targets[0]].notna(), :]

    if sess_time is not None and "SessionUploadTime" not in subset.columns:
        subset["SessionUploadTime"] = sess_time

    if "site" in list(confounder):
        site_enc = site_encoder(subset.pop("site").tolist())
        site_enc.index = subset.index
        confounder = [x for x in confounder if x != "site"] + list(
            SITE_CONVERSION.keys()
        )
        subset = pd.concat([subset, site_enc], axis=1)

    if "season" in list(confounder):
        season_enc = season_encoder(subset.pop("season").tolist())
        season_enc.index = subset.index
        confounder = [x for x in confounder if x != "season"] + list(
            SEASON_CONVERSION.keys()
        )
        subset = pd.concat([subset, season_enc], axis=1)

    preds = [x for x in preds if x in subset.columns]
    xdata = subset.loc[:, preds + confounder]
    ydata = subset.loc[:, targets] if targets is not None else None
    groups = subset.loc[:, "group"].map(lambda x: NEW_GROUPS_INVERSE.get(x))

    return xdata, ydata, groups


@app.command()
def generate_gold_standard(
    input_file: Path,
    output_xdata: Path,
    output_group: Path,
    output_features: Path,
    keepna: bool = False,
):
    if str(input_file).endswith("csv"):
        merged_data = pd.read_csv(input_file)
    else:
        merged_data = pd.read_excel(input_file)
        merged_data.rename({"AIADL_SV_tscore": "tscore"}, axis=1, inplace=True)

    def psqi(df):
        def _c1(df):
            return df["psqi_9"]

        def _c2(df):
            p1 = df["psqi_2"].to_numpy()
            p1[np.where(p1 <= 15)] = 0
            p1[np.where((p1 > 15) & (p1 <= 30))] = 1
            p1[np.where((p1 > 30) & (p1 <= 60))] = 2
            p1[np.where((p1 > 60))] = 3
            p2 = df["psqi_5a"].to_numpy()
            psum = p1 + p2
            psum[np.where((psum == 0))] = 0
            psum[np.where((psum > 0) & (psum <= 2))] = 1
            psum[np.where((psum > 2) & (psum <= 4))] = 2
            psum[np.where((psum > 4) & (psum <= 6))] = 3
            return psum

        def _c3(df):
            x = df["psqi_4"].to_numpy()

            result = copy(x)
            result[np.where((x > 7))] = 0
            result[np.where((x >= 6) & (x <= 7))] = 1
            result[np.where((x >= 5) & (x < 6))] = 2
            result[np.where((x < 5))] = 3
            return result

        def _c4(df):
            bed_time = pd.to_datetime(df["psqi_1"], format="%H:%M")
            wake_time = pd.to_datetime(df["psqi_3"], format="%H:%M") + pd.DateOffset(
                days=1
            )
            hours_slept = df["psqi_4"]

            time_diff = [None] * len(bed_time)
            for i, (wake, bed) in enumerate(zip(wake_time, bed_time)):
                diff = wake - bed
                if diff.days > 0:
                    bed = bed + pd.DateOffset(days=1)
                    diff = wake - bed

                time_diff[i] = diff.seconds / 3600

            efficiency = ((hours_slept / time_diff) * 100).to_numpy()
            result = copy(efficiency)
            result[np.where(efficiency >= 85)] = 0
            result[np.where((efficiency >= 75) & (efficiency < 85))] = 1
            result[np.where((efficiency >= 65) & (efficiency < 75))] = 2
            result[np.where((efficiency < 65))] = 3

            return result

        def _c5(df):
            ssum = (
                df[
                    [
                        "psqi_5b",
                        "psqi_5c",
                        "psqi_5d",
                        "psqi_5e",
                        "psqi_5f",
                        "psqi_5g",
                        "psqi_5h",
                        "psqi_5i",
                    ]
                ]
                .sum(axis=1)
                .to_numpy()
            )

            result = copy(ssum)
            result[np.where(ssum == 0)] = 0
            result[np.where((ssum > 0) & (ssum <= 9))] = 1
            result[np.where((ssum > 9) & (ssum <= 18))] = 2
            result[np.where((ssum > 18) & (ssum <= 27))] = 3
            return result

        def _c6(df):
            return df["psqi_6"]

        def _c7(df):
            ssum = df[["psqi_7", "psqi_8"]].sum(axis=1).to_numpy()

            result = copy(ssum)
            result[np.where((ssum == 0))] = 0
            result[np.where((ssum > 0) & (ssum <= 2))] = 1
            result[np.where((ssum > 2) & (ssum <= 4))] = 2
            result[np.where((ssum > 4) & (ssum <= 6))] = 3

            return result

        return _c1(df) + _c2(df) + _c3(df) + _c4(df) + _c5(df) + _c6(df) + _c7(df)

    def sfs(df):
        p1 = (15 - df["sfs_withdrawal"]) / 15
        p2 = (
            30
            - df[
                [
                    "sfs_interpersonal_1",
                    "sfs_interpersonal_2",
                    "sfs_interpersonal_3",
                    "sfs_interpersonal_4",
                    "sfs_interpersonal_5",
                    "sfs_interpersonal_6",
                    "sfs_interpersonal_7",
                    "sfs_interpersonal_8",
                    "sfs_interpersonal_9",
                    "sfs_interpersonal_10",
                ]
            ].sum(axis=1)
        ) / 30
        p3 = (78 - df["sfs_prosocial"]) / 78
        return p1 + p2 + p3

    s1 = merged_data["R20"]
    s2 = (
        merged_data[
            [
                "mmse_year",
                "mmse_season",
                "mmse_month",
                "mmse_date",
                "mmse_day_of_week",
                "mmse_state",
                "mmse_county",
                "mmse_town",
                "mmse_hospital",
                "mmse_floor",
            ]
        ]
        .sum(axis=1)
        .tolist()
    )
    s3 = merged_data["word_list_recall"]
    s4 = merged_data["R29"]
    s5 = merged_data["rey_recall"]
    s6 = merged_data["ecog_memory"] / merged_data["ecog_memory_completed"]
    s7 = (
        merged_data["ecog_visual_spatial"]
        / merged_data["ecog_visual_spatial_completed"]
    )
    s8 = merged_data["ecog_planning"] / merged_data["ecog_planning_completed"]
    s9 = merged_data[["R11", "R14", "R15", "R16", "R17"]].sum(axis=1).tolist()
    s10 = merged_data["eq5d_selfcare"]
    s11 = merged_data["R30"]
    s12 = 10 - merged_data[["adcs_5", "adcs_6a", "adcs_6b"]].sum(axis=1)
    s13 = merged_data[["R1", "R2", "R3", "R4", "R5", "R6"]].sum(axis=1).tolist()
    s14 = merged_data["ecog_organization"] / merged_data["ecog_organization_completed"]
    s15 = psqi(merged_data)
    s16 = merged_data[
        [
            "ess_sitting_reading",
            "ess_watchingtv",
            "ess_sitting_inactive",
            "ess_passenger",
            "ess_lying_down",
            "ess_sitting_talking",
            "ess_sitting_after_lunch",
            "ess_car_stopped",
        ]
    ].sum(axis=1)
    s17 = (
        merged_data["npiq_nighttime_frequency"] * merged_data["npiq_nighttime_severity"]
    )
    s18 = 60 - merged_data[
        [
            "sp_phonecall_dex",
            "sp_phonecall_selfass",
            "sp_phonecall_eff",
            "sp_sms_dex",
            "sp_sms_selfass",
            "sp_sms_eff",
            "sp_whatsapp_dex",
            "sp_whatsapp_selfass",
            "sp_whatsapp_eff",
            "sp_email_dex",
            "sp_email_selfass",
            "sp_email_eff",
        ]
    ].sum(axis=1)
    s19 = merged_data[["R7", "R8", "R9", "R10", "R22", "R23", "R24", "R25", "R26"]].sum(
        axis=1
    )
    s20 = 9 - merged_data[["adcs_7a", "adcs_23b"]].sum(axis=1)
    s21 = merged_data[
        [
            "fluency_letter1_score",
            "fluency_letter2_score",
            "fluency_letter3_score",
            "fluency_animal",
        ]
    ].sum(axis=1)
    s22 = merged_data["boston_total"]
    s23 = merged_data["eq5d_mobility"]
    s24 = merged_data[["R28", "R27"]].sum(axis=1)
    s25 = sfs(merged_data)
    s26 = merged_data["gdss_score"]
    s27 = merged_data["npiq_apathy_frequency"] * merged_data["npiq_apathy_severity"]
    s28 = 15 - merged_data["sfs_withdrawal"]
    clinical_gs = pd.DataFrame(
        {
            "record_id": merged_data["record_id"],
            "group": merged_data["group"],
            "age": merged_data["age"],
            "sex": merged_data["sex"],
            "site": merged_data["site"],
            "education_years": merged_data["education_years"],
            "planning_skills": s8,
            "finances": s9,
            "difficulties_at_work": s1,
            "spatial_navigation_memory": average_z_scores(
                [s3, s4, s5, s6, s7]
            ),  # removal of s2 as it contains mmse 230929
            "self_care": average_z_scores([s10, s11, s12]),
            "self_management": average_z_scores([s13, s14]),
            "spatial_navigation_memory_1": s2,
            "spatial_navigation_memory_2": s3,
            "spatial_navigation_memory_3": s4,
            "spatial_navigation_memory_4": s5,
            "spatial_navigation_memory_5": s6,
            "spatial_navigation_memory_6": s7,
            "self_care_1": s10,
            "self_care_2": s11,
            "self_care_3": s12,
            "self_management_1": s13,
            "self_management_2": s14,
            "sleep": average_z_scores([s15, s16, s17]),
            "sleep_1": s15,
            "sleep_2": s16,
            "sleep_3": s17,
            "technology": average_z_scores([s18, s19, s20]),
            "technology_1": s18,
            "technology_2": s19,
            "technology_3": s20,
            "dysnomia": average_z_scores([s21, s22]),
            "dysnomia_1": s21,
            "dysnomia_2": s22,
            "gait": s23,
            "difficulties_driving": s24,
            "interpersonal": s25,
            "motivation": average_z_scores([s26, s27, s28]),
            "motivation_1": s26,
            "motivation_2": s27,
            "motivation_3": s28,
        }
    )

    targets = [
        "difficulties_at_work",
        "spatial_navigation_memory",
        "planning_skills",
        "finances",
        "self_care",
        "self_management",
        "sleep",
        "technology",
        "dysnomia",
        "gait",
        "difficulties_driving",
        "interpersonal",
        "motivation",
    ]
    data = prepare_df(
        clinical_gs,
        targets,
        [],
        None,
        extra=["group", "record_id"] + list(CONFOUNDERS),
    )
    confounder = list(CONFOUNDERS)
    if "site" in confounder:
        confounder = [x for x in list(CONFOUNDERS) if x != "site"] + list(
            SITE_CONVERSION.keys()
        )

    if "season" in confounder:
        confounder = [x for x in list(CONFOUNDERS) if x != "season"] + list(
            SEASON_CONVERSION.keys()
        )

    predictors, group, _ = clean_dataframe(
        data,
        target_columns=targets + confounder,
        row_threshold=0.65,
        column_threshold=0.9,
        keepna=keepna,
    )

    predictors.to_csv(output_xdata)
    group = group.map(lambda x: NEW_GROUPS_INVERSE.get(x, x))
    group.to_csv(output_group)

    with output_features.open("w+") as f:
        for col in list(predictors.columns):
            f.write(f"{col}\n")

    print(Counter(map(lambda x: NEW_GROUPS[x], group)))


@app.command()
def generate_iadl_dataset(
    input_file: Path,
    output_xdata: Path,
    output_group: Path,
    output_features: Path,
    keepna: bool = False,
):
    if str(input_file).endswith("csv"):
        merged_data = pd.read_csv(input_file)
    else:
        merged_data = pd.read_excel(input_file)
        merged_data.rename({"AIADL_SV_tscore": "tscore"}, axis=1, inplace=True)

    clinical_gs = pd.DataFrame(
        {
            "record_id": merged_data["record_id"],
            "group": merged_data["group"],
            "age": merged_data["age"],
            "sex": merged_data["sex"],
            "site": merged_data["site"],
            "education_years": merged_data["education_years"],
            "tscore": merged_data["tscore"],
        }
    )

    targets = ["tscore"]
    data = prepare_df(
        clinical_gs,
        targets,
        [],
        None,
        extra=["group", "record_id"] + list(CONFOUNDERS),
    )
    confounder = list(CONFOUNDERS)
    if "site" in confounder:
        confounder = [x for x in list(CONFOUNDERS) if x != "site"] + list(
            SITE_CONVERSION.keys()
        )

    if "season" in confounder:
        confounder = [x for x in list(CONFOUNDERS) if x != "season"] + list(
            SEASON_CONVERSION.keys()
        )

    predictors, group, _ = clean_dataframe(
        data,
        target_columns=targets + confounder,
        row_threshold=0.65,
        column_threshold=0.9,
        keepna=keepna,
    )

    predictors.to_csv(output_xdata)
    group = group.map(lambda x: NEW_GROUPS_INVERSE.get(x, x))
    group.to_csv(output_group)

    with output_features.open("w+") as f:
        for col in list(predictors.columns):
            f.write(f"{col}\n")

    print(Counter(map(lambda x: NEW_GROUPS[x], group)))


if __name__ == "__main__":
    app()
