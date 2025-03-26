from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import typer

from scipy.stats import median_abs_deviation
from src.definitions import NEW_GROUPS, DEVICE_NAMES, BASE_MODELS
from src.visualization.visualize_clf_results import select_best_clf_only


app = typer.Typer()


@app.command()
def mean(
    input_files: List[Path],
    concise_output: Path,
    full_output: Path,
    plot_path: Path,
    metric: str = "auroc",
):
    """
    Calculate mean performance metrics across multiple input files.

    Args:
        input_files: List of input CSV files containing classification results
        concise_output: Path to save concise output table (CSV)
        full_output: Path to save detailed output table (CSV)
        plot_path: Path to save the visualization plot
        metric: Metric to evaluate (auroc or aupr, default: auroc)

    Returns:
        None. Results are saved to the specified output paths.
    """
    def _read_file(file: Path) -> pd.DataFrame:
        """Reads a file and adds the dataset name as a column"""
        dataset = str(file).split("/")[2]
        try:
            tmp = pd.read_csv(file, index_col=0)
        except:
            raise Exception(f"{file} could not be parsed")
        tmp["dataset"] = dataset
        if str(file).endswith("gs.csv"):
            tmp["model"] = tmp["model"] + "_gs"
        return tmp

    # read all files and concat them
    dfs = [_read_file(file) for file in input_files]
    df = pd.concat(dfs)

    # group by model, comparison, seed, and dataset and compute the mean
    df_avg = (
        df.groupby(["model", "comparison", "seed", "dataset"])
        .mean()
        .drop("split", axis=1)
        .reset_index()
    )

    # melt the dataframe to have a column for the metric
    df_melt = df_avg.melt(
        id_vars=["model", "seed", "comparison", "dataset"],
        value_vars=["auroc", "aupr"],
    ).rename({"model": "Model"}, axis=1)

    # separate the base model from the other models
    base_model = df_melt.loc[df_melt["Model"].isin(BASE_MODELS), :].rename(
        {"value": "base_score"}, axis=1
    )
    base_model = select_best_clf_only(
        base_model,
        id_vars=["comparison", "dataset"],
        agg_vars=["Model"],
        metric_var="variable",
        decision_var=metric,
        sort_var="base_score",
    )

    assert_table = base_model.loc[base_model["variable"] == metric, :].pivot_table(
        index=["comparison", "seed", "Model"],
        columns="dataset",
        values="base_score",
    )
    for idx, row in assert_table.iterrows():
        group_1 = row[
            ["altoida", "altoidaDns", "banking", "gs", "iadl", "mezurio"]
        ].round(2)
        group_2 = row[["fitbit", "axivity", "gait_dual", "gait_tug"]].round(2)
        unique_group_1 = group_1.dropna().unique()
        unique_group_2 = group_2.dropna().unique()

        if len(unique_group_1) > 1 and max(unique_group_1) - min(unique_group_1) > 0.03:
            raise ValueError(
                f"Multiple different values found in row {idx}: {group_1.dropna().tolist()}"
            )
        if len(unique_group_2) > 1 and max(unique_group_2) - min(unique_group_2) > 0.01:
            raise ValueError(
                f"Multiple different values found in row {idx}: {group_2.dropna().tolist()}"
            )

    # separate the other models from the base model
    other_models = df_melt.loc[~df_melt["Model"].isin(BASE_MODELS), :]

    # separate the combined and data-only models
    combined_models = other_models.loc[other_models["Model"].str.contains("gs"), :]
    data_only_models = other_models.loc[~other_models["Model"].str.contains("gs"), :]

    # perform the other processing steps to merge base and other models
    data_only_processed = process_data(base_model, data_only_models, metric)
    combined_processed = process_data(base_model, combined_models, metric)
    combined_processed.rename(
        {
            "value": "combined_score",
            "ref_diff": "combined_ref_diff",
            "Model": "Combined_Model",
            "base_score": "combined_base_score",
        },
        axis=1,
        inplace=True,
    )

    def se(x) -> float:
        print(f"{x=}")
        print(f"{x.std()=}")
        print(f"{len(x)=}")
        return x.std() / np.sqrt(len(x))

    # Split aggregation into steps
    agg_single = (
        data_only_processed.groupby(["Model", "comparison", "dataset", "variable"])
        .aggregate(
            {
                "value": ["mean", "std", se],
                "base_score": ["mean", "std", se],
                "ref_diff": ["mean", "std", se],
            }
        )
        .pipe(lambda x: x.set_axis([f"{i}_{j}" for i, j in x.columns], axis=1))
    )

    agg_combined = (
        combined_processed.groupby(
            ["Combined_Model", "comparison", "dataset", "variable"],
        )
        .agg(
            {
                "combined_score": ["mean", "std", se],
                "combined_base_score": ["mean", "std", se],
                "combined_ref_diff": ["mean", "std", se],
            }
        )
        .pipe(lambda x: x.set_axis([f"{i}_{j}" for i, j in x.columns], axis=1))
    )

    merged = pd.merge(
        agg_single,
        agg_combined,
        on=["dataset", "comparison", "variable"],
        how="outer",
    ).reset_index()

    # Multiple by 100 to get percentage
    cols_to_multiply = [
        "value_mean",
        "base_score_mean",
        "ref_diff_mean",
        "value_se",
        "base_score_se",
        "ref_diff_se",
        "value_std",
        "base_score_std",
        "ref_diff_std",
        "combined_score_mean",
        "combined_base_score_mean",
        "combined_ref_diff_mean",
        "combined_score_se",
        "combined_base_score_se",
        "combined_ref_diff_se",
        "combined_score_std",
        "combined_base_score_std",
        "combined_ref_diff_std",
    ]
    for col in cols_to_multiply:
        merged[col] = merged[col] * 100

    # select only the auroc metric
    auc = merged.loc[merged.variable == metric, :]

    # translate the comparison names to meaningful names
    class_names = []
    for cls_instance in auc["comparison"]:
        a, b = cls_instance.split(" vs. ")
        a_trans = NEW_GROUPS[int(a) + 1]
        b_trans = NEW_GROUPS[int(b) + 1]
        class_names.append(f"{a_trans} vs. {b_trans}")
    auc.insert(0, "Comparison", class_names)

    # translate the dataset names to meaningful names
    auc.insert(0, "Dataset", list(map(DEVICE_NAMES.get, auc["dataset"])))

    # remove the Altoida dataset for the MildAD comparisons
    auc.loc[
        (auc["Comparison"].str.contains("MildAD"))
        & (auc["Dataset"].str.contains("Altoida")),
        [
            "value_mean",
            "combined_score_mean",
            "base_score_mean",
            "value_se",
            "combined_score_se",
            "base_score_se",
            "value_std",
            "combined_score_std",
            "base_score_std",
        ],
    ] = np.nan

    # create a dataframe for plotting
    plot_df = auc.rename({"variable": "Metric"}, axis=1).melt(
        id_vars=["Dataset", "Comparison", "Metric"],
        value_vars=["value_mean", "combined_score_mean", "base_score_mean"],
    )
    plot_df["variable_name"] = plot_df["variable"].map(
        {
            "base_score_mean": "Base",
            "value_mean": "DS",
            "combined_score_mean": "DS-GS",
        }.get
    )
    plot_df = plot_df.loc[plot_df["Metric"] == metric, :]
    plot_df["Dataset"] = plot_df["Dataset"].astype("category")
    plot_df["Dataset"].cat.reorder_categories(
        sorted(plot_df["Dataset"].drop_duplicates().tolist())
    )
    try:
        plot_df["Dataset"].cat.remove_categories("In-Clinic Assessment (Questionnaire)")
    except ValueError:
        pass

    sns.set_context("talk")
    g = sns.FacetGrid(plot_df, col="Comparison", col_wrap=3, height=6)
    g.map(
        sns.barplot,
        "variable_name",
        "value",
        "Dataset",
        order=["Base", "DS", "DS-GS"],
        palette=sns.color_palette("Paired"),
    )
    g.add_legend()
    g.set_ylabels("AUC")
    g.set_xlabels("Variants")
    g.savefig(str(plot_path))

    def make_string(row, mean_col, se_col):
        return (
            f"{row[mean_col]} ({row[se_col]})" if not np.isnan(row[mean_col]) else "-"
        )

    concise = (
        auc.applymap(lambda x: round(x, 1) if isinstance(x, float) else x)
        .assign(
            value_string=lambda x: x.apply(
                lambda row: make_string(row, "value_mean", "value_se"), axis=1
            ),
            combined_string=lambda x: x.apply(
                lambda row: make_string(
                    row, "combined_score_mean", "combined_score_se"
                ),
                axis=1,
            ),
            base_string=lambda x: x.apply(
                lambda row: make_string(row, "base_score_mean", "base_score_se"), axis=1
            ),
        )
        .loc[
            :,
            [
                "value_string",
                "combined_string",
                "base_string",
                "Dataset",
                "Comparison",
            ],
        ]
        .pivot(
            index="Dataset",
            columns="Comparison",
            values=[
                "value_string",
                "combined_string",
                "base_string",
            ],
        )
        .sort_index(axis=1, level=0)
        .rename(
            {
                "value_string": "DS",
                "combined_string": "DS-GS",
                "base_string": "Base",
            },
            axis=1,
        )
    )

    # Update full table string to use standard error
    full_table_string = []
    for value, vse, cval, cse, ref, refse in zip(
        auc["value_mean"],
        auc["value_se"],
        auc["combined_score_mean"],
        auc["combined_score_se"],
        auc["base_score_mean"],
        auc["base_score_se"],
    ):
        full_table_string.append(
            f"{round(value, 3)} ({round(vse, 3)}) - {round(cval, 3)} ({round(cse, 3)}) - {round(ref, 3)} ({round(refse, 3)})"
        )
    auc.insert(0, "full_string", full_table_string)

    full = auc.pivot(index="Dataset", columns="Comparison", values="full_string").loc[
        :,
        [
            "HC vs. PreAD",
            "HC vs. ProAD",
            "HC vs. MildAD",
            "PreAD vs. ProAD",
            "ProAD vs. MildAD",
            "PreAD vs. MildAD",
            "HC vs. AD Spectrum",
            "HC vs. ProAD+MildAD",
        ],
    ]

    concise.to_csv(concise_output)
    full.to_csv(full_output)

    # Convert to LaTeX with booktabs
    latex_table = format_for_latex(concise).to_latex(
        index=False,
        column_format="l" + "X" * (len(concise.columns.levels[1])),
        multicolumn=True,
        multicolumn_format="c",
        caption="Your caption here",
        label=f"tab:{metric}_mean_se",
        escape=False,
    )

    # Add midrules between different model types
    latex_lines = latex_table.split("\n")
    for i, line in enumerate(latex_lines):
        if "Base" in line and "RMT" in latex_lines[i + 1]:
            latex_lines.insert(i + 1, "\\midrule")
        elif (
            "RMT" in line and "RMT+FDS" in latex_lines[i + 1] and "RMT+FDS" not in line
        ):
            latex_lines.insert(i + 1, "\\midrule")

    latex_table = "\n".join(latex_lines)

    # Save to file
    with open(concise_output.with_suffix(".tex"), "w") as f:
        f.write(latex_table)

    # After creating the concise DataFrame, modify it to get the desired LaTeX format


def format_for_latex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the DataFrame for LaTeX output.

    This function reformats the input DataFrame (which is a multi-level column DataFrame
    with Base, DS, and DS-GS performance metrics) into a format suitable for LaTeX output.
    It creates separate rows for Base, RMT (Remote Monitoring Technology), and RMT+FDS
    (Remote Monitoring Technology + Functional Digital Signatures) models.

    Args:
        df: A pandas DataFrame with multi-level columns for Base, DS, and DS-GS metrics

    Returns:
        A reformatted pandas DataFrame ready for LaTeX output
    """
    # Create separate DataFrames for each model type
    base_df = df.xs("Base", axis=1, level=0).copy()  # Create a copy
    # Assign ["fitbit", "axivity", "gait"] as Base* Model and others as Base Model
    model_col = base_df.index.map(
        lambda x: "Base*"
        if x in ["Fitbit", "Axivity", "Physilog (Dual)", "Physilog (TUG)"]
        else "Base"
    )
    base_df.insert(0, "Model", model_col)

    # Group into one Base and Base* row
    def select_value(x):
        available = [val for val in set(x) if val != "-"]
        if len(available) == 1:
            return available[0]
        else:
            raise ValueError(f"Multiple values found: {available}")

    # get unique values for each column except Model
    base_df = base_df.groupby("Model").agg(
        {
            "HC vs. PreAD": select_value,
            "HC vs. ProAD": select_value,
            "HC vs. MildAD": select_value,
            "PreAD vs. ProAD": select_value,
            "ProAD vs. MildAD": select_value,
            "PreAD vs. MildAD": select_value,
            "HC vs. AD Spectrum": select_value,
            "HC vs. ProAD+MildAD": select_value,
        }
    )

    ds_df = df.xs("DS", axis=1, level=0)
    ds_gs_df = df.xs("DS-GS", axis=1, level=0)

    # Concatenate vertically with labels
    final_df = pd.concat(
        [
            base_df.assign(Type=base_df.index),
            ds_df.assign(Type="RMT"),
            ds_gs_df.assign(Type="RMT+FDS"),
        ]
    )

    # Concatenate index and Type
    final_df["Type"] = final_df.index + " (" + final_df["Type"] + ")"

    # Move Model column to front
    cols = final_df.columns.tolist()
    cols = [
        "Type",
        "HC vs. PreAD",
        "HC vs. ProAD",
        "HC vs. MildAD",
        "PreAD vs. ProAD",
        "ProAD vs. MildAD",
        "PreAD vs. MildAD",
        "HC vs. AD Spectrum",
        "HC vs. ProAD+MildAD",
    ]
    final_df = final_df[cols]

    return final_df


def process_data(
    base_model: pd.DataFrame, other_models: pd.DataFrame, metric: str = "auroc"
) -> pd.DataFrame:
    """
    Process data by selecting the best model per comparison and dataset.

    This function takes a base model DataFrame and another model DataFrame,
    selects the best model per comparison and dataset from the other models,
    merges it with the base model, and computes the difference between them.

    Args:
        base_model: DataFrame containing base model performance metrics
        other_models: DataFrame containing other models' performance metrics
        metric: Metric to use for model selection (default: "auroc")

    Returns:
        DataFrame with merged data and computed differences
    """

    # select the best model per comparison and dataset
    selected = select_best_clf_only(
        other_models,
        id_vars=["comparison", "dataset"],
        agg_vars=["Model"],
        metric_var="variable",
        decision_var=metric,
        sort_var="value",
    )
    # merge the base model with the selected models
    merged = (
        pd.merge(
            selected,
            base_model,
            on=["dataset", "comparison", "variable", "seed"],
            how="left",
        )
        .drop("Model_y", axis=1)
        .rename({"Model_x": "Model"}, axis=1)
    )
    # compute the difference between the selected model and the base model
    merged["ref_diff"] = merged["value"] - merged["base_score"]
    return merged


@app.command()
def median(
    input_files: List[Path],
    concise_output: Path,
    full_output: Path,
    plot_path: Path,
):
    """
    Calculate median performance metrics across multiple input files.

    Args:
        input_files: List of input CSV files containing classification results
        concise_output: Path to save concise output table (CSV)
        full_output: Path to save detailed output table (CSV)
        plot_path: Path to save the visualization plot

    Returns:
        None. Results are saved to the specified output paths.
    """
    def _read_file(file: Path) -> pd.DataFrame:
        """Reads a file and adds the dataset name as a column"""
        dataset = str(file).split("/")[2]
        try:
            tmp = pd.read_csv(file, index_col=0)
        except:
            raise Exception(f"{file} could not be parsed")
        tmp["dataset"] = dataset
        if str(file).endswith("gs.csv"):
            tmp["model"] = tmp["model"] + "_gs"
        return tmp

    # read all files and concat them
    dfs = [_read_file(file) for file in input_files]
    df = pd.concat(dfs)

    # group by model, comparison, seed, and dataset and compute the median
    df_avg = (
        df.groupby(["model", "comparison", "seed", "dataset"])
        .median()
        .drop("split", axis=1)
        .reset_index()
    )

    # melt the dataframe to have a column for the metric
    df_melt = df_avg.melt(
        id_vars=["model", "seed", "comparison", "dataset"],
        value_vars=["auroc", "aupr"],
    ).rename({"model": "Model"}, axis=1)

    # separate the base model from the other models
    base_model = df_melt.loc[df_melt["Model"].isin(BASE_MODELS), :].rename(
        {"value": "base_score"}, axis=1
    )
    base_model = select_best_clf_only(
        base_model,
        id_vars=["comparison", "dataset"],
        agg_vars=["Model"],
        metric_var="variable",
        decision_var="auroc",
        sort_var="base_score",
    )

    # separate the other models from the base model
    other_models = df_melt.loc[~df_melt["Model"].isin(BASE_MODELS), :]

    # separate the combined and data-only models
    combined_models = other_models.loc[other_models["Model"].str.contains("gs"), :]
    data_only_models = other_models.loc[~other_models["Model"].str.contains("gs"), :]

    # perform the other processing steps to merge base and other models
    data_only_processed = process_data(base_model, data_only_models)
    combined_processed = process_data(base_model, combined_models)
    combined_processed.rename(
        {
            "value": "combined_score",
            "ref_diff": "combined_ref_diff",
            "Model": "Combined_Model",
            "base_score": "combined_base_score",
        },
        axis=1,
        inplace=True,
    )

    agg_single = (
        data_only_processed.groupby(
            ["Model", "comparison", "dataset", "variable"],
        )
        .agg(
            {
                "value": ["median", median_abs_deviation],
                "base_score": ["median", median_abs_deviation],
                "ref_diff": ["median", median_abs_deviation],
            }
        )
        .pipe(lambda x: x.set_axis([f"{i}_{j}" for i, j in x.columns], axis=1))
    )

    agg_combined = (
        combined_processed.groupby(
            ["Combined_Model", "comparison", "dataset", "variable"],
        )
        .agg(
            {
                "combined_score": ["median", median_abs_deviation],
                "combined_base_score": ["median", median_abs_deviation],
                "combined_ref_diff": ["median", median_abs_deviation],
            }
        )
        .pipe(lambda x: x.set_axis([f"{i}_{j}" for i, j in x.columns], axis=1))
    )

    merged = pd.merge(
        agg_single,
        agg_combined,
        on=["dataset", "comparison", "variable"],
        how="outer",
    ).reset_index()

    auc = merged.loc[merged.variable == "auroc", :]

    # translate the comparison names to meaningful names
    class_names = []
    for cls_instance in auc["comparison"]:
        a, b = cls_instance.split(" vs. ")
        a_trans = NEW_GROUPS[int(a) + 1]
        b_trans = NEW_GROUPS[int(b) + 1]
        class_names.append(f"{a_trans} vs. {b_trans}")
    auc.insert(0, "Comparison", class_names)

    # translate the dataset names to meaningful names
    auc.insert(0, "Dataset", list(map(DEVICE_NAMES.get, auc["dataset"])))

    # remove the Altoida dataset for the MildAD comparisons
    auc.loc[
        (auc["Comparison"].str.contains("MildAD"))
        & (auc["Dataset"].str.contains("Altoida")),
        [
            "value_median",
            "combined_score_median",
            "base_score_median",
            "value_median_abs_deviation",
            "combined_score_median_abs_deviation",
            "base_score_median_abs_deviation",
        ],
    ] = np.nan

    # Create plot dataframe
    plot_df = auc.rename({"variable": "Metric"}, axis=1).melt(
        id_vars=["Dataset", "Comparison", "Metric"],
        value_vars=[
            "value_median",
            "combined_score_median",
            "base_score_median",
        ],
    )
    plot_df["variable_name"] = plot_df["variable"].map(
        {
            "base_score_median": "Base",
            "value_median": "DS",
            "combined_score_median": "DS-GS",
        }.get
    )
    plot_df = plot_df.loc[plot_df["Metric"] == "auroc", :]
    plot_df["Dataset"] = plot_df["Dataset"].astype("category")
    plot_df["Dataset"].cat.reorder_categories(
        sorted(plot_df["Dataset"].drop_duplicates().tolist())
    )
    try:
        plot_df["Dataset"].cat.remove_categories("In-Clinic Assessment (Questionnaire)")
    except ValueError:
        pass

    # Create and save plot
    sns.set_context("talk")
    g = sns.FacetGrid(plot_df, col="Comparison", col_wrap=3, height=6)
    g.map(
        sns.barplot,
        "variable_name",
        "value",
        "Dataset",
        order=["Base", "DS", "DS-GS"],
        palette=sns.color_palette("Paired"),
    )
    g.add_legend()
    g.set_ylabels("AUC")
    g.set_xlabels("Variants")
    g.savefig(str(plot_path))

    # Helper function for string formatting
    def make_string(row, mean_col, mad_col):
        return (
            f"{row[mean_col]} ({row[mad_col]})" if not np.isnan(row[mean_col]) else "-"
        )

    # Create concise table with consistent formatting
    concise = (
        auc.applymap(lambda x: round(x, 2) if isinstance(x, float) else x)
        .assign(
            value_string=lambda x: x.apply(
                lambda row: make_string(
                    row, "value_median", "value_median_abs_deviation"
                ),
                axis=1,
            ),
            combined_string=lambda x: x.apply(
                lambda row: make_string(
                    row,
                    "combined_score_median",
                    "combined_score_median_abs_deviation",
                ),
                axis=1,
            ),
            base_string=lambda x: x.apply(
                lambda row: make_string(
                    row, "base_score_median", "base_score_median_abs_deviation"
                ),
                axis=1,
            ),
        )
        .loc[
            :,
            [
                "value_string",
                "combined_string",
                "base_string",
                "Dataset",
                "Comparison",
            ],
        ]
        .pivot(
            index="Dataset",
            columns="Comparison",
            values=[
                "value_string",
                "combined_string",
                "base_string",
            ],
        )
        .sort_index(axis=1, level=0)
        .rename(
            {
                "value_string": "DS",
                "combined_string": "DS-GS",
                "base_string": "Base",
            },
            axis=1,
        )
    )

    # Create full table string with median and MAD
    full_table_string = []
    for value, vmad, cval, cmad, ref, refmad in zip(
        auc["value_median"],
        auc["value_median_abs_deviation"],
        auc["combined_score_median"],
        auc["combined_score_median_abs_deviation"],
        auc["base_score_median"],
        auc["base_score_median_abs_deviation"],
    ):
        full_table_string.append(
            f"{round(value, 3)} ({round(vmad, 3)}) - {round(cval, 3)} ({round(cmad, 3)}) - {round(ref, 3)} ({round(refmad, 3)})"
        )
    auc.insert(0, "full_string", full_table_string)

    # Create full table with consistent column ordering
    full = auc.pivot(index="Dataset", columns="Comparison", values="full_string").loc[
        :,
        [
            "HC vs. PreAD",
            "HC vs. ProAD",
            "HC vs. MildAD",
            "PreAD vs. ProAD",
            "ProAD vs. MildAD",
            "PreAD vs. MildAD",
            "HC vs. AD Spectrum",
            "HC vs. ProAD+MildAD",
        ],
    ]

    concise.to_csv(concise_output)
    full.to_csv(full_output)


if __name__ == "__main__":
    app()
