from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import typer
from plotnine import (
    aes,
    facet_wrap,
    geom_boxplot,
    ggplot,
    labs,
    theme_bw,
    theme,
    element_line,
)

from src.definitions import (
    NEW_GROUPS,
    BASE_MODELS,
)

MODEL_NAMES = {
    "lr_base": "LR (Base)",
    "lr": "LR",
    "rf": "RF",
    "cart": "DT",
    "xgb": "XGBoost",
    "xgb_base": "XGBoost (Base)",
    "rf_base": "RF (Base)",
    "cart_base": "DT (Base)",
    "lr_clinical": "LR (+Q)",
    "rf_clinical": "RF (+Q)",
    "xgb_clinical": "XGBoost (+Q)",
    "cart_clinical": "DT (+Q)",
}


app = typer.Typer()


def count_unique(x):
    """Return the number of unique elements in a sequence."""
    return len(set(x))


def abs_diff(x):
    """Calculate the absolute difference between the maximum and minimum values in a sequence."""
    return abs(x.max() - x.min())


def select_best_clf_only(
    df: pd.DataFrame,
    id_vars: List[str],
    agg_vars: List[str] = ["Model"],
    decision_var: str = "referenced_auroc",
    sort_var: str = "Diff. to reference",
    metric_var: str = "Metric",
    model_col: str = "Model",
) -> pd.DataFrame:
    """Selects only the best model per comparison"""
    tmp = df.copy()

    # group by id vars; usually comparison and dataset
    # data is already averaged over the folds
    def get_best_model(subset):
        var_subset = subset.loc[subset[metric_var] == decision_var, :]
        # Average by seed
        grouping_cols = agg_vars + [metric_var] + id_vars
        average_per_seed = (
            var_subset.groupby(grouping_cols + ["seed"]).mean().reset_index()
        )
        # average over seeds
        averaged_over_seed = average_per_seed.groupby(grouping_cols).agg(
            {
                sort_var: [
                    "mean",
                    "std",
                    "count",
                    count_unique,
                    abs_diff,
                    "max",
                    "min",
                ],
            }
        )
        averaged_over_seed.columns = [
            "_".join(col).strip() for col in averaged_over_seed.columns.values
        ]
        averaged_over_seed = averaged_over_seed.sort_values(
            f"{sort_var}_mean", ascending=False
        )
        best_model = (
            averaged_over_seed.index[0][0]
            if not isinstance(averaged_over_seed.index[0], str)
            else averaged_over_seed.index[0]
        )
        return best_model

    selection = []
    for _, subset in tmp.groupby(by=id_vars):
        best_model = get_best_model(subset)
        selected_subset = subset.loc[subset[model_col] == best_model, :]
        selection.append(selected_subset)
    return pd.concat(selection)


def plot_metric(
    input_files: List[Path],
    output: Path,
    metric: str,
    best_model_only: bool = False,
    raw_score: bool = False,
):
    """
    Generic function to plot either AUROC or AUPR metrics.

    Args:
        input_files: List of input file paths
        output: Output file path
        metric: Metric to plot ('auroc' or 'aupr')
        best_model_only: Whether to plot only the best model
        raw_score: Whether to plot raw scores instead of differences
    """
    if best_model_only:
        raise NotImplementedError

    metric_map = {
        "auroc": {"raw": "auroc", "referenced": "referenced_auroc", "label": "AUC"},
        "aupr": {"raw": "aupr", "referenced": "referenced_aupr", "label": "AUPR"},
    }

    if metric not in metric_map:
        raise ValueError(f"Metric must be one of {list(metric_map.keys())}")

    y_label = (
        f"{metric_map[metric]['label']} diff. to reference"
        if not raw_score
        else "Value"
    )

    if raw_score:
        df_melt, raw = data_preprocessing(
            input_files, y_label=y_label, comp_to_ref=False
        )
        df_melt = df_melt.loc[df_melt["Metric"] == metric_map[metric]["raw"], :]
    else:
        df_melt, raw = data_preprocessing(input_files, y_label=y_label)
        df_melt = df_melt.loc[df_melt["Metric"] == metric_map[metric]["referenced"], :]
        raw = raw.loc[raw["Metric"] == metric_map[metric]["raw"], :]

    if any(["altoida" in str(x) for x in input_files]):
        # remove rows with "HC vs. ProAD+MildAD" or rows that contains "MildAD"
        df_melt = df_melt.loc[
            ~df_melt["Comparison"].str.contains("ProAD\+MildAD|MildAD"), :
        ]
        # Set categories again
        df_melt["Comparison"] = pd.Categorical(
            values=df_melt["Comparison"],
            categories=[
                "HC\nvs.\nPreAD",
                "HC\nvs.\nProAD",
                "PreAD\nvs.\nProAD",
                "HC\nvs.\nAD Spectrum",
            ],
        )

    assert df_melt is not None
    width = 11.69
    height = 8.27

    if not raw_score:
        print(df_melt)
        g = (
            ggplot(
                data=df_melt,
                mapping=aes(x="Comparison", y=y_label),
            )
            + geom_boxplot(aes(color="Model"))
            + labs(x="Comparisons", y=y_label)
            + theme_bw()
            + theme(
                panel_grid_major=element_line(color="gray", linewidth=0.5),
                panel_grid_minor=element_line(color="lightgray", linewidth=0.25),
            )
        )
        g.save(str(output), dpi=300, width=width, height=height)
    else:
        df_melt["Group"] = df_melt["Model"].map(
            lambda x: "Base" if "Base" in x else "Base+RMT"
        )
        df_melt["model_type"] = df_melt["Model"].map(
            lambda x: x.replace("(Base)", "").strip()
        )
        g = (
            ggplot(
                data=df_melt,
                mapping=aes(x="model_type", y="Value"),
            )
            + geom_boxplot(aes(color="Group"))
            + facet_wrap("Comparison")
            + labs(x="Model", y=y_label)
            + theme_bw()
            + theme(
                panel_grid_major=element_line(color="gray", linewidth=0.5),
                panel_grid_minor=element_line(color="lightgray", linewidth=0.5),
            )
        )
        g.save(str(output), dpi=300, width=width, height=height)


# Wrapper for AUROC
@app.command()
def roc(
    input_files: List[Path],
    output: Path,
    best_model_only: bool = False,
    raw_score: bool = False,
):
    """Plot AUROC metrics."""
    plot_metric(input_files, output, "auroc", best_model_only, raw_score)


# Wrapper for AUPR
@app.command()
def pr(
    input_files: List[Path],
    output: Path,
    best_model_only: bool = False,
    raw_score: bool = False,
):
    """Plot AUPR metrics."""
    plot_metric(input_files, output, "aupr", best_model_only, raw_score)


def data_preprocessing(
    input_files, y_label: str = "Diff. to reference", comp_to_ref: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Preprocesses input data files, merges them, and computes reference-based metrics if requested.

    This function reads multiple input CSV files (excluding those with "single" in their path),
    combines them into a single dataframe, and can calculate differences between models and
    baseline/reference models.

    Args:
        input_files: List of input file paths to process
        y_label: Label for the y-axis in visualizations (default: "Diff. to reference")
        comp_to_ref: Whether to compute differences against reference models (default: True)

    Returns:
        Tuple containing:
        - processed and melted dataframe with metrics
        - raw metrics dataframe (if comp_to_ref is True, otherwise None)
    """
    input_files = [x for x in input_files if "single" not in str(x)]

    def _read_file(file: Path) -> pd.DataFrame:
        tmp = pd.read_csv(file, index_col=0)
        if str(file).endswith("clinical.csv"):
            tmp["model"] = tmp["model"] + "_clinical"
        return tmp

    dfs = [_read_file(file) for file in input_files if "sgl" not in str(file)]
    df = pd.concat(dfs)
    df_avg = (
        df.groupby(["model", "comparison", "seed"])
        .mean()
        .drop("split", axis=1)
        .reset_index()
    )
    raw = None
    if comp_to_ref:
        base = df_avg.loc[df_avg["model"].isin(BASE_MODELS), :]
        base["model_type"] = base["model"].map(lambda x: x.split("_")[0])
        base.rename({"auroc": "auroc_base", "aupr": "aupr_base"}, axis=1, inplace=True)
        models = df_avg.loc[~df_avg["model"].isin(BASE_MODELS), :]
        models["model_type"] = models["model"]
        metrics = pd.merge(
            models,
            base.drop(columns=["model"]),
            on=["comparison", "seed", "model_type"],
            how="outer",
        )
        metrics["referenced_aupr"] = metrics["aupr"] - metrics["aupr_base"]
        metrics["referenced_auroc"] = metrics["auroc"] - metrics["auroc_base"]
        df_melt = metrics.melt(
            id_vars=["model", "seed", "comparison"],
            value_vars=["referenced_auroc", "referenced_aupr", "auroc", "aupr"],
        )
        raw = df_avg.melt(
            id_vars=["model", "seed", "comparison"],
            value_vars=["auroc", "aupr"],
        )
    else:
        df_melt = df_avg.melt(
            id_vars=["model", "seed", "comparison"],
            value_vars=["auroc", "aupr"],
        )
        y_label = "Value"

    df_melt = prepare_df(y_label, df_melt)
    if raw is not None:
        raw = prepare_df("auroc", raw)
    return df_melt, raw


def prepare_df(y_label: str, df_melt: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataframe for visualization by transforming class comparison labels
    and renaming columns.

    Args:
        y_label: Label for the y-axis in visualizations
        df_melt: The melted dataframe to prepare

    Returns:
        The prepared dataframe with transformed comparisons and renamed columns
    """
    class_names = []
    for cls_instance in df_melt["comparison"]:
        a, b = cls_instance.split(" vs. ")
        a = int(a) + 1
        b = int(b) + 1
        a_trans = NEW_GROUPS[a]
        b_trans = NEW_GROUPS[b]
        class_names.append(f"{a_trans}\nvs.\n{b_trans}")
    df_melt["Comparison"] = pd.Categorical(
        values=class_names,
        categories=[
            "HC\nvs.\nPreAD",
            "HC\nvs.\nProAD",
            "HC\nvs.\nMildAD",
            "PreAD\nvs.\nProAD",
            "PreAD\nvs.\nMildAD",
            "ProAD\nvs.\nMildAD",
            "HC\nvs.\nAD Spectrum",
            "HC\nvs.\nProAD+MildAD",
        ],
    )

    df_melt.rename(
        {"variable": "Metric", "value": y_label, "model": "Model"},
        inplace=True,
        axis=1,
    )
    df_melt["Model"] = df_melt["Model"].map(lambda x: MODEL_NAMES.get(x))
    return df_melt


def data_preprocessing_batch(
    input_files: List[Path],
    y_label: str = "Diff. to reference",
    comp_to_ref: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses multiple datasets for comparison, combining them into a single dataframe
    and calculating reference differences if specified.

    Args:
        input_files: List of input file paths
        y_label: Label for the y-axis in visualizations (default: "Diff. to reference")
        comp_to_ref: Whether to compare to reference/base models (default: True)

    Returns:
        Tuple containing:
        - processed and melted dataframe with comparison metrics
        - raw metrics dataframe (if comp_to_ref is True, otherwise None)
    """

    def _read_file(file: Path) -> pd.DataFrame:
        dataset = str(file).split("/")[2]
        tmp = pd.read_csv(file, index_col=0)
        tmp["dataset"] = dataset
        return tmp

    dfs = [_read_file(file) for file in input_files if "sgl" not in str(file)]
    df = pd.concat(dfs)
    df_avg = (
        df.groupby(["model", "comparison", "seed", "dataset"])
        .mean()
        .drop("split", axis=1)
        .reset_index()
    )
    raw = None
    if comp_to_ref:
        base = df_avg.loc[df_avg["model"].isin(BASE_MODELS), :]
        base["model_type"] = base["model"].map(lambda x: x.split("_")[0])
        base.rename({"auroc": "auroc_base", "aupr": "aupr_base"}, axis=1, inplace=True)
        models = df_avg.loc[~df_avg["model"].isin(BASE_MODELS), :]
        models["model_type"] = models["model"]
        metrics = pd.merge(
            models,
            base.drop(columns=["model"]),
            on=["comparison", "seed", "model_type", "dataset"],
            how="outer",
        )
        metrics["referenced_aupr"] = metrics["aupr"] - metrics["aupr_base"]
        metrics["referenced_auroc"] = metrics["auroc"] - metrics["auroc_base"]
        df_melt = metrics.melt(
            id_vars=["model", "seed", "comparison", "dataset"],
            value_vars=["referenced_auroc", "referenced_aupr", "auroc", "aupr"],
        )
        raw = df_avg.melt(
            id_vars=["model", "seed", "comparison", "dataset"],
            value_vars=["auroc", "aupr"],
        )
    else:
        df_melt = df_avg.melt(
            id_vars=["model", "seed", "comparison", "dataset"],
            value_vars=["auroc", "aupr"],
        )
        y_label = "Value"

    df_melt = prepare_batch_df(y_label, df_melt)
    if raw is not None:
        raw = prepare_batch_df("auroc", raw)
    return df_melt, raw


def prepare_batch_df(y_label: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the dataframe by transforming comparison labels and renaming columns for batch visualization.

    This function converts the numerical class comparisons to human-readable labels,
    renames columns for consistency, and maps model names to their display names.

    Args:
        y_label: Label for the y-axis in visualizations
        data: The melted dataframe to prepare

    Returns:
        The prepared dataframe with transformed comparisons and renamed columns
    """
    class_names = []
    for cls_instance in data["comparison"]:
        a, b = cls_instance.split(" vs. ")
        a_trans = NEW_GROUPS[int(a) + 1]
        b_trans = NEW_GROUPS[int(b) + 1]
        class_names.append(f"{a_trans}\n&\n{b_trans}")
    data["Comparison"] = class_names
    data.rename(
        {
            "variable": "Metric",
            "value": y_label,
            "model": "Model",
            "dataset": "Device",
        },
        inplace=True,
        axis=1,
    )
    data["Model"] = data["Model"].map(lambda x: MODEL_NAMES.get(x))
    return data


if __name__ == "__main__":
    app()
