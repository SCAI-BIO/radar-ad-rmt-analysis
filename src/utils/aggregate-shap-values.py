from pathlib import Path
from typing import List, Union
import typer
import pandas as pd
import numpy as np
from src.definitions import NEW_GROUPS

PATTERN = r"(rf|xgb|lr|cart)/([\w-]+)_scores"

app = typer.Typer()


def mean_abs(x: Union[np.ndarray, pd.Series]) -> np.floating:
    """
    Calculate the mean of absolute values.

    Args:
        x: Input array or pandas Series.

    Returns:
        Mean of absolute values of x.
    """
    return np.mean(np.abs(x))

def read_csv(x: Path) -> pd.DataFrame:
    """
    Read a CSV file and extract model and dataset information from the file path.

    Args:
        x: Path object pointing to a CSV file.

    Returns:
        pandas DataFrame with additional 'model' and 'dataset' columns extracted from the path.
    """
    df = pd.read_csv(x)
    is_gs = "scores_gs" in str(x)
    is_base = "scores_base_shap" in str(x)

    split = str(x).split("/")
    model = split[3]
    dataset = split[2]
    if is_base:
        model += "_base"

    unique_model_names = df["model"].unique()
    assert len(unique_model_names) == 1 and model == unique_model_names[0], (
        f"Unique: {unique_model_names}, assumed: {model}, file: {str(x)}"
    )

    if is_gs:
        df["model"] += "_gs"
    elif "base" not in unique_model_names[0] and is_base:
        df["model"] += "_base"

    df["dataset"] = dataset
    return df


@app.command()
def aggregate_shap_values(files: List[Path], output: str, use_absolute: bool = False):
    """
    Aggregate SHAP values from multiple CSV files.

    Reads multiple CSV files containing SHAP values, aggregates them,
    and writes the aggregated results to a CSV file.

    Args:
        files: List of Path objects pointing to CSV files with SHAP values.
        output: Path where the aggregated data should be saved.
        use_absolute: Flag to use absolute values during aggregation.
    """

    # Read and concatenate all CSV files into a single DataFrame
    dfs = [read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    # Potentially remove Unnamed column
    df.drop(columns="Unnamed: 0", inplace=True)

    relevant_cols = [
        k
        for k in df.columns
        if k not in ["comparison", "model", "split", "seed", "dataset", "record_id"]
    ]

    # Aggregate over seeds (only for test)
    agg_dict = {k: "mean" if not use_absolute else mean_abs for k in relevant_cols}
    agg_per_subj = (
        df.groupby(["dataset", "comparison", "model", "record_id"])
        .agg(agg_dict)
        .reset_index()
    )

    # Aggregate across different seeds (mean and mean absolute)
    agg_final = (
        agg_per_subj.groupby(["dataset", "comparison", "model"])
        .agg({k: [mean_abs, "mean"] for k in relevant_cols})
        .reset_index()
    )

    # Melt df
    agg_melt = (
        agg_final.melt(id_vars=["dataset", "comparison", "model"])
        .rename(
            {
                "variable_0": "feature",
                "variable_1": "agg_type",
            },
            axis=1,
        )
        .pivot_table(
            index=["dataset", "comparison", "model", "feature"],
            columns="agg_type",
            values="value",
        )
        .reset_index()
        .rename(columns={"mean_abs": "mean_abs_value", "mean": "mean_value"})
    )
    agg_sorted = agg_melt.sort_values(
        ["dataset", "comparison", "model", "mean_abs_value"],
        ascending=[True, True, True, False],
    )
    agg_sorted["rank"] = agg_sorted.groupby(["dataset", "comparison", "model"])[
        "mean_abs_value"
    ].rank(method="first", ascending=False)

    # Write the final aggregated DataFrame to CSV
    agg_sorted.to_csv(output, index=False)


# First step to merge the SHAP values with the scores
@app.command()
def complete_merge(
    input_files: List[Path],
    score_file: Path,
    output_file: Path,
):
    """
    Merge SHAP values with model scores from different files.

    This function reads SHAP value files, combines them, and merges with model scores
    from a score file. It adds a 'type' column to identify different model types (base, rmt, gs)
    and ensures the merge preserves the original data shape.

    Args:
        input_files: List of Path objects pointing to CSV files with SHAP values
        score_file: Path to the CSV file containing model scores
        output_file: Path where the merged data should be saved
    """
    dfs = [pd.read_csv(f) for f in input_files]
    df = pd.concat(dfs, ignore_index=True).drop(columns="Unnamed: 0")
    df["type"] = df["model"].map(
        lambda x: "base" if "base" in x else "rmt" if "_gs" not in x else "gs"
    )

    score_df = pd.read_csv(score_file).drop(columns="Unnamed: 0")
    score_df["type"] = score_df["model"].map(
        lambda x: "base" if "base" in x else "rmt" if "_gs" not in x else "gs"
    )
    prior_shape = df.shape
    df = df.merge(
        score_df,
        on=["split", "seed", "model", "type", "comparison"],
        how="left",
    )

    assert df.shape[0] == prior_shape[0]
    df.to_csv(output_file, index=False)


@app.command()
def filter(input_files: List[Path], output_file: Path):
    """
    Filter and aggregate SHAP values from multiple input files.

    This function reads multiple CSV files containing SHAP values, filters them based on model
    performance rankings, aggregates the values, and saves the result to an output file.
    It specifically keeps only the top-performing models (rank=1) for each comparison type.

    Args:
        input_files: List of Path objects pointing to CSV files with SHAP values
        output_file: Path where the filtered data should be saved
    """
    shap_df = pd.concat([pd.read_csv(x) for x in input_files], ignore_index=True)

    metric_df = shap_df.loc[
        :, ["comparison", "model", "type", "split", "seed", "auroc"]
    ].drop_duplicates()
    seed_performance = (
        metric_df.groupby(["model", "comparison", "type", "seed"])
        .aggregate({"auroc": "mean"})
        .reset_index()
    )
    ncv_performance = (
        seed_performance.groupby(["model", "comparison", "type"])
        .aggregate({"auroc": "mean"})
        .reset_index()
    )
    ncv_performance["rank"] = ncv_performance.groupby(["comparison", "type"])[
        "auroc"
    ].rank(method="first", ascending=False)

    # average shap values over repeats
    shap_df = shap_df.drop(columns=["auroc", "aupr", "split"])
    agg_cols = ["record_id", "comparison", "model", "type"]
    other_cols = [x for x in shap_df.columns if x not in agg_cols and x != "seed"]
    shap_agg = (
        shap_df.drop(columns="seed")
        .groupby(agg_cols)
        .aggregate({x: "mean" for x in other_cols})
        .reset_index()
    )
    merged = pd.merge(
        left=shap_agg,
        right=ncv_performance,
        on=["model", "comparison", "type"],
        how="left",
    )
    merged = merged.loc[merged["rank"] == 1.0]
    merged.to_csv(output_file)


@app.command()
def final(files: List[Path], y: Path):
    """
    Process and finalize SHAP values from multiple files.

    This function reads multiple CSV files containing SHAP values, processes them to calculate
    mean absolute values, ranks features by importance, and merges additional performance metrics.
    The resulting aggregated data is saved to the specified output file with transformed comparison names.

    Args:
        files: List of Path objects pointing to CSV files with processed SHAP values
        y: Path where the final aggregated data should be saved
    """
    def read_csv(x):
        split_rmt = str(x).split("/")[-1].split("_")
        rmt = split_rmt[1]
        if rmt == "gait":
            rmt += "_" + split_rmt[2]
        df = pd.read_csv(x, index_col=0)
        df["rmt"] = rmt

        melt_df = df.melt(
            id_vars=[
                "record_id",
                "comparison",
                "model",
                "type",
                "auroc",
                "rank",
                "rmt",
            ]
        )

        number_of_samples_per_record = melt_df.groupby("record_id").size()
        print(number_of_samples_per_record.value_counts())
        agg = (
            melt_df.groupby(["comparison", "model", "type", "rmt", "variable"])
            .aggregate({"value": mean_abs})
            .rename(columns={"value": "meanAbs"})
            .reset_index()
        )
        agg["featureRank"] = agg.groupby(["comparison", "type", "model", "rmt"])[
            "meanAbs"
        ].rank(method="first", ascending=False)
        agg.sort_values(
            ["comparison", "featureRank"],
            ascending=[True, True],
            inplace=True,
        )

        auroc_subset = df.loc[
            :, ["comparison", "model", "type", "auroc"]
        ].drop_duplicates()
        nrow_before = agg.shape[0]
        agg = agg.merge(auroc_subset, on=["comparison", "model", "type"], how="left")
        assert nrow_before == agg.shape[0]
        return agg

    df = pd.concat([read_csv(x) for x in files], ignore_index=True)
    class_names = []
    for cls_instance in df["comparison"]:
        a, b = cls_instance.split(" vs. ")
        a_trans = NEW_GROUPS[int(a) + 1]
        b_trans = NEW_GROUPS[int(b) + 1]
        class_names.append(f"{a_trans} vs. {b_trans}")
    df.insert(0, "Comparison", class_names)
    df.to_csv(y, index=False)


if __name__ == "__main__":
    app()
