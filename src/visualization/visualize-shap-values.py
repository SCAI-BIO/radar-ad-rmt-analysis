#!/usr/bin/env python

from matplotlib import pyplot as plt
import typer
from pathlib import Path
import pandas as pd

from plotnine import *


app = typer.Typer()

CONVERSION = {
    0: "HC",
    1: "PreAD",
    2: "ProAD",
    3: "MildAD",
    -1: "AD Stages",
    4: "ProAD+MildAD",
}


def rename_comparison(x: str) -> str:
    """
    Rename the comparison column to more readable names.

    Args:
        x (str): The comparison to rename.

    Returns:
        str: The renamed comparison.
    """
    a, b = x.split(" vs. ")
    a = CONVERSION[int(a)]
    b = CONVERSION[int(b)]
    return f"{a} vs. {b}"


@app.command()
def single_rmt(
    base_shap_file: Path,
    base_score_file: Path,
    rmt_shap_file: Path,
    rmt_score_file: Path,
    full_shap_file: Path,
    full_score_file: Path,
    merged_output_file: Path,
    rmt_plot_file: Path,
    top_n: int = 10,
    threshold: float = 0.65,
):
    """
    Generate a visualization of the SHAP values for a single RMT.

    Args:
        shap_file (Path): Path to the SHAP values file.
        score_file (Path): Path to the score file.
        output_dir (Path): Path to the output directory.
    """
    # Read the score files
    base_score_df = pd.read_csv(base_score_file, index_col=0)
    rmt_score_df = pd.read_csv(rmt_score_file, index_col=0)
    full_score_df = pd.read_csv(full_score_file, index_col=0)
    full_score_df["model"] += "_gs"
    merged_score_df = pd.concat([base_score_df, rmt_score_df, full_score_df])

    avg_per_seed = (
        merged_score_df.groupby(["model", "comparison", "seed"])
        .agg({"auroc": "mean"})
        .reset_index()
    )
    avg_per_model = avg_per_seed.groupby(["model", "comparison"]).agg({"auroc": "mean"})

    # Read the SHAP files
    base_shap_df = pd.read_csv(base_shap_file)
    rmt_shap_df = pd.read_csv(rmt_shap_file)
    full_shap_df = pd.read_csv(full_shap_file)
    merged_shap_df = pd.concat([base_shap_df, rmt_shap_df, full_shap_df])

    # Keep only rank 1 to 10 from the SHAP values
    merged_shap_df = merged_shap_df.loc[merged_shap_df["rank"] <= top_n, :].copy()

    # merge the shap and score dataframes
    merged_df = merged_shap_df.merge(
        avg_per_model, on=["model", "comparison"], how="left"
    )
    merged_df["comparison"] = (
        merged_df["comparison"].map(rename_comparison).astype("category")
    )
    merged_df.to_csv(merged_output_file, index=False)

    # Plot the SHAP values from the RMT model
    # The model column strings may not contain the suffix _gs or _base
    merged_df = merged_df.loc[~merged_df["model"].str.endswith("_base"), :]
    merged_df = merged_df.loc[~merged_df["model"].str.endswith("_gs"), :]

    # remove models with AUROC < threshold
    merged_df = merged_df.loc[merged_df["auroc"] >= threshold, :]
    unique_comparisons = merged_df["comparison"].unique()

    if merged_df.shape[0] == 0:
        # save dummy plot
        plt.figure()
        plt.savefig(str(rmt_plot_file))
        return

    plot = (
        ggplot(data=merged_df, mapping=aes(x="feature", y="mean_abs_value"))
        + geom_bar(stat="identity")
        + labs(x="Mean absolute SHAP value", y="Feature")
        + coord_flip()
        + facet_wrap("~comparison", scales="free_y", ncol=1)
        + theme_minimal(base_size=10)
    )

    # Save the plot with specified dimensions
    plt.tight_layout()
    plot.save(
        filename=str(rmt_plot_file),
        height=8 * len(unique_comparisons),
        width=10,
        dpi=300,
        units="cm",
    )


if __name__ == "__main__":
    app()
