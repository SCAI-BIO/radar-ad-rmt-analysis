################################################################################
# imports
################################################################################
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
from plotnine import (
    ggplot,
    aes,
    geom_boxplot,
    facet_wrap,
    scale_x_discrete,
    labs,
    scale_y_continuous,
    scale_color_manual,
    theme_minimal,
    theme,
    element_text,
    geom_jitter,
    geom_hline,
    element_blank,
    element_line,
    theme_bw,
)

from src.definitions import NEW_GROUPS, DEVICE_NAMES, BASE_MODELS
from src.visualization.visualize_clf_results import select_best_clf_only

NDM_STRING = "RMT+FDS"
################################################################################
# support functions
################################################################################


def calculate_breaks(plot_df, num_major=11, num_minor=21):
    """Calculate major and minor breaks for y-axis based on data range."""
    min_val = float(plot_df["Value"].min())
    max_val = float(plot_df["Value"].max())

    def get_breaks(start, stop, num, threshold):
        breaks = []
        min_trim = False
        for x in np.linspace(start, stop, num):
            x = round(x, 2)
            if not min_trim and abs(x - min_val) > threshold:
                continue
            elif not min_trim and abs(x - min_val) <= threshold:
                breaks.append(x)
                min_trim = True
            elif x - max_val > threshold:
                break
            else:
                breaks.append(x)
        return breaks

    return {
        "major": get_breaks(0, 1, num_major, 0.1),
        "minor": get_breaks(0, 1, num_minor, 0.05),
    }


def create_base_plot(plot_df, order, metric_type, model_colors, breaks):
    """Create base ggplot object with common settings."""
    return (
        ggplot(plot_df, aes("clear_names", "Value", color="types"))
        + scale_x_discrete(limits=order)
        + labs(x="", y=metric_type.upper(), color="Experiment")
        + scale_y_continuous(breaks=breaks["major"], minor_breaks=breaks["minor"])
        + scale_color_manual(values=model_colors)
        + theme_minimal(base_size=12)
    )


def add_grid_theme(base_size=12, alpha=1.0):
    """Add common grid theme settings."""
    return theme(
        axis_text_x=element_text(
            angle=45, hjust=1, size=14
        ),  # Reduce text size and angle
        legend_position="top",  # Move legend to right
        legend_box="horizontal",  # Horizontal legend
        legend_spacing=0.1,  # Reduce legend spacing
        legend_box_margin=0,  # Reduce legend margin
        panel_spacing=0.1,
        strip_text=element_text(size=14),
        panel_grid_minor_y=element_line(
            color="#d3d3d3", linetype="dashed", size=0.5, alpha=0.3
        ),
        panel_grid_major_y=element_line(color="#acacac", size=0.8, alpha=0.4),
        panel_grid_major_x=element_blank(),
    )


def add_auroc_line(color="#009E73", size=1.2, alpha=0.5):
    """Add AUROC reference line if needed."""
    return geom_hline(
        yintercept=0.5, color=color, size=size, linetype="dashed", alpha=alpha
    )


def save_plot(g, path, width, height, dpi=None):
    """Save plot with given dimensions."""
    if dpi:
        g.save(path, units="cm", width=width, height=height, dpi=dpi)
    else:
        g.save(path, units="cm", width=width, height=height)


def create_standard_boxplot(
    plot_df, order, metric_type, model_colors, paths, include_jitter=False, width=0.45
):
    """Create standard boxplot with optional jitter."""
    breaks = calculate_breaks(plot_df)

    g = (
        create_base_plot(plot_df, order, metric_type, model_colors, breaks)
        + geom_boxplot(
            fatten=1.5,
            notch=False,
            width=width,
            outlier_shape="" if include_jitter else "o",
        )
        + facet_wrap("~Comparison", ncol=2)
        + theme_bw(base_size=20)
        + add_grid_theme()
    )

    if include_jitter:
        g = g + geom_jitter(width=0.2, alpha=0.6, size=0.4)

    if metric_type == "auroc":
        g = g + add_auroc_line()

    # Save plot in multiple formats if needed
    for path in paths:
        dpi = 300 if str(path).endswith(".png") else None
        save_plot(g, path, width=30, height=29.7, dpi=dpi)

    return g

def create_comparison_boxplot(
    plot_df,
    order,
    metric_type,
    filtered_comparisons,
    base_size=16,
):
    """Create boxplot for specific comparisons."""
    filtered_df = plot_df.loc[~plot_df["Comparison"].isin(filtered_comparisons), :]

    breaks = calculate_breaks(filtered_df)

    g = (
        ggplot(filtered_df, aes("clear_names", "Value", color="types"))
        + geom_boxplot()
        + facet_wrap("~Comparison", ncol=4)  # Increase number of columns
        + scale_x_discrete(limits=order)
        + labs(x="", y=metric_type.upper(), color="Experiment")
        + scale_y_continuous(breaks=breaks["major"])
        + theme_minimal(base_size=base_size)
        + theme(
            axis_text_x=element_text(angle=45, hjust=1, size=8),
            legend_position="top",  #
            legend_box="horizontal",
            panel_spacing=0.5,
            strip_text=element_text(size=10),
            panel_grid_minor_y=element_line(color="gray", linetype="dashed", size=0.5),
            panel_grid_major_y=element_line(color="gray", size=0.8),
            panel_grid_major_x=element_blank(),
        )
    )

    if metric_type == "auroc":
        g = g + add_auroc_line()

    return g


def preprocess_data(input_files: List[Path], base_inv_file: Path) -> pd.DataFrame:
    """Preprocess model performance data from multiple input files.

    Parameters
    ----------
    input_files : List[Path]
        List of paths to CSV files containing model performance data
    base_inv_file : Path
        Path to save the invariance analysis results

    Returns
    -------
    pd.DataFrame
        Preprocessed and merged performance data containing columns:
        ['comparison', 'seed', 'dataset', 'model', 'auroc', 'aupr']
    """
    # Read and combine all input files, adding dataset column
    df = pd.concat(
        [
            pd.read_csv(file, index_col=0).assign(
                dataset=lambda x, f=file: f"{str(f).split('/')[2]}_gs"
                if str(f).split("/")[-1] == "scores_gs.csv"
                else str(f).split("/")[2]
            )
            for file in input_files
        ]
    )

    # Split data into base models and other models
    base_subset = df[df["model"].isin(BASE_MODELS)]
    other_data = df[~df["model"].isin(BASE_MODELS)]

    # Process base model data
    # base_subset = base_subset[~base_subset["dataset"].str.contains("gait|fitbit|axivity")]
    rmt_mask = base_subset["dataset"].str.match("fitbit|axivity|gait")

    def count_unique(x):
        return len(set(x))

    def abs_diff(x):
        return abs(x.max() - x.min())

    def agg_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate performance statistics."""
        return (
            df.groupby(["comparison", "seed", "split", "model"])
            .agg(
                {
                    "auroc": ["mean", "std", "count", count_unique, abs_diff],
                    "aupr": ["mean", "std", "count", count_unique, abs_diff],
                }
            )
            .reset_index()
        )

    # Create invariance analysis and save
    inv_frame = pd.concat(
        [
            agg_stats(base_subset[rmt_mask]).assign(Type="Base*"),
            agg_stats(base_subset[~rmt_mask]).assign(Type="Base"),
        ]
    )
    base_inv_file.parent.mkdir(parents=True, exist_ok=True)
    inv_frame.to_csv(str(base_inv_file), index=False)

    # Select best base models and process metrics
    base_model = select_best_clf_only(
        base_subset.melt(
            id_vars=["comparison", "split", "seed", "model", "dataset"]
        ).rename({"model": "Model"}, axis=1),
        id_vars=["comparison", "dataset"],
        agg_vars=["Model"],
        metric_var="variable",
        decision_var="auroc",
        sort_var="value",
    )

    # Mark Base/Base* types
    base_model["dataset"] = base_model["dataset"].map(
        lambda x: "Base*"
        if any(term in x for term in ["fitbit", "axivity", "gait"])
        else "Base"
    )

    # Process and aggregate base model results
    base_subset = (
        base_model.pivot_table(
            index=["comparison", "split", "seed", "dataset", "Model"],
            columns="variable",
            values="value",
        )
        .reset_index()
        .rename({"Model": "model"}, axis=1)
        .groupby(["comparison", "seed", "dataset"])
        .agg({"auroc": "mean", "aupr": "mean"})
        .reset_index()
    )

    # Aggregate other model results
    other_data = (
        other_data.groupby(["comparison", "seed", "dataset", "model"])
        .agg({"auroc": "mean", "aupr": "mean"})
        .reset_index()
    )

    # Combine and validate results
    merged = pd.concat([base_subset, other_data])

    # Verify each model has same number of seeds
    seed_counts = merged.groupby(["comparison", "dataset", "model"]).count()
    assert all(seed_counts == len(set(merged["seed"]))), (
        "Inconsistent number of seeds across models"
    )

    # Map dataset types to categories
    type_mapping = {
        "_gs": NDM_STRING,
        "Base": "Base",
        "Base*": "Base*",
        "gs": "Questionnaires & Tests",
        "iadl": "Questionnaires & Tests",
    }

    # Set types and clean names
    merged["types"] = merged["dataset"].apply(
        lambda x: next((type_mapping[k] for k in type_mapping if k in x), "RMT")
    )
    merged["clear_names"] = merged["dataset"].str.replace("_gs|_clinical", "")

    # Remove clinical data
    merged = merged[merged["clear_names"] != "clinical"]

    # Fill na values in model column with respective entries from dataset
    merged["model"] = merged["model"].fillna(merged["dataset"])
    return merged


def generate_all_boxplots(
    plot_df, order, metric_type, model_colors, NDM_STRING, output_paths
):
    """
    Generate all boxplot variations and save them to specified paths.

    Args:
        plot_df: DataFrame with the plot data
        order: List defining the order of categories
        metric_type: String specifying the metric type (e.g., 'auroc')
        model_colors: Dict mapping model types to colors
        NDM_STRING: String identifier for NDM type
        output_paths: Dict mapping plot identifiers to file paths
    """

    # Generate first set of standard boxplots (with all data)
    create_standard_boxplot(
        plot_df=plot_df,
        order=order,
        metric_type=metric_type,
        model_colors=model_colors,
        paths=[output_paths["boxplot_8"], output_paths["boxplot_9"]],
    )

    # Filter out NDM_STRING for subsequent plots
    filtered_df = plot_df.loc[plot_df["types"] != NDM_STRING, :].copy()
    # Reset categories without NDM_STRING
    filtered_df["types"] = (
        filtered_df["types"].astype("category").cat.remove_unused_categories()
    )

    # Generate standard boxplots with filtered data and jitter
    create_standard_boxplot(
        plot_df=filtered_df,
        order=order,
        metric_type=metric_type,
        model_colors=model_colors,
        paths=[output_paths["boxplot_1"], output_paths["boxplot_2"]],
        include_jitter=True,
        width=0.85,
    )

    # Generate comparison boxplot excluding specific comparisons
    comparison_plot = create_comparison_boxplot(
        plot_df=filtered_df,
        order=order,
        metric_type=metric_type,
        filtered_comparisons=[
            "PreAD vs. ProAD",
            "ProAD vs. MildAD",
            "PreAD vs. MildAD",
            "HC vs. AD Stages",
            "HC vs. ProAD+MildAD",
        ],
    )
    save_plot(comparison_plot, output_paths["boxplot_3"], width=30, height=10)

    # Generate specialized plots for HC comparisons
    hc_comparisons = ["HC vs. PreAD", "HC vs. ProAD", "HC vs. MildAD"]
    sub_df = (
        plot_df[plot_df["Comparison"].isin(hc_comparisons)]
        .query("clear_names != 'FDS' and types != @NDM_STRING")
        .copy()
    )

    # Convert to object type for modification
    sub_df["clear_names"] = sub_df["clear_names"].astype("object")
    sub_df["types"] = sub_df["types"].astype("object")

    # Update iADL type
    sub_df.loc[sub_df["clear_names"] == "iADL", "types"] = "A-iADL"

    # Convert back to category
    sub_df["clear_names"] = sub_df["clear_names"].astype("category")
    sub_df["types"] = sub_df["types"].astype("category")

    # Calculate breaks for HC comparison plots
    breaks = calculate_breaks(sub_df)

    # Generate HC comparison plot
    hc_plot = (
        ggplot(sub_df, aes("clear_names", "Value", color="types"))
        + geom_boxplot(position="dodge")
        + facet_wrap("~Comparison", ncol=3)
        + scale_x_discrete(limits=[x for x in order if x != "FDS"])
        + labs(x="", y=metric_type.upper(), color="Type")
        + scale_y_continuous(breaks=breaks["major"])
        + geom_hline(yintercept=0.5)
        + theme_minimal(base_size=18)
        + theme(
            axis_text_x=element_text(angle=45, hjust=1),
            legend_position="top",
        )
    )
    save_plot(hc_plot, output_paths["boxplot_6"], width=20, height=13)

    # Further filter for final HC comparison plot
    final_sub_df = sub_df[~sub_df["types"].isin(["Base", "Base*"])].copy()
    final_plot = (
        ggplot(final_sub_df, aes("clear_names", "Value", color="types"))
        + geom_boxplot(position="dodge")
        + facet_wrap("~Comparison", ncol=3)
        + scale_x_discrete(limits=[x for x in order if x not in ["FDS", "Base"]])
        + labs(
            x="Remote Monitoring Technology",
            y=metric_type.upper(),
            color="Type",
        )
        + scale_y_continuous(breaks=breaks["major"])
        + geom_hline(yintercept=0.5)
        + theme_minimal(base_size=18)
        + theme(
            axis_text_x=element_text(angle=45, hjust=1),
            legend_position="top",
        )
    )
    save_plot(final_plot, output_paths["boxplot_7"], width=20, height=13)

    # Generate final overview plots
    overview_plot = (
        ggplot(filtered_df, aes("clear_names", "Value", color="types"))
        + geom_boxplot(width=0.45)
        + facet_wrap("~Comparison", ncol=2)
        + scale_x_discrete(limits=order)
        + labs(x="", y=metric_type.upper(), color="Experiment")
        + scale_y_continuous(breaks=breaks["major"])
        + theme_minimal()
        + theme(
            axis_text_x=element_text(angle=90, vjust=1, hjust=0.5),
            subplots_adjust={"wspace": 0.5},
            legend_position="top",
        )
    )
    save_plot(overview_plot, output_paths["boxplot_4"], width=20, height=30)
    save_plot(overview_plot, output_paths["boxplot_5"], width=20, height=30, dpi=300)


def prepare_plotting_data(metric_type, merged):
    selection = []
    for dataset, comparison, dtype in pd.MultiIndex.from_product(
        [
            merged["dataset"].unique(),
            merged["comparison"].unique(),
            ["RMT", NDM_STRING, "FDS", "Questionnaires & Tests"],
        ]
    ):
        subset = merged.loc[
            (merged["dataset"] == dataset)
            & (merged["comparison"] == comparison)
            & (merged["types"] == dtype)
        ]
        if not subset.empty:
            best_model = subset.groupby("model")["auroc"].mean().idxmax()
            selection.append(subset[subset["model"] == best_model])

    # Combine data and prepare for plotting
    combined_data = pd.concat(
        [merged[merged["types"].isin(["Base", "Base*"])], pd.concat(selection)]
    )

    plot_df = combined_data.melt(
        id_vars=["model", "comparison", "types", "clear_names"],
        value_vars=metric_type,
        value_name="Value",
        var_name="Metric",
    )
    plot_df["clear_names"] = plot_df["clear_names"].map(
        lambda x: DEVICE_NAMES.get(x, x)
    )

    # Add comparison names
    plot_df["Comparison"] = plot_df["comparison"].apply(
        lambda x: f"{NEW_GROUPS[int(x.split(' vs. ')[0]) + 1]} vs. "
        f"{NEW_GROUPS[int(x.split(' vs. ')[1]) + 1]}"
    )

    # Handle special cases and sort
    plot_df.loc[
        (plot_df["Comparison"].str.contains("MildAD"))
        & (plot_df["clear_names"].str.contains("Altoida")),
        "Value",
    ] = np.nan

    order = ["Base", "Base*"] + combined_data[
        combined_data["types"].isin(["RMT", "FDS", "Questionnaires & Tests"])
    ].groupby(["dataset", "comparison"])["auroc"].mean().unstack().mean(
        1
    ).sort_values().index.map(lambda x: DEVICE_NAMES.get(x, x)).tolist()

    # Set categories and clean data
    plot_df["Comparison"] = pd.Categorical(
        plot_df["Comparison"],
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
    )
    plot_df["clear_names"] = pd.Categorical(plot_df["clear_names"], order)
    plot_df["types"] = plot_df["types"].astype("category")
    plot_df.dropna(inplace=True)
    return plot_df, order


################################################################################
# main
################################################################################


def main(
    input_files: List[Path],
    boxplot_path_1: Path,
    boxplot_path_2: Path,
    boxplot_path_3: Path,
    boxplot_path_4: Path,
    boxplot_path_5: Path,
    boxplot_path_6: Path,
    boxplot_path_7: Path,
    boxplot_path_8: Path,
    boxplot_path_9: Path,
    base_inv_file: Path,
    metric_type: str = "auroc",
):
    # Read all data files and combine them
    merged = preprocess_data(input_files, base_inv_file)
    plot_df, order = prepare_plotting_data(metric_type, merged)

    min_auc = float(plot_df["Value"].min())
    max_auc = float(plot_df["Value"].max())

    breaks = []
    min_trim = False
    for x in np.linspace(0, 1, 11):
        x = round(x, 2)
        if not min_trim and abs(x - min_auc) > 0.1:
            continue
        elif not min_trim and abs(x - min_auc) < 0.1:
            breaks.append(x)
            min_trim = True
        elif x - max_auc > 0.1:
            break
        else:
            breaks.append(x)

    minor_breaks = []
    min_trim = False
    for x in np.linspace(0, 1, 21):
        x = round(x, 2)
        if not min_trim and min_auc - x > 0.05:
            continue
        elif not min_trim and min_auc - x < 0.05:
            minor_breaks.append(x)
            min_trim = True
        elif x - max_auc > 0.05:
            break
        else:
            minor_breaks.append(x)

    model_colors = {
        "Base": "#0072B2",
        "Base*": "#56B4E9",
        "RMT": "#D55E00",
        "RMT+FDS": "#E69F00",
        "Questionnaires & Tests": "#CC79A7",
    }

    output_paths = {
        "boxplot_1": boxplot_path_1,
        "boxplot_2": boxplot_path_2,
        "boxplot_3": boxplot_path_3,
        "boxplot_4": boxplot_path_4,
        "boxplot_5": boxplot_path_5,
        "boxplot_6": boxplot_path_6,
        "boxplot_7": boxplot_path_7,
        "boxplot_8": boxplot_path_8,
        "boxplot_9": boxplot_path_9,
    }

    generate_all_boxplots(
        plot_df=plot_df,
        order=order,
        metric_type=metric_type,
        model_colors=model_colors,
        NDM_STRING=NDM_STRING,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    typer.run(main)
