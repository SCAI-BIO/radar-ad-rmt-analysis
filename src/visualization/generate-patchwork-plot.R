#!/usr/bin/env Rscript

# Load necessary libraries
library(ggplot2)
library(patchwork)
library(jsonlite)
library(dplyr)
library(readr)

DEVICE_NAMES <- c(
    "altoida" = "Altoida (CDS)",
    "altoidaDns" = "Altoida (DNS)",
    "banking" = "Banking",
    "iadl" = "A-iADL",
    "fitbitReduced" = "Fitbit",
    "fitbit" = "Fitbit",
    "gait_dual" = "Physilog (Dual)",
    "gait_tug" = "Physilog (TUG)",
    "axivity" = "Axivity",
    "clinical" = "In-Clinic Assessment (Questionnaire)",
    "mezurio" = "Mezurio",
    "gs" = "FDS",
    "altoida_gs" = "Altoida (CDS) (+QS)",
    "banking_gs" = "Banking (+QS)",
    "fitbitReduced_gs" = "Fitbit (+QS)",
    "fitbit_gs" = "Fitbit (+hourly) (+QS)",
    "gait_dual_gs" = "Physilog (Dual) (+QS)",
    "gait_tug_gs" = "Physilog (TUG) (+QS)",
    "axivity_gs" = "Axivity (+QS)",
    "clinical_gs" = "In-Clinic Assessment (Questionnaire) (+QS)",
    "mezurio_gs" = "Mezurio (+QS)"
)

read_data <- function(shap_file, selection_json) {
    shap_data <- read_csv(shap_file) %>%
        select(Comparison, model, type, rmt, variable, meanAbs, featureRank) %>%
        distinct()
    selection_data <- fromJSON(selection_json)
    return(list(shap_data = shap_data, selection_data = selection_data))
}

gen_subplot <- function(df, comp, device, n = 10) {
    sub <- df %>%
        filter(Comparison == comp) %>%
        filter(rmt == device) %>%
        filter(type == "rmt") %>%
        filter(featureRank <= n)

    p <- ggplot(sub, aes(x = reorder(variable, meanAbs, .fun = function(x) {
        sort(x, decreasing = TRUE)
    }), y = meanAbs)) +
        geom_bar(stat = "identity") +
        coord_flip() +
        labs(x = NULL, y = "Mean Absolute SHAP Value") +
        ggtitle(DEVICE_NAMES[device]) +
        theme_bw(base_size = 18)
    return(p)
}

create_plots <- function(shap_data, selection_data, output_dir) {
    comparisons <- names(selection_data)
    available_comparison <- shap_data$Comparison %>% unique()
    comparisons_to_plot <- available_comparison[available_comparison %in% comparisons]
    print(sprintf("Comparisons to plot: %s", comparisons_to_plot))
    print(sprintf("Available comparisons: %s", available_comparison))

    for (current_comparison in sort(comparisons_to_plot)) {
        comp_list <- list()
        rmts <- selection_data[[current_comparison]]
        if (length(rmts) == 0) {
            next
        }
        sorted_rmts <- sort(rmts)
        for (current_rmt in sorted_rmts) {
            print(sprintf("Generating plot for %s, %s", current_comparison, current_rmt))
            p <- gen_subplot(shap_data, current_comparison, current_rmt)
            comp_list[[length(comp_list) + 1]] <- p
        }
        if (length(comp_list) == 0) {
            next
        }

        # generate patchwork plot
        pp <- patchwork::wrap_plots(comp_list) + plot_layout(guides = "collect", axis_titles = "collect")
        # replace space with underscore
        comparison_string <- gsub(" ", "_", current_comparison)
        # replace "."
        comparison_string <- gsub("\\.", "", comparison_string)

        ggsave(paste0(output_dir, "/shap_", comparison_string, ".png"), pp, dpi = 300, width = 7.5 * length(comp_list), height = 12)
        ggsave(paste0(output_dir, "/shap_", comparison_string, ".eps"), pp, dpi = 300, width = 7.5 * length(comp_list), height = 12)
    }
}

# Main execution
# Read command line arguments
args <- commandArgs(trailingOnly = TRUE)
shap_file <- args[1]
selection_json <- args[2]
output_dir <- args[3]

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    cat("Created output directory:", output_dir, "\n")
} else {
    cat("Output directory already exists:", output_dir, "\n")
}

data_json <- read_data(shap_file, selection_json)
shap <- data_json$shap_data
selection <- data_json$selection_data
plots <- create_plots(shap_data = shap, selection_data = selection, output_dir = output_dir)
