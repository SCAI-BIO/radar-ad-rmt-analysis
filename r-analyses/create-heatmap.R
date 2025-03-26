# library imports
library(ComplexHeatmap)
library(dplyr)
library(tidyr)
library(readr)

# data preparation
distinct_data <- read_csv("numeric.csv") %>%
    distinct() %>%
    arrange(RMT, Feature)

encoding <- c(
    "NA" = "NA",
    "-***" = "p<0.001 (-)",
    "-**" = "p<0.01 (-)",
    "-*" = "p<0.05 (-)",
    "NS" = "NS",
    "+*" = "p<0.05 (+)",
    "+**" = "p<0.01 (+)",
    "+***" = "p<0.001 (+)"
)

mapped_colors <- c(
    "NA" = "#FFFFFF",
    "p<0.001 (-)" = "#67001f",
    "p<0.01 (-)" = "#d6604d",
    "p<0.05 (-)" = "#f4a582",
    "NS" = "#f7f7f7",
    "p<0.05 (+)" = "#92c5de",
    "p<0.01 (+)" = "#4393c3",
    "p<0.001 (+)" = "#2166ac"
)
comp_cols <- c("HC vs. PreAD", "HC vs. ProAD", "HC vs. MildAD", "PreAD vs. ProAD", "PreAD vs. MildAD", "ProAD vs. MildAD")

data_encoded <- distinct_data
for (col in comp_cols) {
    data_encoded[, col] <- sapply(data_encoded[, col], function(x) encoding[x])
}

# create a matrix of the data
data_matrix <- as.matrix(data_encoded[, 3:ncol(data_encoded)])
row.names(data_matrix) <- data_encoded$Feature

# create color matrix with the colors
color_matrix <- matrix(mapped_colors[data_matrix], nrow = nrow(data_matrix))
row.names(color_matrix) <- data_encoded$Feature
color_matrix

# Create the heatmap
ht <- Heatmap(
    data_matrix,
    col = mapped_colors,
    name = "Significance level",
    show_row_names = TRUE,
    row_names_side = "right",
    row_names_gp = gpar(fontsize = 11),
    row_split = data_encoded$RMT,
    column_split = factor(colnames(data_matrix), levels = comp_cols),
    row_title_rot = 0,
    row_title_side = "left",
    column_title = "Comparisons",
    column_title_side = "bottom",
    column_names_rot = 90,
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    na_col = "#ffffff",
    border = TRUE,
    heatmap_legend_param = list(direction = "horizontal", title_gp = gpar(fontsize = 14), nrow = 2, title_position = "topcenter"),
    row_names_max_width = unit(8.5, "cm"),
    gap = unit(0.15, "cm"),
)

png("heatmap.png", width = 2100, height = 2970, res = 300)
draw(ht, heatmap_legend_side = "top")
dev.off()

postscript("heatmap.eps", horizontal = FALSE)
draw(ht, heatmap_legend_side = "bottom")
dev.off()
