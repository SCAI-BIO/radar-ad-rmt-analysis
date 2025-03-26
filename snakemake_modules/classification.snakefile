# Description
# This file contains the Snakefile for the classification module. It defines the
# rules for generating classification results and visualizations.

################################################################################
# Imports
################################################################################

from glob import glob
import re
from typing import Set
import random
from itertools import combinations
from snakemake.utils import min_version
import tarfile
from datetime import datetime

################################################################################
# Global config
################################################################################

min_version("6.0")
random.seed(42)

configfile: "config/config.yaml"
workdir: "/home/mlentzen/git/public_repos/radar-ad-public"

DATASETS = config["datasets"]
FULLDATASETS = config["datasets"] + config["reference_datasets"]
CLASSIFIERS = {"lr", "rf", "xgb", "cart"}
RANDOM_SEEDS = config["random_seeds"]

################################################################################
# Snakemake output
################################################################################

rule all:
    input:
        "reports/clf/concise_results.csv",
        "reports/clf/concise_results_median.csv",
        expand("reports/clf/{var}/{ptype}.png",
              var=FULLDATASETS,
              ptype=["roc", "roc_raw", "pr", "pr_raw"]),
        expand("reports/figures/clf_boxplot_{metric_type}.eps", metric_type=["auroc", "aupr"]),
        expand("reports/clf/{var}/{clf_type}/models/", var=FULLDATASETS, clf_type=CLASSIFIERS),
        expand("reports/clf/{dataset}/{clf_type}/{prefix}_shap_{var}.csv", clf_type=CLASSIFIERS, dataset=DATASETS, prefix=["scores", "scores_gs", "scores_base"], var=["train", "test"]),
        expand("reports/tables/rmt_shap_{var}.csv", var = ["train", "test"]),
        expand("reports/clf/{dataset}/{clf_type}/{prefix}_{agg}_shap_{var}.csv", clf_type=CLASSIFIERS, dataset=DATASETS, prefix=["scores", "scores_gs", "scores_base"], agg=["abs", "raw"], var=["train", "test"]),
        expand("reports/clf/{dataset}/{clf_type}/{prefix}_{agg}_shap_{var}.csv", clf_type=CLASSIFIERS, dataset=config["reference_datasets"], prefix=["scores", "scores_base"], agg=["abs", "raw"], var=["train", "test"]),
        expand("reports/clf/{dataset}/{clf_type}/shap_plot_{var}.png", dataset=DATASETS, clf_type=CLASSIFIERS, var=["train", "test"]),
        expand("reports/figures/shap_plots_{var}", var=["train", "test"]),
        "reports/clf/concise_results_aupr.csv",
        expand("reports/tables/rmt_shap_{var}.csv", var=["train", "test"])

################################################################################
# Module dependencies
################################################################################

module preprocessing:
    snakefile: "preprocessing.snakefile"
    config: config

use rule * from preprocessing as preprocessing_*

################################################################################
# Generate results
################################################################################

rule RmtModel:
    input:
        xdata="data/processed/{var}_x_clf_with_na.csv",
        group="data/processed/{var}_group_clf_with_na.csv",
        script_file="src/analysis/clf.py"
    output:
        scores="reports/clf/{var}/{clf_type}/{dataset}.csv-{repeat}",
        train_shap_values="reports/clf/{var}/{clf_type}/{dataset}_shap_train.csv-{repeat}",
        test_shap_values="reports/clf/{var}/{clf_type}/{dataset}_shap_test.csv-{repeat}",
    wildcard_constraints:
        dataset="(?!(importances))[a-z\-]+",
    resources:
        mem_mb=8000,
        disk_mb=8000,
        runtime="24h",
    params:
        folds=config["general"]["folds"],
        repeats=len(RANDOM_SEEDS),
        trials=lambda x: config["trials"][x.clf_type],
        single_observation_data="y",
        db_storage=config["general"]["db"]
    threads: 1
    log:
        "logs/{var}_{clf_type}{dataset,[a-z\-]+}_clf.log-{repeat}",
    shell:
        """
        PYTHONHASHSEED=42 python -m src.analysis.clf ncv {wildcards.clf_type} {input.xdata} {input.group} \
        {output} --folds={params.folds} --cores={threads} --repeats={params.repeats} \
        --trials={params.trials} --single-observation-data={params.single_observation_data} \
        --repeat {wildcards.repeat} --impute-values --data-name={wildcards.var}_rmt \
        --db-storage={params.db_storage} > {log} 2>&1
        """

rule FitRmtModel:
    input:
        xdata="data/processed/{var}_x_clf_with_na.csv",
        group="data/processed/{var}_group_clf_with_na.csv",
        script_file="src/analysis/clf.py"
    output:
        model_dir=directory("reports/clf/{var}/{clf_type}/models/")
    wildcard_constraints:
        dataset="(?!(importances))[a-z\-]+",
    resources:
        mem_mb=8000,
        disk_mb=8000,
        runtime="24h",
    params:
        folds=config["general"]["folds"],
        trials=lambda x: config["trials"][x.clf_type],
        single_observation_data="y",
        db_storage=config["general"]["db"]
    threads: 1
    log:
        "logs/{var}_{clf_type}_fit_clf.log",
    shell:
        """
        PYTHONHASHSEED=42 python -m src.analysis.clf fit fit-{wildcards.clf_type} {input.xdata} {input.group} \
        {output} --folds={params.folds} --cores={threads} \
        --trials={params.trials} --single-observation-data={params.single_observation_data} \
        --impute-values --data-name={wildcards.var}_rmt \
        --db-storage={params.db_storage} > {log} 2>&1
        """

rule FitReducedRmtModel:
    input:
        xdata="data/processed/{var}_x_clf_with_na.csv",
        group="data/processed/{var}_group_clf_with_na.csv",
        excluded_features="config/excluded_features.txt",
        script_file="src/analysis/clf.py"
    output:
        model_dir=directory("reports/clf/{var}/{clf_type}/reduced_models/")
    wildcard_constraints:
        dataset="(?!(importances))[a-z\-]+",
    resources:
        mem_mb=8000,
        disk_mb=8000,
        runtime="24h",
    params:
        folds=config["general"]["folds"],
        trials=lambda x: config["trials"][x.clf_type],
        single_observation_data="y",
        db_storage=config["general"]["db"]
    threads: 1
    log:
        "logs/{var}_{clf_type}_fit_clf.log",
    shell:
        """
        PYTHONHASHSEED=42 python -m src.analysis.clf fit fit-{wildcards.clf_type} {input.xdata} {input.group} \
        {output} --folds={params.folds} --cores={threads} \
        --trials={params.trials} --single-observation-data={params.single_observation_data} \
        --impute-values --data-name={wildcards.var}_rmt \
        --feature-exclusion-file={input.excluded_features} \
        --db-storage={params.db_storage} > {log} 2>&1
        """

rule RmtGsModel:
    input:
        xdata="data/processed/{var}_x_clf_with_na.csv",
        group="data/processed/{var}_group_clf_with_na.csv",
        gs="data/processed/gs_x_clf_with_na.csv",
        script_file="src/analysis/clf.py"
    output:
        scores="reports/clf/{var}/{clf_type}/{dataset}_gs.csv-{repeat}",
        train_shap_values="reports/clf/{var}/{clf_type}/{dataset}_gs_shap_train.csv-{repeat}",
        test_shap_values="reports/clf/{var}/{clf_type}/{dataset}_gs_shap_test.csv-{repeat}",
    wildcard_constraints:
        dataset="(?!(importances))[a-z\-]+",
        var="(?!(gs))[a-z\-\_A-Z]+"
    resources:
        mem_mb=8000,
        disk_mb=8000,
        runtime="24h",
    params:
        folds=config["general"]["folds"],
        repeats=len(RANDOM_SEEDS),
        trials=lambda x: config["trials"][x.clf_type],
        single_observation_data="y",
        db_storage=config["general"]["db"]  #"sqlite:///reports/clf/{var}/{clf_type}/{dataset}_gs_{repeat}.db"
    threads: 1
    log:
        "logs/{var}_{clf_type}{dataset,[a-z\-]+}_clf.log-{repeat}",
    shell:
        """
        PYTHONHASHSEED=42 python -m src.analysis.clf ncv {wildcards.clf_type} {input.xdata} {input.group} \
        {output} --folds={params.folds} --cores={threads} \
        --trials={params.trials} --single-observation-data={params.single_observation_data} \
        --additional-data={input.gs} \
        --repeat={wildcards.repeat} --impute-values --data-name={wildcards.var}_full \
        --db-storage={params.db_storage} > {log} 2>&1
        """

rule BaseModel:
    input:
        xdata="data/processed/{var}_x_clf_with_na.csv",
        group="data/processed/{var}_group_clf_with_na.csv",
        script_file="src/analysis/clf.py"
    output:
        scores="reports/clf/{var}/{clf_type}/{dataset}_base.csv-{repeat}",
        train_shap_values="reports/clf/{var}/{clf_type}/{dataset}_base_shap_train.csv-{repeat}",
        test_shap_values="reports/clf/{var}/{clf_type}/{dataset}_base_shap_test.csv-{repeat}",
    wildcard_constraints:
        dataset="(?!(importances))[a-z\-]+",
    resources:
        mem_mb=16000,
        disk_mb=8000,
        runtime="8h",
    params:
        folds=config["general"]["folds"],
        repeats=len(RANDOM_SEEDS),
        trials=lambda x: config["trials"][x.clf_type],
        single_observation_data="y",
        db_storage=config["general"]["db"]
    threads: 1
    log:
        "logs/{var}_{clf_type}_{dataset}_clf_base.log-{repeat}",
    shell:
        """
        PYTHONHASHSEED=42 python -m src.analysis.clf ncv {wildcards.clf_type} {input.xdata} {input.group} \
        {output} --folds={params.folds} --cores={threads} --repeats={params.repeats} \
        --trials={params.trials} \
        --only-confounders  --single-observation-data={params.single_observation_data} \
        --repeat {wildcards.repeat} --impute-values --data-name={wildcards.var}_base \
        --db-storage={params.db_storage} > {log} 2>&1
        """

################################################################################
# Process results
################################################################################

rule MergeScores:
    input:
        all_files = lambda x:[f"reports/clf/{x.data_dir}/{x.clf_type}/{x.prefix}.csv-{repeat}" for repeat in RANDOM_SEEDS],
        script_file="src/utils/merge_files.py"
    output:
        "reports/clf/{data_dir}/{clf_type}/{prefix}.csv"
    wildcard_constraints:
        prefix="(?!(shap))[a-z\-]+\_*(?!(shap))[a-z\-]*",
        data_dir="(?!(combined))[a-z\-\_A-Z]+",
    threads: 1
    resources:
        mem_mb=2000,
        disk_mb=4000,
        runtime="1h",
    shell:
        """
        python -m src.utils.merge_files {input.all_files} {output}
        """

rule MergeShap:
    input:
        all_files = lambda x:[f"reports/clf/{x.dir}/{x.clf_type}/{x.prefix}_shap_{x.var}.csv-{repeat}" for repeat in RANDOM_SEEDS],
        score_file = "reports/clf/{dir}/{clf_type}/{prefix}.csv",
        script_file="src/utils/aggregate-shap-values.py"
    output:
        "reports/clf/{dir}/{clf_type}/{prefix}_shap_{var}.csv"
    wildcard_constraints:
        prefix="[a-z\-]+",
    threads: 1
    resources:
        mem_mb=2000,
        disk_mb=4000,
        runtime="1h",
    shell:
        """
        python {input.script_file} complete-merge {input.all_files} {input.score_file} {output}
        """

rule FilterMergedShap:
    input:
        shap_files= lambda x: [f"reports/clf/{x.dir}/{clf_type}/scores_shap_{x.var}.csv" for clf_type in CLASSIFIERS],
        script_file="src/utils/aggregate-shap-values.py"
    output:
        "reports/tables/rmt_{dir}_shap_{var}.csv"
    wildcard_constraints:
        prefix="[a-z\-]+",
    threads: 1
    resources:
        mem_mb=2000,
        disk_mb=4000,
        runtime="1h",
    shell:
        """
        python {input.script_file} filter {input.shap_files} {output}
        """

rule CompleteMerge:
    input:
        shap_files=lambda x: [f"reports/tables/rmt_{dir}_shap_{x.var}.csv" for dir in DATASETS],
        script_file="src/utils/aggregate-shap-values.py"
    output:
        "reports/tables/rmt_shap_{var}.csv"
    threads: 1
    shell:
        """
        python {input.script_file} final {input.shap_files} {output}
        """


rule AggregateShapValues:
    input:
        all_files = lambda x:[f"reports/clf/{x.dir}/{x.clf_type}/{x.prefix}_shap_{x.var}.csv-{repeat}" for repeat in RANDOM_SEEDS],
        script_file="src/utils/aggregate-shap-values.py"
    output:
        "reports/clf/{dir}/{clf_type}/{prefix}_{agg}_shap_{var}.csv"
    params:
        absolute=lambda x: "--use-absolute" if x.agg == "abs" else ""
    threads: 1
    resources:
        mem_mb=2000,
        disk_mb=4000,
        runtime="1h",
    shell:
        """
        python {input.script_file} aggregate-shap-values {input.all_files} {output} {params.absolute}
        """

rule GenerateTable:
    input:
        score_files = lambda x: [f"reports/clf/{var}/{clf_type}/scores.csv" for var in FULLDATASETS for clf_type in CLASSIFIERS],
        base_files = lambda x: [f"reports/clf/{var}/{clf_type}/scores_base.csv" for var in FULLDATASETS for clf_type in CLASSIFIERS],
        combined_files = lambda x:[f"reports/clf/{var}/{clf_type}/scores_gs.csv" for var in DATASETS for clf_type in CLASSIFIERS],
        script_file="src/utils/compile_table.py"
    threads:1
    output:
        concise="reports/clf/concise_results.csv",
        full="reports/clf/full_results.csv",
        plot="reports/figures/clf_results.png"
    shell:
        """
        python -m src.utils.compile_table mean {input.score_files} {input.base_files} {input.combined_files} {output}
        """

rule GenerateTableAupr:
    input:
        score_files = [f"reports/clf/{var}/{clf_type}/scores.csv" for var in FULLDATASETS for clf_type in CLASSIFIERS],
        base_files = [f"reports/clf/{var}/{clf_type}/scores_base.csv" for var in FULLDATASETS for clf_type in CLASSIFIERS],
        combined_files = [f"reports/clf/{var}/{clf_type}/scores_gs.csv" for var in DATASETS for clf_type in CLASSIFIERS],
        script_file="src/utils/compile_table.py"
    threads:1
    output:
        concise="reports/clf/concise_results_aupr.csv",
        full="reports/clf/full_results_aupr.csv",
        plot="reports/figures/clf_results_aupr.png"
    shell:
        """
        python -m src.utils.compile_table mean {input.score_files} {input.base_files} {input.combined_files} {output} --metric="aupr"
        """

rule GenerateTableMedian:
    input:
        score_files = [f"reports/clf/{var}/{clf_type}/scores.csv" for var in FULLDATASETS for clf_type in CLASSIFIERS],
        base_files = [f"reports/clf/{var}/{clf_type}/scores_base.csv" for var in FULLDATASETS for clf_type in CLASSIFIERS],
        combined_files = [f"reports/clf/{var}/{clf_type}/scores_gs.csv" for var in DATASETS for clf_type in CLASSIFIERS],
        script_file="src/utils/compile_table.py"
    threads:1
    output:
        concise="reports/clf/concise_results_median.csv",
        full="reports/clf/full_results_median.csv",
        plot="reports/figures/clf_results_median.png"
    shell:
        """
        python -m src.utils.compile_table median {input.score_files} {input.base_files} {input.combined_files} {output}
        """

################################################################################
# visualizations
################################################################################

rule GeneratePaperPlot:
    input:
        score_files = [f"reports/clf/{var}/{clf_type}/scores.csv" for var in [x for x in FULLDATASETS if x != "clinical"] for clf_type in CLASSIFIERS],
        base_files = [f"reports/clf/{var}/{clf_type}/scores_base.csv" for var in [x for x in FULLDATASETS if x != "clinical"] for clf_type in CLASSIFIERS],
        combined_files = [f"reports/clf/{var}/{clf_type}/scores_gs.csv" for var in DATASETS for clf_type in CLASSIFIERS],
        script_file="src/visualization/generate_boxplots.py"
    threads:1
    wildcard_constraints:
        metric_type="[a-zA-Z]+"
    output:
        boxplot1="reports/figures/clf_boxplot_{metric_type}.eps",
        boxplot2="reports/figures/clf_boxplot_{metric_type}.png",
        boxplot3="reports/figures/clf_boxplot_{metric_type}_talk.png",
        boxplot4="reports/figures/clf_boxplot_{metric_type}_only_single.eps",
        boxplot5="reports/figures/clf_boxplot_{metric_type}_only_single.png",
        boxplot6="reports/figures/clf_boxplot_{metric_type}_talk_mip.png",
        boxplot7="reports/figures/clf_boxplot_{metric_type}_talk_mip_simple.png",
        boxplot8="reports/figures/clf_boxplot_{metric_type}_full.eps",
        boxplot9="reports/figures/clf_boxplot_{metric_type}_full.png",
        inv_file="reports/debug/base_investigation_{metric_type}.csv"
    shell:
        """
        python -m src.visualization.generate_boxplots {input.score_files} {input.base_files} {input.combined_files} {output} --metric-type={wildcards.metric_type}
        """


rule GenerateRocCurves:
    input:
        fall=lambda x: [
            f"reports/clf/{x.var}/{clf}/{stype}.csv" for clf in CLASSIFIERS for stype in ["scores"] #, "scores_gs"
        ],
        base=lambda x: [f"reports/clf/{x.var}/{clf}/scores_base.csv" for clf in CLASSIFIERS],
        script_file="src/visualization/visualize_clf_results.py"
    output:
        "reports/clf/{var}/{ptype}.png",
    wildcard_constraints:
        ptype="roc.*",
        var="(?!(clinical))[a-z\-\_A-Z]+"
    params:
        ptype=lambda x: "--best-model-only" if x.ptype == "roc_best" else "",
        rawtype=lambda x: "--raw-score" if x.ptype == "roc_raw" else "",
    threads: 1
    resources:
        runtime="10m",
    log:
        "logs/{var}_{ptype}_visualization_clf_roc.log",
    shell:
        """
        python -m src.visualization.visualize_clf_results roc {input.fall} {input.base} {output} \
        {params.ptype} {params.rawtype}
        """

rule GeneratePrCurves:
    input:
        fall=lambda x: [
            f"reports/clf/{x.var}/{clf}/{stype}.csv" for clf in CLASSIFIERS for stype in ["scores"] #, "scores_gs"
        ],
        base=lambda x: [f"reports/clf/{x.var}/{clf}/scores_base.csv" for clf in CLASSIFIERS],
        script_file="src/visualization/visualize_clf_results.py"
    output:
        "reports/clf/{var}/{ptype}.png",
    wildcard_constraints:
        ptype="pr.*",
        var="(?!(clinical))[a-z\-\_A-Z]+"
    params:
        ptype=lambda x: "--best-model-only" if x.ptype == "pr_best" else "",
        rawtype=lambda x: "--raw-score" if x.ptype == "pr_raw" else "",
    threads: 1
    resources:
        runtime="10m",
    log:
        "logs/{var}_{ptype}_visualization_clf_pr.log",
    shell:
        """
        python -m src.visualization.visualize_clf_results pr {input.fall} {input.base} {output} \
        {params.ptype} {params.rawtype}
        """

rule ShapPlot:
    input:
        base_shap="reports/clf/{dataset}/{clf_type}/scores_base_abs_shap_{var}.csv",
        base_score="reports/clf/{dataset}/{clf_type}/scores_base.csv",
        rmt_shap="reports/clf/{dataset}/{clf_type}/scores_abs_shap_{var}.csv",
        rmt_score="reports/clf/{dataset}/{clf_type}/scores.csv",
        full_shap="reports/clf/{dataset}/{clf_type}/scores_gs_abs_shap_{var}.csv",
        full_score="reports/clf/{dataset}/{clf_type}/scores_gs.csv",
        script_file="src/visualization/visualize-shap-values.py"
    threads:1
    output:
        merged_file="reports/clf/{dataset}/{clf_type}/shap_merged_{var}.csv",
        plot="reports/clf/{dataset}/{clf_type}/shap_plot_{var}.png",
    shell:
        """
        python {input.script_file} {input.base_shap} {input.base_score} {input.rmt_shap} {input.rmt_score} {input.full_shap} {input.full_score} {output.merged_file} {output.plot}
        """


rule ShapPatchworkPlot:
    input:
        shap_file = rules.CompleteMerge.output,
        selection_json = "config/shap_plot.json",
        script_file="src/visualization/generate-patchwork-plot.R"
    threads:1
    output:
        directory("reports/figures/shap_plots_{var}")
    shell:
        """
        Rscript {input.script_file} {input.shap_file} {input.selection_json} {output}
        """
