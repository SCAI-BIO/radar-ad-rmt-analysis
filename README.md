# RADAR-AD: assessment of multiple remote monitoring technologies for early detection of Alzheimer's disease

This repository contains the code to perform analyses and generate plots for our paper [RADAR-AD: assessment of multiple remote monitoring technologies for early detection of Alzheimer's disease](https://alzres.biomedcentral.com/articles/10.1186/s13195-025-01675-0).

## Dependencies

Dependencies can be installed using [Pixi](https://pixi.sh/latest/) with:

"`bash
pixi install
pixi shell
```

## Snakemake pipeline

All scripts are in *src* and organized via [Snakemake](https://snakemake.github.io/). We cannot share the input data within this repository. Still, in principle, execution of the pipeline is possible with the following:

"`bash
snakemake -c64
```

By default, a Postgresql database is required for the hyperparameter optimization. This can be specified within the config/config.yaml file.

## R-Analyses

Some analyses were conducted independently. They are located in *r-analyses*:

- r-analyses/univariate-analysis.R: Script with the univariate analysis.
- r-analyses/create-heatmap.R: Script to generate the heatmap figure based on the output of the univariate analysis.

## Contact

Please post a GitHub issue or e-mail manuel.lentzen@scai.fraunhofer.de if you have any questions.
