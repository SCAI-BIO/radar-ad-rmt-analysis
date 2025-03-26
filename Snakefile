from snakemake.utils import min_version

min_version("6.0")


configfile: "config/config.yaml"


module classification:
    snakefile:
        "snakemake_modules/classification.snakefile"
    config:
        config


use rule * from classification as classification_*


rule all:
    input:
        rules.classification_all.input,
    default_target: True
