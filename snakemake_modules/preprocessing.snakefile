################################################################################
# Global config
################################################################################
configfile: "config/config.yaml"


FULLDATASETS = config["datasets"] + config["reference_datasets"]

################################################################################
# Snakemake output
################################################################################


rule all:
    input:
        expand(
            "data/processed/{var}_x_clf_with_na.csv",
            var=[x for x in FULLDATASETS if x != "clinical"],
        ),


################################################################################
# Rules
################################################################################


rule prepare_data:
    input:
        merged=config["data"]["merged"],
        file="src/data/process_merged.py",
    output:
        xdata=temp("data/processed/{var}_x_clf.csv"),
        group=temp("data/processed/{var}_group_clf.csv"),
        features="data/processed/{var}_features_clf.txt",
    wildcard_constraints:
        var="(?!(clinical))(?!(gs))(?!(idal))[A-Za-z\-_]*",
    threads: 1
    priority: 100
    shell:
        """
        python -m src.data.process_merged process-data '{input.merged}' {output.xdata} {output.group} \
        {output.features} {wildcards.var}
        """


rule prepare_data_with_na:
    input:
        merged=config["data"]["merged"],
        file="src/data/process_merged.py",
    output:
        xdata=temp("data/processed/{var}_x_clf_with_na.csv"),
        group=temp("data/processed/{var}_group_clf_with_na.csv"),
        features="data/processed/{var}_features_clf_with_na.txt",
    wildcard_constraints:
        var="(?!(clinical))(?!(gs))[A-Za-z\-_]*",
    threads: 1
    priority: 100
    shell:
        """
        python -m src.data.process_merged process-data '{input.merged}' {output.xdata} {output.group} \
        {output.features} {wildcards.var} --keepna
        """


rule prepare_reference_with_na:
    input:
        merged=rules.prepare_data_with_na.input.merged,
        file="src/data/process_merged.py",
    output:
        xdata="data/processed/gs_x_clf_with_na.csv",
        group="data/processed/gs_group_clf_with_na.csv",
        features="data/processed/gs_features_clf_with_na.txt",
    threads: 1
    priority: 100
    shell:
        """
        python -m src.data.process_merged generate-gold-standard '{input.merged}' {output.xdata} {output.group} \
        {output.features} --keepna
        """


rule prepare_iadl_with_na:
    input:
        merged=rules.prepare_data_with_na.input.merged,
        file="src/data/process_merged.py",
    output:
        xdata="data/processed/iadl_x_clf_with_na.csv",
        group="data/processed/iadl_group_clf_with_na.csv",
        features="data/processed/iadl_features_clf_with_na.txt",
    threads: 1
    priority: 100
    shell:
        """
        python -m src.data.process_merged generate-iadl-dataset '{input.merged}' {output.xdata} {output.group} \
        {output.features} --keepna
        """
