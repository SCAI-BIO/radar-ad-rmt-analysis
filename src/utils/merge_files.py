from pathlib import Path
from typing import List

import pandas as pd
import typer


def main(input_files: List[Path], output_file: Path):
    """Merge files into one dataframe and save it to a csv file."""
    dataframes = []
    for file in input_files:
        tmp = pd.read_csv(file, index_col=0)
        if "reports/clf" not in str(file):
            # Regression case
            dataframes.append(
                tmp.groupby(["model", "seed"])
                .aggregate({"r2": "mean", "rmse": "mean"})
                .reset_index()
            )
        else:
            dataframes.append(tmp)
    pd.concat(dataframes).to_csv(output_file)


if __name__ == "__main__":
    typer.run(main)
