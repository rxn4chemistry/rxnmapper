import logging

import click
import pandas as pd
from tqdm import tqdm

from rxnmapper import RXNMapper

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--file_path", "-f", help='Input file path to csv, tsv or json with "rxn" column'
)
@click.option("--output_path", "-o", help="Output file path")
@click.option("--batch_size", "-bs", default=1, help="Batch size")
@click.option(
    "--canonicalize/--no_canonicalize",
    default=True,
    help="Canonicalize inputs (default: True)",
)
@click.option(
    "--detailed", "-d", default=False, is_flag=True, help="Return detailed output"
)
def main(
    file_path: str,
    output_path: str,
    batch_size: int,
    canonicalize: bool,
    detailed: bool,
) -> None:
    if file_path.endswith(".json"):
        df = pd.read_json(file_path)
    elif file_path.endswith(".tsv"):
        df = pd.read_csv(file_path, sep="\t")
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        ValueError("Unrecognized file type")
    df.reset_index(inplace=True)

    rxn_mapper = RXNMapper()

    results = []
    rxns = []

    for i, row in tqdm(df.iterrows(), total=len(df)):

        rxns.append(row["rxn"])

        if (i + 1) % batch_size == 0:
            results += rxn_mapper.get_attention_guided_atom_maps(
                rxns, canonicalize_rxns=canonicalize, detailed_output=detailed
            )
            rxns = []

    if rxns:
        results += rxn_mapper.get_attention_guided_atom_maps(
            rxns, canonicalize_rxns=canonicalize
        )
    mapped = [r["mapped_rxn"] for r in results]
    print("\n".join(mapped))
    results_df = pd.DataFrame(results)

    results_df.to_json(output_path)


if __name__ == "__main__":
    main()
