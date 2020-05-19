# Examples

### Mapping of a reaction data set

Requires a data set file (.csv, .tsv, .json) with a "rxn" column.

```bash
python ../scripts/run_rxnmapper_on_dataset.py  \
    --file_path example_rxns.csv \
    --output_path  out.json \
    --batch_size 8
```
