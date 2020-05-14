# Unsupervised attention-guided atom-mapping
Enable robust atom mapping on valid reaction SMILES. The atom-mapping information was learned by an ALBERT model trained in an unsupervised fashion on a large dataset of chemical reactions.

- [PrePrint](http://dx.doi.org/10.26434/chemrxiv.12298559)
- [Demo](http://rxnmapper.ai/demo.html)
- [Documentation](https://rxn4chemistry.github.io/rxnmapper/)

<iframe src="https://ibm.ent.box.com/embed/s/ux4fikraz5if1f93x1jje2w8at0p0k61?sortColumn=date&view=list" width="500" height="400" frameborder="0" allowfullscreen webkitallowfullscreen msallowfullscreen></iframe>

## Installation
For all installations, we recommend using `conda` to get the necessary `rdkit` dependency:

### From pip
```console
conda create -n rxnmapper python=3.6 -y
conda activate rxnmapper
conda install -c rdkit rdkit
pip install rxnmapper
```

### From github
You can install the package and setup the environment directly from github using:

```console
git clone https://github.com/rxn4chemistry/rxnmapper.git 
cd rxnmapper
conda env create -f environment.yml
conda activate rxnmapper
pip install -e .
pip install -r requirements.txt
```

## Usage

### Basic usage

```python
from rxnmapper import RXNMapper
rxn_mapper = RXNMapper()
rxns = ['CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F', 'C1COCCO1.CC(C)(C)OC(=O)CONC(=O)NCc1cccc2ccccc12.Cl>>O=C(O)CONC(=O)NCc1cccc2ccccc12']
results = rxn_mapper.get_attention_guided_atom_maps(rxns)
```

The results contain the mapped reactions and confidence scores:

```python
[{'mapped_rxn': 'CN(C)C=O.F[c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11].O=C([O-])[O-].[CH3:1][CH:2]([CH3:3])[SH:4].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]',
  'confidence': 0.9565619900376546},
 {'mapped_rxn': 'C1COCCO1.CC(C)(C)[O:3][C:2](=[O:1])[CH2:4][O:5][NH:6][C:7](=[O:8])[NH:9][CH2:10][c:11]1[cH:12][cH:13][cH:14][c:15]2[cH:16][cH:17][cH:18][cH:19][c:20]12.Cl>>[O:1]=[C:2]([OH:3])[CH2:4][O:5][NH:6][C:7](=[O:8])[NH:9][CH2:10][c:11]1[cH:12][cH:13][cH:14][c:15]2[cH:16][cH:17][cH:18][cH:19][c:20]12',
  'confidence': 0.9704424331552834}]
```

### Testing

You can run the examples above with the test suite as well:

1. `pip install -r dev_requirements.txt` 
2. `pytest tests` from the root 

## Examples

To learn more see the [examples](./examples).

## Data 

Data can be found at: https://ibm.box.com/v/RXNMapperData

## Citation

```
@article{Schwaller2020Unsupervised,
author = "Philippe Schwaller and Benjamin Hoover and Jean-Louis Reymond and Hendrik Strobelt and Teodoro Laino",
title = "{Unsupervised Attention-Guided Atom-Mapping}",
year = "2020",
month = "5",
url = "https://chemrxiv.org/articles/Unsupervised_Attention-Guided_Atom-Mapping/12298559",
doi = "10.26434/chemrxiv.12298559.v1"
}
```