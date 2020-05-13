# Unsupervised attention-guided atom-mapping

## Install

```console
conda create -n rxnmapper python=3.6 -y
conda activate rxnmapper
conda install -c rdkit rdkit
pip install git+https://github.ibm.com/PHS/rxnmapper.git
```

## For development
You can install the environment for local development using the following:

```console
conda env create -f environment.yml
conda activate rxnmapper
pip install -e .
pip install -r requirements.txt
```

## Usage

### Basic usage

```
from rxnmapper import RXNMapper
rxn_mapper = RXNMapper()
rxns = ['CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F', 'C1COCCO1.CC(C)(C)OC(=O)CONC(=O)NCc1cccc2ccccc12.Cl>>O=C(O)CONC(=O)NCc1cccc2ccccc12']
results = rxn_mapper.get_attention_guided_atom_maps(rxns)
```
The results contain the mapped reactions and confidence scores:
```
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

