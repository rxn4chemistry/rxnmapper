# Extraction of organic chemistry grammar from unsupervised learning of chemical reactions
Enable robust atom mapping on valid reaction SMILES. The atom-mapping information was learned by an ALBERT model trained in an unsupervised fashion on a large dataset of chemical reactions.

- [Extraction of organic chemistry grammar from unsupervised learning of chemical reactions](https://advances.sciencemag.org/content/7/15/eabe4166): peer-reviewed Science Advances publication (open access).
- [Demo](http://rxnmapper.ai/demo.html): give RXNMapper a try! 
- [Unsupervised attention-guided atom-mapping preprint](http://dx.doi.org/10.26434/chemrxiv.12298559): presented at the ML Interpretability for Scientific Discovery ICML workshop, 2020.

## Installation

### From pip
```console
conda create -n rxnmapper python=3.6 -y
conda activate rxnmapper
pip install rxnmapper
```

### From github
You can install the package and setup the environment directly from github using:

```console
git clone https://github.com/rxn4chemistry/rxnmapper.git 
cd rxnmapper
conda create -n rxnmapper python=3.6 -y
conda activate rxnmapper
pip install -e .
```

### RDkit

In both installation settings above, the `RDKit` dependency is not installed automatically, unless you include the extra when installing: `pip install "rxmapper[rdkit]"`.
It can also be installed via Conda or Pypi:

```bash
# Install RDKit from Conda
conda install -c conda-forge rdkit

# Install RDKit from Pypi
pip install rdkit
# for Python<3.7
# pip install rdkit-pypi
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

1. In your Conda environment: `pip install -e .[dev]`
2. `pytest tests` from the root 

## Examples

To learn more see the [examples](./examples).

## Data 

Data can be found at: https://ibm.box.com/v/RXNMapperData

## Citation

```
@article{schwaller2021extraction,
  title={Extraction of organic chemistry grammar from unsupervised learning of chemical reactions},
  author={Schwaller, Philippe and Hoover, Benjamin and Reymond, Jean-Louis and Strobelt, Hendrik and Laino, Teodoro},
  journal={Science Advances},
  volume={7},
  number={15},
  pages={eabe4166},
  year={2021},
  publisher={American Association for the Advancement of Science}
}
```
