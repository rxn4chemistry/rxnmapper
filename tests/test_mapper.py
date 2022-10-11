import numpy as np
import pytest

from rxnmapper import RXNMapper


@pytest.fixture(scope="module")
def rxn_mapper() -> RXNMapper:
    """
    Fixture to get the RXNMapper, cached with module scope so that the weights
    do not need to be loaded multiple times.
    """
    return RXNMapper()


def is_correct_map(result, exp):
    assert result["mapped_rxn"] == exp["mapped_rxn"]
    assert np.isclose(result["confidence"], exp["confidence"])


def test_example_maps(rxn_mapper: RXNMapper):
    rxns = [
        "CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F",
        "C1COCCO1.CC(C)(C)OC(=O)CONC(=O)NCc1cccc2ccccc12.Cl>>O=C(O)CONC(=O)NCc1cccc2ccccc12",
        "C=CCN=C=S.CNCc1ccc(C#N)cc1.NNC(=O)c1cn2c(n1)CCCC2>>C=CCN1C(C2=CN3CCCCC3=N2)=NN=C1N(C)CC1=CC=C(C#N)C=C1",
    ]
    expected = [
        {
            "mapped_rxn": "[CH3:1][CH:2]([CH3:3])[SH:4].CN(C)C=O.F[c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11].O=C([O-])[O-].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]",
            "confidence": 0.9565619900376546,
        },
        {
            "mapped_rxn": "C1COCCO1.CC(C)(C)[O:3][C:2](=[O:1])[CH2:4][O:5][NH:6][C:7](=[O:8])[NH:9][CH2:10][c:11]1[cH:12][cH:13][cH:14][c:15]2[cH:16][cH:17][cH:18][cH:19][c:20]12.Cl>>[O:1]=[C:2]([OH:3])[CH2:4][O:5][NH:6][C:7](=[O:8])[NH:9][CH2:10][c:11]1[cH:12][cH:13][cH:14][c:15]2[cH:16][cH:17][cH:18][cH:19][c:20]12",
            "confidence": 0.9704424331552834,
        },
        {
            "mapped_rxn": "S=[C:17]=[N:4][CH2:3][CH:2]=[CH2:1].[NH:18]([CH3:19])[CH2:20][c:21]1[cH:22][cH:23][c:24]([C:25]#[N:26])[cH:27][cH:28]1.O=[C:5]([c:6]1[cH:7][n:8]2[c:9]([n:10]1)[CH2:11][CH2:12][CH2:13][CH2:14]2)[NH:15][NH2:16]>>[CH2:1]=[CH:2][CH2:3][n:4]1[c:5](-[c:6]2[cH:7][n:8]3[c:9]([n:10]2)[CH2:11][CH2:12][CH2:13][CH2:14]3)[n:15][n:16][c:17]1[N:18]([CH3:19])[CH2:20][c:21]1[cH:22][cH:23][c:24]([C:25]#[N:26])[cH:27][cH:28]1",
            "confidence": 0.919023506871605,
        },
    ]

    results = rxn_mapper.get_attention_guided_atom_maps(rxns)
    for res, exp in zip(results, expected):
        is_correct_map(res, exp)


def test_fragment_bond(rxn_mapper: RXNMapper):
    rxns = ["CC[O-]~[Na+].BrCC.[Na+]~[H-]>>CCOCC"]
    expected = [
        {
            "mapped_rxn": "Br[CH2:2][CH3:1].[Na+]~[O-:3][CH2:4][CH3:5].[H-]~[Na+]>>[CH3:1][CH2:2][O:3][CH2:4][CH3:5]",
            "confidence": 0.9606074439250337,
        }
    ]

    results = rxn_mapper.get_attention_guided_atom_maps(rxns)
    for res, exp in zip(results, expected):
        is_correct_map(res, exp)


def test_extended_smiles_format(rxn_mapper: RXNMapper):
    rxns = ["CC[O-].[Na+].BrCC.[Na+].[H-]>>CCOCC |f:0.1,3.4|"]
    expected = [
        {
            "mapped_rxn": "Br[CH2:2][CH3:1].[Na+].[O-:3][CH2:4][CH3:5].[H-].[Na+]>>[CH3:1][CH2:2][O:3][CH2:4][CH3:5] |f:1.2,3.4|",
            "confidence": 0.9606074439250337,
        }
    ]

    results = rxn_mapper.get_attention_guided_atom_maps(rxns)
    for res, exp in zip(results, expected):
        is_correct_map(res, exp)


def test_no_canonicalization(rxn_mapper: RXNMapper):
    rxns = ["C(C)O.BrC(C)>>CCOCC"]
    # Note that in the mapped RXN, the first reactant still has a parenthesis (which is
    # desired). The other parenthesis disappears, but there is probably not much we can
    # do here to keep it.
    expected = [
        {
            "mapped_rxn": "[CH2:2]([CH3:1])[OH:3].Br[CH2:4][CH3:5]>>[CH3:1][CH2:2][O:3][CH2:4][CH3:5]",
            "confidence": 0.9754605679009868,
        }
    ]

    results = rxn_mapper.get_attention_guided_atom_maps(rxns, canonicalize_rxns=False)
    for res, exp in zip(results, expected):
        is_correct_map(res, exp)


def test_reaction_with_invalid_valence(rxn_mapper: RXNMapper):
    # Here, "BrCFC" and "CCOCFC" are valid SMILES with invalid valence. Still,
    # the model is able to do a prediction for them.
    rxns = ["CCO.BrCFC>>CCOCFC"]
    expected = [
        {
            "mapped_rxn": "[CH3:1][CH2:2][OH:3].Br[CH2:4][F:5][CH3:6]>>[CH3:1][CH2:2][O:3][CH2:4][F:5][CH3:6]",
            "confidence": 0.9730222157275086,
        }
    ]

    results = rxn_mapper.get_attention_guided_atom_maps(rxns, canonicalize_rxns=False)
    for res, exp in zip(results, expected):
        is_correct_map(res, exp)


def test_multiple_products(rxn_mapper: RXNMapper):
    # Reverse the reaction from the previous example
    rxns = ["CCOCC.[Na+]~[Br-]>>CC[O-]~[Na+].BrCC"]
    expected = [
        {
            "mapped_rxn": "[CH3:1][CH2:2][O:6][CH2:5][CH3:4].[Br-:3]~[Na+:7]>>[CH3:1][CH2:2][Br:3].[CH3:4][CH2:5][O-:6]~[Na+:7]",
            "confidence": 0.7312865053856896,
        }
    ]

    results = rxn_mapper.get_attention_guided_atom_maps(rxns)
    for res, exp in zip(results, expected):
        is_correct_map(res, exp)
