from rxnmapper.smiles_utils import (
    canonicalize_smi,
    process_reaction_with_product_maps_atoms,
)


def test_canonicalize_smi():
    assert canonicalize_smi("C(C)O") == "CCO"
    assert canonicalize_smi("[CH2:4](C)O") == "C[CH2:4]O"
    assert canonicalize_smi("[CH2:4](C)O", True) == "CCO"


def test_process_reaction_with_product_maps_atoms():
    # Two equivalent reactions in terms of atom mapping
    reaction_1 = "[F:9][CH2:5][CH2:6][CH3:7].[OH2:8]>>[F:9][CH2:5][CH2:6][CH2:7][OH:8]"
    reaction_2 = (
        "[OH2:4].[CH2:2]([CH3:1])[CH2:3][F:5]>>[OH:4][CH2:1][CH2:2][CH2:3][F:5]"
    )

    v1 = process_reaction_with_product_maps_atoms(reaction_1)
    v2 = process_reaction_with_product_maps_atoms(reaction_2)

    assert v1 == v2
    assert v1[0] == "CCCF.O>>OCCCF"
    assert v1[1] == [4, 0, 1, 2, 3]
