from rxnmapper.smiles_utils import canonicalize_smi


def test_canonicalize_smi():
    assert canonicalize_smi("C(C)O") == "CCO"
    assert canonicalize_smi("[CH2:4](C)O") == "C[CH2:4]O"
    assert canonicalize_smi("[CH2:4](C)O", True) == "CCO"
