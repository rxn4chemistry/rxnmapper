import itertools

import pytest

from rxnmapper.batched_mapper import BatchedMapper


@pytest.fixture(scope="module")
def batched_mapper() -> BatchedMapper:
    """
    Fixture to get the RXNMapper, cached with module scope so that the weights
    do not need to be loaded multiple times.
    """
    return BatchedMapper(batch_size=4, canonicalize=False)


def test_normal_behavior(batched_mapper: BatchedMapper) -> None:
    # Simple example with 5 reactions, given in different RXN formats
    rxns = [
        "CC[O-]~[Na+].BrCC>>CCOCC",
        "CCC[O-]~[Na+].BrCC>>CCOCCC",
        "CC[O-].[Na+].BrCC>>CCOCC |f:0.1|",
        "NCC[O-]~[Na+].BrCC>>NCCOCC",
        "C(C)C[O-]~[Na+].BrCC>>C(C)COCC",
    ]

    results = batched_mapper.map_reactions(rxns)

    assert list(results) == [
        "[CH3:5][CH2:4][O-:3]~[Na+].Br[CH2:2][CH3:1]>>[CH3:1][CH2:2][O:3][CH2:4][CH3:5]",
        "[CH3:6][CH2:5][CH2:4][O-:3]~[Na+].Br[CH2:1][CH3:2]>>[CH3:1][CH2:2][O:3][CH2:4][CH2:5][CH3:6]",
        "Br[CH2:2][CH3:1].[CH3:5][CH2:4][O-:3].[Na+]>>[CH3:1][CH2:2][O:3][CH2:4][CH3:5] |f:1.2|",
        "[NH2:1][CH2:2][CH2:3][O-:4]~[Na+].Br[CH2:5][CH3:6]>>[NH2:1][CH2:2][CH2:3][O:4][CH2:5][CH3:6]",
        "[CH2:1]([CH3:2])[CH2:3][O-:4]~[Na+].Br[CH2:5][CH3:6]>>[CH2:1]([CH3:2])[CH2:3][O:4][CH2:5][CH3:6]",
    ]


def test_error(batched_mapper: BatchedMapper) -> None:
    # When there is an error, the placeholder is returned instead
    rxns = [
        "CC[O-]~[Na+].BrCC>>CCOCC",
        ".".join(itertools.repeat("ClCCl", 200))
        + ".CCC[O-]~[Na+].BrCC>>CCOCCC",  # too long
        "CC[O-].[Na+].BrCC>>CCOCC |f:0.1|",
        "NCC[O-]~[Na+].BrCC>>NCCOCC",
        "AAgCC[O-]~[Na+].BrCC>>C(C)COCC",  # invalid symbol
    ]

    results = batched_mapper.map_reactions(rxns)

    assert list(results) == [
        "[CH3:5][CH2:4][O-:3]~[Na+].Br[CH2:2][CH3:1]>>[CH3:1][CH2:2][O:3][CH2:4][CH3:5]",
        ">>",
        "Br[CH2:2][CH3:1].[CH3:5][CH2:4][O-:3].[Na+]>>[CH3:1][CH2:2][O:3][CH2:4][CH3:5] "
        "|f:1.2|",
        "[NH2:1][CH2:2][CH2:3][O-:4]~[Na+].Br[CH2:5][CH3:6]>>[NH2:1][CH2:2][CH2:3][O:4][CH2:5][CH3:6]",
        ">>",
    ]


# TODO: assert that does it lazily?
# TODO: check when error raised!
