from typing import Any, Dict, Iterable

import numpy as np


def assert_correct_map(value_1: Dict[str, Any], value_2: Dict[str, Any]) -> None:
    mandatory_keys = ["mapped_rxn", "confidence"]

    # Exact matches
    for key in ["mapped_rxn", "pxr_mapping_vector", "tokens"]:
        if key not in mandatory_keys and key not in value_1:
            continue
        assert value_1[key] == value_2[key]

    # close match on single number
    for key in ["confidence"]:
        if key not in mandatory_keys and key not in value_1:
            continue
        assert np.isclose(value_1[key], value_2[key])

    # close match on multiple values
    for key in [
        "pxr_confidences",
        "pxrrxp_attns",
        "tokensxtokens_attns",
        "mapping_tuples",
    ]:
        if key not in mandatory_keys and key not in value_1:
            continue
        assert np.allclose(value_1[key], value_2[key], rtol=1e-4, atol=1e-7)


def assert_correct_maps(
    values_1: Iterable[Dict[str, Any]], values_2: Iterable[Dict[str, Any]]
) -> None:
    for value_1, value_2 in zip(values_1, values_2):
        assert_correct_map(value_1, value_2)
