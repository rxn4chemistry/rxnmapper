"""Contains functions needed to process reaction SMILES and their tokens"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import re
from functools import partial
from typing import Any, List, Optional, Tuple

import numpy as np
from rdkit import Chem, rdBase
from rxn.chemutils.conversion import canonicalize_smiles, smiles_to_mol
from rxn.chemutils.exceptions import InvalidSmiles
from rxn.chemutils.reaction_equation import (
    ReactionEquation,
    apply_to_compounds,
    merge_reactants_and_agents,
    sort_compounds,
)
from rxn.chemutils.reaction_smiles import parse_any_reaction_smiles
from rxn.chemutils.utils import remove_atom_mapping

LOGGER = logging.getLogger("attnmapper:smiles_utils")

# rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")

SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
BAD_TOKS = ["[CLS]", "[SEP]"]  # Default Bad Tokens


def tokenize(smiles: str) -> List[str]:
    """Tokenize a SMILES molecule or reaction"""
    regex = re.compile(SMI_REGEX_PATTERN)
    tokens = [token for token in regex.findall(smiles)]
    assert smiles == "".join(tokens)
    return tokens


def get_atom_types(smiles: str):
    """Return atomic numbers for every token in (reaction) SMILES"""
    atom_tokens = get_atom_tokens_mask(smiles)
    if ">>" in smiles:
        precursors, products = smiles.split(">>")

        atom_types = get_atom_types_smiles(precursors)
        atom_types += get_atom_types_smiles(products)

    else:
        atom_types = get_atom_types_smiles(smiles)

    token_types = []
    atom_counter = 0

    for i in atom_tokens:
        if i == 1:
            token_types.append(atom_types[atom_counter])
            atom_counter += 1
        else:
            token_types.append(-1)
    return token_types


def get_atom_types_smiles(smiles: str) -> List[int]:
    """Convert each atom in a SMILES into a list of their atomic numbers

    Args:
        smiles: SMILES representation of a molecule or set of molecules.

    Returns:
        List of atom numbers for each atom in the smiles. Reports atoms in the same order they were passed in the original SMILES
    """
    # If `smiles` is a set of molecules, it may contain a "~".
    smiles_mol = smiles_to_mol(smiles.replace("~", "."), sanitize=False)

    atom_types = [atom.GetAtomicNum() for atom in smiles_mol.GetAtoms()]

    return atom_types


def is_atom(token: str, special_tokens: List[str] = BAD_TOKS) -> bool:
    """Determine whether a token is an atom.

    Args:
        token: Token fed into the transformer model
        special_tokens: List of tokens to consider as non-atoms (often introduced by tokenizer)

    Returns:
        bool: True if atom, False if not
    """
    bad_toks = set(special_tokens)
    normal_atom = token[0].isalpha() or token[0] == "["
    is_bad = token in bad_toks
    return (not is_bad) and normal_atom


def number_tokens(tokens: List[str], special_tokens: List[str] = BAD_TOKS) -> List[int]:
    """Map list of tokens to a list of numbered atoms

    Args:
        tokens: Tokenized SMILES
        special_tokens: List of tokens to not consider as atoms

    Example:
        >>> number_tokens(['[CLS]', 'C', '.', 'C', 'C', 'C', 'C', 'C', 'C','[SEP]'])
                #=> [-1, 0, -1, 1, 2, 3, 4, 5, 6, -1]
    """
    atom_num = 0
    isatm = partial(is_atom, special_tokens=special_tokens)

    def check_atom(t):
        if isatm(t):
            nonlocal atom_num
            ind = atom_num
            atom_num = atom_num + 1
            return ind
        return -1

    out = [check_atom(t) for t in tokens]

    return out


def get_graph_distance_matrix(smiles: str):
    """
    Compute graph distance matrix between atoms. Only works for single molecules atm and not for rxns

    Args:
        smiles {[type]} -- [description]

    Returns:
        Numpy array representing the graphwise distance between each atom and every other atom in the molecular SMILES
    """
    mol = smiles_to_mol(smiles, sanitize=False)
    return Chem.GetDistanceMatrix(mol)


def get_adjacency_matrix(smiles: str):
    """
    Compute adjacency matrix between atoms. Only works for single molecules atm and not for rxns

    Args:
        smiles: SMILES representation of a molecule

    Returns:
        Numpy array representing the adjacency between each atom and every other atom in the molecular SMILES.
        Equivalent to `distance_matrix[distance_matrix == 1]`
    """

    mol = smiles_to_mol(smiles, sanitize=False)
    return Chem.GetAdjacencyMatrix(mol)


def is_mol_end(a: str, b: str) -> bool:
    """Determine if `a` and `b` are both tokens within a molecule (Used by the `group_with` function).

    Returns False whenever either `a` or `b` is a molecule delimeter (`.` or `>>`)"""
    no_dot = (a != ".") and (b != ".")
    no_arrow = (a != ">>") and (b != ">>")

    return no_dot and no_arrow


def group_with(predicate, xs: List[Any]):
    """Takes a list and returns a list of lists where each sublist's elements are
    all satisfied pairwise comparison according to the provided function.
    Only adjacent elements are passed to the comparison function

        Original implementation here: https://github.com/slavaGanzin/ramda.py/blob/master/ramda/group_with.py

        Args:
            predicate ( f(a,b) => bool): A function that takes two subsequent inputs and returns True or Fale
            xs: List to group
    """
    out = []
    is_str = isinstance(xs, str)
    group = [xs[0]]

    for x in xs[1:]:
        if predicate(group[-1], x):
            group += [x]
        else:
            out.append("".join(group) if is_str else group)
            group = [x]

    out.append("".join(group) if is_str else group)

    return out


def split_into_mols(tokens: List[str]) -> List[List[str]]:
    """Split a reaction SMILES into SMILES for each molecule"""
    split_toks = group_with(is_mol_end, tokens)
    return split_toks


def tokens_to_smiles(tokens: List[str], special_tokens: List[str]) -> str:
    """Combine tokens into valid SMILES string, filtering out special tokens

    Args:
        tokens: Tokenized SMILES
        special_tokens: Tokens to not count as atoms

    Returns:
        SMILES representation of provided tokens, without the special tokens
    """
    bad_toks = set(special_tokens)
    return "".join([t for t in tokens if t not in bad_toks])


def tokens_to_adjacency(tokens: List[str]) -> np.ndarray:
    """Convert a tokenized reaction SMILES into a giant adjacency matrix.

    Note that this is a large, sparse Block Diagonal matrix of the adjacency matrix for each molecule in the reaction.

    Args:
        tokens: Tokenized SMILES representation

    Returns:
        Numpy Array, where non-zero entries in row `i` indicate the tokens that are atom-adjacent to token `i`
    """
    from scipy.linalg import block_diag

    mol_tokens = split_into_mols(tokens)

    smiles = [
        tokens_to_smiles(mol, BAD_TOKS) for mol in mol_tokens
    ]  # Cannot process SMILES if it is a '.' or '>>'

    # Calculate adjacency matrix for reaction
    altered_smiles = [
        s for s in smiles if s not in {".", "~", ">>"}
    ]  # Only care about atoms
    adjacency_mats = [
        get_adjacency_matrix(s) for s in altered_smiles
    ]  # Or filter if we don't need to save the spot
    rxn_mask = block_diag(*adjacency_mats)
    return rxn_mask


def get_mask_for_tokens(
    tokens: List[str], special_tokens: Optional[List[str]] = None
) -> List[int]:
    """Return a mask for a tokenized smiles, where atom tokens
    are converted to 1 and other tokens to 0.

    e.g. c1ccncc1 would give [1, 0, 1, 1, 1, 1, 1, 0]

    Args:
        tokens: tokens of the reaction
        special_tokens: Any special tokens to explicitly not call an atom.
            E.g. "[CLS]" or "[SEP]". Defaults to the empty string.

    Returns:
        Binary mask as a list where non-zero elements represent atoms
    """
    if special_tokens is None:
        special_tokens = []

    check_atom = partial(is_atom, special_tokens=special_tokens)

    atom_token_mask = [1 if check_atom(t) else 0 for t in tokens]
    return atom_token_mask


def tok_mask(
    tokens: List[str], special_tokens: Optional[List[str]] = None
) -> np.ndarray:
    """Return a mask for a tokenized smiles, where atom tokens
    are converted to 1 and other tokens to 0.

    e.g. c1ccncc1 would give [1, 0, 1, 1, 1, 1, 1, 0]

    Args:
        tokens: tokens of the reaction
        special_tokens: Any special tokens to explicitly not call an atom.
            E.g. "[CLS]" or "[SEP]". Defaults to BAD_TOKS.

    Returns:
        Binary mask as a boolean numpy array where True elements represent atoms
    """
    if special_tokens is None:
        special_tokens = BAD_TOKS
    mask = get_mask_for_tokens(tokens, special_tokens=special_tokens)
    return np.array(mask).astype(bool)


def get_atom_tokens_mask(smiles: str, special_tokens: Optional[List[str]] = None):
    """Return a mask for a smiles, where atom tokens
    are converted to 1 and other tokens to 0.

    e.g. c1ccncc1 would give [1, 0, 1, 1, 1, 1, 1, 0]

    Args:
        smiles: Smiles string of reaction
        special_tokens: Any special tokens to explicitly not call an atom.
            E.g. "[CLS]" or "[SEP]". Defaults to the empty string.

    Returns:
        Binary mask as a list where non-zero elements represent atoms
    """
    tokens = tokenize(smiles)
    if special_tokens is None:
        special_tokens = []
    return get_mask_for_tokens(tokens, special_tokens)


def canonicalize_and_atom_map(smi: str, return_equivalent_atoms=False):
    """Remove atom mapping, canonicalize and return mapping numbers in order of canonicalization.

    Args:
        smi: reaction SMILES str
        return_equivalent_atoms

    Returns:

    """
    mol = smiles_to_mol(smi, sanitize=False)
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom_map = atom.GetAtomMapNum()
            atom.SetProp("atom_map", str(atom_map))
            atom.ClearProp("molAtomMapNumber")
        else:
            atom.SetProp("atom_map", str(0))
    can_smi = Chem.MolToSmiles(mol)
    order = list(
        mol.GetPropsAsDict(includePrivate=True, includeComputed=True)[
            "_smilesAtomOutputOrder"
        ]
    )

    atom_maps_canonical = [mol.GetAtoms()[idx].GetProp("atom_map") for idx in order]

    if not return_equivalent_atoms:
        return (can_smi, atom_maps_canonical)

    raise NotImplementedError


def generate_atom_mapped_reaction_atoms(
    rxn: str, product_atom_maps, expected_atom_maps=None, canonical: bool = False
) -> Tuple[ReactionEquation, List[int]]:
    """
    Generate atom-mapped reaction from unmapped reaction and
    product-2-reactant atoms mapping vector.
    Args:
        rxn: unmapped reaction, in the format that the transformer model relies on.
        product_atom_maps: product to reactant atom maps.
        expected_atom_maps: if given, return the differences.
        canonical: whether to canonicalize the resulting SMILES.

    Returns: Atom-mapped reaction

    """

    reactants, agents, products = parse_any_reaction_smiles(rxn)
    precursors_mols = [smiles_to_mol(pr, sanitize=False) for pr in reactants + agents]
    products_mols = [smiles_to_mol(prod, sanitize=False) for prod in products]

    precursors_atom_maps = []

    differing_maps = []

    product_mapping_dict = {}

    i = -1
    atom_mapped_precursors_list = []
    for precursor_mol in precursors_mols:
        for atom in precursor_mol.GetAtoms():
            i += 1
            if i in product_atom_maps:
                # atom maps start at an index of 1
                corresponding_product_atom_map = product_atom_maps.index(i) + 1
                precursors_atom_maps.append(corresponding_product_atom_map)
                atom.SetProp("molAtomMapNumber", str(corresponding_product_atom_map))

                indices = [idx for idx, x in enumerate(product_atom_maps) if x == i]

                if len(indices) > 1:
                    for idx in indices[1:]:
                        product_mapping_dict[idx] = corresponding_product_atom_map

                if expected_atom_maps is not None:
                    if (
                        i not in expected_atom_maps
                        or corresponding_product_atom_map
                        != expected_atom_maps.index(i) + 1
                    ):
                        differing_maps.append(corresponding_product_atom_map)
        atom_mapped_precursors_list.append(
            Chem.MolToSmiles(precursor_mol, canonical=canonical)
        )

    i = -1
    atom_mapped_products_list = []
    for products_mol in products_mols:
        for atom in products_mol.GetAtoms():
            i += 1
            atom_map = product_mapping_dict.get(i, i + 1)
            atom.SetProp("molAtomMapNumber", str(atom_map))
        atom_mapped_products_list.append(
            Chem.MolToSmiles(products_mol, canonical=canonical)
        )

    atom_mapped_rxn = ReactionEquation(
        atom_mapped_precursors_list, [], atom_mapped_products_list
    )

    return atom_mapped_rxn, differing_maps


def canonicalize_smi(smi: str, remove_mapping: bool = False) -> str:
    """Convert a SMILES string into its canonicalized form

    Args:
        smi: Reaction SMILES
        remove_mapping: If True, remove atom mapping information from the canonicalized SMILES output

    Raises:
        InvalidSmiles: if the SMILES string cannot be canonicalized.

    Returns:
        SMILES reaction, canonicalized, as a string
    """
    if remove_mapping:
        smi = remove_atom_mapping(smi)

    return canonicalize_smiles(smi)


def process_reaction(reaction: ReactionEquation) -> ReactionEquation:
    """
    Remove atom-mapping, move reagents to reactants and canonicalize reaction.

    Args:
        reaction: Reaction equation to process.

    Returns:
        Processed reaction.
    """
    reaction = merge_reactants_and_agents(reaction)

    try:
        canonicalize_and_remove_atom_map = partial(
            canonicalize_smi, remove_mapping=True
        )
        reaction = apply_to_compounds(reaction, canonicalize_and_remove_atom_map)
    except InvalidSmiles:
        return ReactionEquation([], [], [])

    reaction = sort_compounds(reaction)
    return reaction


def process_reaction_with_product_maps_atoms(rxn, skip_if_not_in_precursors=False):
    """
    Remove atom-mapping, move reagents to reactants and canonicalize reaction.
    If fragment group information is given, keep the groups together using
    the character defined with fragment_bond.

    Args:
        rxn: Reaction SMILES
        skip_if_not_in_precursors: accept unmapped atoms in the product (default: False)

    Returns: joined_precursors>>joined_products reaction SMILES
    """
    reactants, reagents, products = rxn.split(">")
    try:
        precursors = [canonicalize_and_atom_map(r) for r in reactants.split(".")]
        if len(reagents) > 0:
            precursors += [canonicalize_and_atom_map(r) for r in reagents.split(".")]
        products = [canonicalize_and_atom_map(p) for p in products.split(".")]
    except InvalidSmiles:
        return ""
    sorted_precursors = sorted(precursors, key=lambda x: x[0])
    sorted_products = sorted(products, key=lambda x: x[0])
    joined_precursors = ".".join([p[0] for p in sorted_precursors])
    joined_products = ".".join([p[0] for p in sorted_products])
    precursors_atom_maps = [
        i for p in sorted_precursors for i in p[1]
    ]  # could contain duplicate entries
    product_atom_maps = [
        i for p in sorted_products for i in p[1]
    ]  # could contain duplicate entries

    joined_rxn = f"{joined_precursors}>>{joined_products}"

    products_maps = []
    warnings = []

    for p_map in product_atom_maps:

        if skip_if_not_in_precursors and p_map not in precursors_atom_maps:
            products_maps.append(-1)
        elif int(p_map) == 0:
            products_maps.append(-1)
        else:
            corresponding_precursors_atom = precursors_atom_maps.index(p_map)
            if (
                corresponding_precursors_atom in products_maps
            ):  # handle equivalent atoms
                found_alternative = False
                for atom_idx, precursor_map in enumerate(precursors_atom_maps):
                    if (precursor_map == p_map) and atom_idx not in products_maps:
                        products_maps.append(atom_idx)
                        found_alternative = True
                        break
                if not found_alternative:
                    warnings.append(
                        f"Two product atoms mapped to the same precursor atom: {rxn}"
                    )
                    products_maps.append(corresponding_precursors_atom)
            else:
                products_maps.append(corresponding_precursors_atom)
    for w in list(set(warnings)):
        LOGGER.warning(w)
    return joined_rxn, products_maps
