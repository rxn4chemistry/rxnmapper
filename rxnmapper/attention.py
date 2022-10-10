"""Attention handling and transformations

Attentions are always calculated from the tokenized SMILES. To convert this into a proper atom mapping software,
the need arises to map the tokens (which include parentheses, special tokens, and bonds) to the atom domain.

This module contains all of the helper methods needed to convert the attention matrix into the atom domain,
separating on reactants and products, including special tokens and not including special tokens, in the atom
domain / in the token domain, and accounting for adjacent atoms in molecules.
"""
import logging
from typing import List, Optional

import numpy as np

from .smiles_utils import (
    get_atom_types_smiles,
    get_mask_for_tokens,
    number_tokens,
    tokens_to_adjacency,
)

LOGGER = logging.getLogger("attnmapper:attention")


class AttentionScorer:
    def __init__(
        self,
        rxn_smiles: str,
        tokens: List[str],
        attentions: np.ndarray,
        special_tokens: List[str] = ["[CLS]", "[SEP]"],
        attention_multiplier: float = 90.0,
        mask_mapped_product_atoms: bool = True,
        mask_mapped_reactant_atoms: bool = True,
        output_attentions: bool = False,
    ):
        """Convenience wrapper for mapping attentions into the atom domain, separated by reactants and products, and introducing neighborhood locality.

        Args:
            rxn_smiles: Smiles for reaction
            tokens: Tokenized smiles of reaction of length N
            attentions: NxN attention matrix
            special_tokens: Special tokens used by the model that do not count as atoms
            attention_multiplier: Amount to increase the attention connection from adjacent atoms of a newly mapped product atom to adjacent atoms of the newly mapped reactant atom.
                Boosts the likelihood of an atom having the same adjacent atoms in reactants and products
            mask_mapped_product_atoms: If true, zero attentions to product atoms that have already been mapped
            mask_mapped_reactant_atoms: If true, zero attentions to reactant atoms that have already been mapped
            output_attentions: If true, output the raw attentions along with generated atom maps
        """
        self.rxn, self.tokens, self.attentions = rxn_smiles, tokens, attentions
        self.special_tokens = special_tokens
        self.N = len(tokens)
        self.attention_multiplier = attention_multiplier
        self.mask_mapped_product_atoms = mask_mapped_product_atoms
        self.mask_mapped_reactant_atoms = mask_mapped_reactant_atoms
        self.output_attentions = output_attentions

        try:
            self.split_ind = tokens.index(
                ">>"
            )  # Index that separates products from reactants
            self._product_inds = slice(self.split_ind + 1, self.N)
            self._reactant_inds = slice(0, self.split_ind)
        except ValueError:
            raise ValueError(
                "rxn smiles is not a complete reaction. Can't find the '>>' to separate the products"
            )

        # Mask of atoms
        self.atom_token_mask = np.array(
            get_mask_for_tokens(self.tokens, self.special_tokens)
        ).astype(bool)

        # Atoms numbered in the array
        self.token2atom = np.array(number_tokens(tokens))
        self.atom2token = {
            k: v for k, v in zip(self.token2atom, range(len(self.token2atom)))
        }

        # Adjacency graph for all tokens
        self.adjacency_matrix = tokens_to_adjacency(tokens).astype(bool)

        self._precursors_atom_types: Optional[List[int]] = None
        self._product_atom_types: Optional[List[int]] = None
        self._rnums_atoms: Optional[np.ndarray] = None
        self._pnums_atoms: Optional[np.ndarray] = None
        self._nreactant_atoms: Optional[int] = None
        self._nproduct_atoms: Optional[int] = None
        self._adjacency_matrix_products: Optional[np.ndarray] = None
        self._adjacency_matrix_precursors: Optional[np.ndarray] = None
        self._pxr_filt_atoms: Optional[np.ndarray] = None
        self._rxp_filt_atoms: Optional[np.ndarray] = None
        self._atom_type_mask: Optional[np.ndarray] = None
        self._atom_type_masked_attentions: Optional[np.ndarray] = None

        # Attention multiplication matrix
        self.attention_multiplier_matrix = np.ones_like(
            self.combined_attentions_filt_atoms
        ).astype(float)

    @property
    def atom_attentions(self) -> np.ndarray:
        """The MxM attention matrix, selected for only attentions that are from atoms, to atoms"""
        return self.attentions[self.atom_token_mask].T[self.atom_token_mask].T

    @property
    def adjacent_atom_attentions(self) -> np.ndarray:
        """The MxM attention matrix, where all attentions are zeroed if the attention is not to an adjacent atom."""
        atts = self.atom_attentions.copy()
        mask = np.logical_not(self.adjacency_matrix)
        atts[mask] = 0
        return atts

    @property
    def adjacency_matrix_reactants(self) -> np.ndarray:
        """The adjacency matrix of the reactants"""
        if self._adjacency_matrix_precursors is None:
            self._adjacency_matrix_precursors = self.adjacency_matrix[
                : len(self.rnums_atoms), : len(self.rnums_atoms)
            ]
        return self._adjacency_matrix_precursors

    @property
    def adjacency_matrix_products(self) -> np.ndarray:
        """The adjacency matrix of the products"""
        if self._adjacency_matrix_products is None:
            self._adjacency_matrix_products = self.adjacency_matrix[
                len(self.rnums_atoms) :, len(self.rnums_atoms) :
            ]
        return self._adjacency_matrix_products

    @property
    def atom_type_masked_attentions(self) -> np.ndarray:
        """Generate a"""
        if self._atom_type_masked_attentions is None:
            self._atom_type_masked_attentions = np.multiply(
                self.combined_attentions_filt_atoms, self.get_atom_type_mask()
            )
        return self._atom_type_masked_attentions

    @property
    def rxp(self) -> np.ndarray:
        """Subset of attentions relating the reactants to the products"""
        return self.attentions[: self.split_ind, (self.split_ind + 1) :]

    @property
    def rxp_filt(self) -> np.ndarray:
        """RXP without the special tokens"""
        return self.rxp[1:, :-1]

    @property
    def rxp_filt_atoms(self) -> np.ndarray:
        """RXP only the atoms, no special tokens"""
        if self._rxp_filt_atoms is None:
            self._rxp_filt_atoms = self.rxp[[i != -1 for i in self.rnums]][
                :, [i != -1 for i in self.pnums]
            ]
        return self._rxp_filt_atoms

    @property
    def pxr(self) -> np.ndarray:
        """Subset of attentions relating the products to the reactants"""
        i = self.split_ind
        return self.attentions[(i + 1) :, :i]

    @property
    def pxr_filt(self) -> np.ndarray:
        """PXR without the special tokens"""
        return self.pxr[:-1, 1:]

    @property
    def pxr_filt_atoms(self) -> np.ndarray:
        """PXR only the atoms, no special tokens"""
        if self._pxr_filt_atoms is None:
            self._pxr_filt_atoms = self.pxr[[i != -1 for i in self.pnums]][
                :, [i != -1 for i in self.rnums]
            ]
        return self._pxr_filt_atoms

    @property
    def combined_attentions(self) -> np.ndarray:
        """Summed pxr and rxp"""
        return self.pxr + self.rxp.T

    @property
    def combined_attentions_filt(self) -> np.ndarray:
        """Summed pxr_filt and rxp_filt (no special tokens)"""
        return self.pxr_filt + self.rxp_filt.T

    @property
    def combined_attentions_filt_atoms(self) -> np.ndarray:
        """Summed pxr_filt_atoms and rxp_filt_atoms (no special tokens, no "non-atom" tokens)"""
        return self.pxr_filt_atoms + self.rxp_filt_atoms.T

    @property
    def combined_attentions_filt_atoms_same_type(self) -> np.ndarray:
        """Summed pxr_filt_atoms and rxp_filt_atoms (no special tokens, no "non-atom" tokens). All attentions to atoms of a different type are zeroed"""

        atom_type_mask = np.zeros(self.combined_attentions_filt_atoms.shape)
        precursor_atom_types = get_atom_types_smiles("".join(self.rtokens[1:]))
        for i, atom_type in enumerate(
            get_atom_types_smiles("".join(self.ptokens[:-1]))
        ):
            if atom_type > 0:
                atom_type_mask[i, :] = (
                    np.array(precursor_atom_types) == atom_type
                ).astype(int)
        combined_attentions = np.multiply(
            self.combined_attentions_filt_atoms, atom_type_mask
        )
        row_sums = combined_attentions.sum(axis=1)
        normalized_attentions = np.divide(
            combined_attentions,
            row_sums[:, np.newaxis],
            out=np.zeros_like(combined_attentions),
            where=row_sums[:, np.newaxis] != 0,
        )
        return normalized_attentions

    @property
    def pnums(self) -> np.ndarray:
        """Get atom indexes for just the product tokens.

        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        return self.token2atom[(self.split_ind + 1) :]

    @property
    def pnums_filt(self) -> np.ndarray:
        """Get atom indexes for just the product tokens, without the [SEP].

        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        return self.pnums

    @property
    def pnums_atoms(self) -> np.ndarray:
        """Get atom indexes for just the product ATOMS, without the [SEP].

        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        if self._pnums_atoms is None:
            self._pnums_atoms = np.array([a for a in self.pnums if a != -1])
        return self._pnums_atoms

    @property
    def rnums(self) -> np.ndarray:
        """Get atom indexes for the reactant tokens.

        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        return self.token2atom[: self.split_ind]

    @property
    def rnums_filt(self) -> np.ndarray:
        """Get atom indexes for just the reactant tokens, without the [CLS].

        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        return self.rnums[1:]

    @property
    def rnums_atoms(self) -> np.ndarray:
        """Get atom indexes for the reactant ATOMS, without the [CLS].

        Numbers in this vector that are >= 0 are atoms, whereas indexes == -1 represent special tokens (e.g., bonds, parens, [CLS])
        """
        if self._rnums_atoms is None:
            self._rnums_atoms = np.array([a for a in self.rnums if a != -1])
        return self._rnums_atoms

    @property
    def nreactant_atoms(self) -> int:
        """The number of atoms in the reactants"""
        if self._nreactant_atoms is None:
            self._nreactant_atoms = len(self.rnums_atoms)

        return self._nreactant_atoms

    @property
    def nproduct_atoms(self) -> int:
        """The number of atoms in the product"""
        if self._nproduct_atoms is None:
            self._nproduct_atoms = len(self.pnums_atoms)

        return self._nproduct_atoms

    @property
    def rtokens(self) -> List[str]:
        """Just the reactant tokens"""
        return self.tokens[self._reactant_inds]

    @property
    def rtokens_filt(self) -> List[str]:
        """Reactant tokens without special tokens"""
        return self.rtokens[1:]

    @property
    def ptokens(self) -> List[str]:
        """Just the product tokens"""
        return self.tokens[self._product_inds]

    @property
    def ptokens_filt(self) -> List[str]:
        """Product tokens without special tokens"""
        return self.ptokens[:-1]

    def token_ind(self, atom_num) -> int:
        """Get token index from an atom number

        Note that this is not a lossless mapping. -1 represents any special token, but it is always mapped to the token at index [N - 1]
        """
        return self.atom2token[atom_num]

    def atom_num(self, token_ind) -> int:
        """Get the atom number corresponding to a token"""
        return self.token2atom[token_ind]

    def is_atom(self, token_ind) -> int:
        """Check whether a token is an atom"""
        return self.atom_token_mask[token_ind]

    def get_neighboring_attentions(self, atom_num) -> np.ndarray:
        """Get a vector of shape (n_atoms,) representing the neighboring attentions to an atom number.

        Non-zero attentions are the attentions for neighboring atoms
        """
        return self.atom_attentions[atom_num] * self.adjacency_matrix[atom_num]

    def get_neighboring_atoms(self, atom_num):
        """Get the atom indexes neighboring the desired atom"""
        return np.nonzero(self.adjacency_matrix[atom_num])[0]

    def get_precursors_atom_types(self) -> List[int]:
        """Convert reactants into their atomic numbers"""
        if self._precursors_atom_types is None:
            self._precursors_atom_types = get_atom_types_smiles(
                "".join(self.rtokens[1:])
            )
        return self._precursors_atom_types

    def get_product_atom_types(self) -> List[int]:
        """Convert products into their atomic indexes"""
        if self._product_atom_types is None:
            self._product_atom_types = get_atom_types_smiles("".join(self.ptokens[:-1]))
        return self._product_atom_types

    def get_atom_type_mask(self) -> np.ndarray:
        """Return a mask where only atoms of the same type are True"""
        if self._atom_type_mask is None:
            atom_type_mask = np.zeros(self.combined_attentions_filt_atoms.shape)
            precursor_atom_types = self.get_precursors_atom_types()
            for i, atom_type in enumerate(self.get_product_atom_types()):
                atom_type_mask[i, :] = (
                    np.array(precursor_atom_types) == atom_type
                ).astype(int)
            self._atom_type_mask = atom_type_mask
        return self._atom_type_mask

    def _get_combined_normalized_attentions(self):
        """Get normalized attention matrix from product atoms to candidate reactant atoms."""

        combined_attentions = np.multiply(
            self.atom_type_masked_attentions, self.attention_multiplier_matrix
        )

        row_sums = combined_attentions.sum(axis=1)
        normalized_attentions = np.divide(
            combined_attentions,
            row_sums[:, np.newaxis],
            out=np.zeros_like(combined_attentions),
            where=row_sums[:, np.newaxis] != 0,
        )
        return normalized_attentions

    def generate_attention_guided_pxr_atom_mapping(
        self, absolute_product_inds: bool = False
    ):
        """
        Generate attention guided product to reactant atom mapping.
        Args:
            absolute_product_inds: If True, adjust all indexes related to the product to be relative to that atom's position
                in the entire reaction SMILES
        """

        pxr_mapping_vector = (np.ones(len(self.pnums_atoms)) * -1).astype(int)

        output = {}

        confidences = np.ones(len(self.pnums_atoms))

        mapping_tuples = []

        for i in range(len(self.pnums_atoms)):
            attention_matrix = self._get_combined_normalized_attentions()

            if i == 0 and self.output_attentions:
                output["pxrrxp_attns"] = attention_matrix

            product_atom_to_map = np.argmax(np.max(attention_matrix, axis=1))
            corresponding_reactant_atom = np.argmax(attention_matrix, axis=1)[
                product_atom_to_map
            ]
            confidence = np.max(attention_matrix)

            if np.isclose(confidence, 0.0):
                confidence = 1.0
                corresponding_reactant_atom = pxr_mapping_vector[
                    product_atom_to_map
                ]  # either -1 or already mapped
                break

            pxr_mapping_vector[product_atom_to_map] = corresponding_reactant_atom
            confidences[product_atom_to_map] = confidence
            self._update_attention_multiplier_matrix(
                product_atom_to_map, corresponding_reactant_atom
            )

            if absolute_product_inds:
                adjusted_product_atom = product_atom_to_map + self.nreactant_atoms
            else:
                adjusted_product_atom = product_atom_to_map
            mapping_tuples.append(
                (adjusted_product_atom, corresponding_reactant_atom, confidence)
            )

        output["pxr_mapping_vector"] = pxr_mapping_vector.tolist()
        output["confidences"] = confidences
        output["mapping_tuples"] = mapping_tuples
        return output

    def _update_attention_multiplier_matrix(
        self, product_atom: np.signedinteger, reactant_atom: int
    ):
        """Perform the "neighbor multiplier" step of the atom mapping

        Increase the attention connection between the neighbors of specified product atom
        to the neighbors of the specified reactant atom. A stateful operation.

        Args:
            product_atom: Atom index of the product atom (relative to the beginning of the products)
            reactant_atom: Atom index of the reactant atom (relative to the beginning of the reactants)
        """
        if not reactant_atom == -1:
            neighbors_in_products = self.adjacency_matrix_products[product_atom]
            neighbors_in_reactants = self.adjacency_matrix_reactants[reactant_atom]

            self.attention_multiplier_matrix[
                np.ix_(neighbors_in_products, neighbors_in_reactants)
            ] *= float(self.attention_multiplier)

        if self.mask_mapped_product_atoms:
            self.attention_multiplier_matrix[product_atom] = np.zeros(
                len(self.rnums_atoms)
            )
        if self.mask_mapped_reactant_atoms:
            self.attention_multiplier_matrix[:, reactant_atom] = np.zeros(
                len(self.pnums_atoms)
            )

    def __len__(self) -> int:
        """Length of provided tokens"""
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"AttMapper(`{self.rxn[:50]}...`)"
