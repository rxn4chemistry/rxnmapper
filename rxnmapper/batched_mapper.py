import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pkg_resources
from rxn.utilities.containers import chunker
from rxn.utilities.files import PathLike

from .core import RXNMapper

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Alias for what the original mapper returns
ResultWithInfo = Dict[str, Any]


class BatchedMapper:
    """
    Class to atom-map reactions in batches, with error control.
    """

    def __init__(
        self,
        batch_size: int,
        model_path: Optional[PathLike] = None,
        head: int = 5,
        attention_multiplier: float = 90.0,
        layer: int = 10,
        model_type: str = "albert",
        canonicalize: bool = False,
        placeholder_for_invalid: str = ">>",
    ):
        """
        Args:
            batch_size: batch size for inference.
            model_path: path to the model directory, defaults to the model from
                the original publication.
            head: head related to atom mapping in the model. The default is the
                one for the original publication.
            attention_multiplier: attention multiplier, no need to change the default.
            layer: layer, no need to change the default.
            model_type: model type.
            canonicalize: whether to canonicalize before predicting the atom mappings.
            placeholder_for_invalid: placeholder to use in the output when there
                is an issue in the prediction (number of tokens, invalid SMILES, ...).
        """
        if model_path is None:
            model_path = pkg_resources.resource_filename(
                "rxnmapper", "models/transformers/albert_heads_8_uspto_all_1310k"
            )
        self.mapper = RXNMapper(
            config=dict(
                model_path=str(model_path),
                head=head,
                layers=[layer],
                model_type=model_type,
                attention_multiplier=attention_multiplier,
            ),
        )
        self.batch_size = batch_size
        self.canonicalize = canonicalize
        self.placeholder_for_invalid = placeholder_for_invalid

    def map_reactions(self, reaction_smiles: Iterable[str]) -> Iterator[str]:
        """Map the given reactions, returning the mapped SMILES strings.

        Args:
            reaction_smiles: reaction SMILES strings to map.

        Returns:
            iterator over mapped strings; a placeholder is returned for the
            entries that failed.
        """
        for result in self.map_reactions_with_info(reaction_smiles):
            if result == {}:
                yield self.placeholder_for_invalid
            else:
                yield result["mapped_rxn"]

    def map_reactions_with_info(
        self, reaction_smiles: Iterable[str], detailed: bool = False
    ) -> Iterator[ResultWithInfo]:
        """Map the given reactions, returning the results as dictionaries.

        Args:
            reaction_smiles: reaction SMILES strings to map.
            detailed: detailed output or not.

        Returns:
            iterator over dictionaries (in the format returned by the RXNMapper class);
            an empty dictionary is returned for the entries that failed.
        """
        for rxns_chunk in chunker(reaction_smiles, chunk_size=self.batch_size):
            yield from self._map_reaction_batch(rxns_chunk, detailed=detailed)

    def _map_reaction_batch(
        self, reaction_batch: List[str], detailed: bool
    ) -> Iterator[ResultWithInfo]:
        try:
            yield from self._try_map_reaction_batch(reaction_batch, detailed=detailed)
        except Exception as e:
            logger.warning(
                f"Error while mapping chunk of {len(reaction_batch)} reactions: {e}. "
                "Mapping them individually."
            )
            yield from self._map_reactions_one_by_one(reaction_batch, detailed=detailed)

    def _try_map_reaction_batch(
        self, reaction_batch: List[str], detailed: bool
    ) -> List[ResultWithInfo]:
        """
        Map a reaction batch, without error handling.

        Note: we return a list, not a generator function, to avoid returning partial
        results.
        """
        return self.mapper.get_attention_guided_atom_maps(
            reaction_batch,
            canonicalize_rxns=self.canonicalize,
            detailed_output=detailed,
        )

    def _map_reactions_one_by_one(
        self, reaction_batch: Iterable[str], detailed: bool
    ) -> Iterator[ResultWithInfo]:
        """
        Map a reaction batch, one reaction at a time.

        Reactions causing an error will be replaced by a placeholder.
        """
        for reaction in reaction_batch:
            try:
                yield self._try_map_reaction_batch([reaction], detailed=detailed)[0]
            except Exception as e:
                logger.info(
                    f"Reaction causing the error: {reaction}; {e.__class__.__name__}: {e}"
                )
                yield {}
