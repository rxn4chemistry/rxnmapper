import logging
from typing import Any, Dict, Iterable, Iterator, List

from rxn.utilities.containers import chunker

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
        canonicalize: bool = False,
        placeholder_for_invalid: str = ">>",
    ):
        self.mapper = RXNMapper()
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
        except Exception:
            logger.warning(
                f"Error while mapping chunk of {len(reaction_batch)} reactions. "
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
