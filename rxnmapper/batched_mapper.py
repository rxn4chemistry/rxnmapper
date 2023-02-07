import logging
from typing import Iterable, Iterator, List

from rxn.utilities.containers import chunker
from .core import RXNMapper

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BatchedMapper:
    """
    Class to atom-map reactions in batches, with error control.
    """

    def __init__(self, batch_size: int, canonicalize: bool = False, placeholder_for_invalid: str = ">>"):
        self.mapper = RXNMapper()
        self.batch_size = batch_size
        self.canonicalize = canonicalize
        self.placeholder_for_invalid = placeholder_for_invalid

    def map_reactions(self, reaction_smiles: Iterable[str]) -> Iterator[str]:
        for rxns_chunk in chunker(reaction_smiles, chunk_size=self.batch_size):
            yield from self._map_reaction_batch(rxns_chunk)

    def _map_reaction_batch(self, reaction_batch: List[str]) -> Iterator[str]:
        try:
            yield from self._try_map_reaction_batch(reaction_batch)
        except Exception:
            logger.warning(
                f"Error while mapping chunk of {len(reaction_batch)} reactions. "
                "Mapping them individually."
            )
            yield from self._map_reactions_one_by_one(reaction_batch)

    def _try_map_reaction_batch(self, reaction_batch: List[str]) -> List[str]:
        """
        Map a reaction batch, without error handling.

        Note: we return a list, not a generator function, to avoid returning partial
        results.
        """
        results = self.mapper.get_attention_guided_atom_maps(
            reaction_batch, canonicalize_rxns=self.canonicalize, detailed_output=False
        )
        return [result["mapped_rxn"] for result in results]

    def _map_reactions_one_by_one(self, reaction_batch: Iterable[str]) -> Iterator[str]:
        """
        Map a reaction batch, one reaction at a time.

        Reactions causing an error will be replaced by a placeholder.
        """
        for reaction in reaction_batch:
            try:
                yield self._try_map_reaction_batch([reaction])[0]
            except Exception as e:
                logger.info(
                    f"Reaction causing the error: {reaction}; {e.__class__.__name__}: {e}"
                )
                yield self.placeholder_for_invalid
