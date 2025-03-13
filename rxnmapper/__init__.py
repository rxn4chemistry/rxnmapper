"""rxnmapper initialization."""

__name__ = "rxnmapper"
__version__ = "0.4.1"  # managed by bump2version


from .batched_mapper import BatchedMapper
from .core import RXNMapper

__all__ = [
    "BatchedMapper",
    "RXNMapper",
]
