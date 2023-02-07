"""rxnmapper initialization."""
__name__ = "rxnmapper"
__version__ = "0.2.4"  # managed by bump2version


from .core import RXNMapper
from .batched_mapper import BatchedMapper

__all__ = [
    "BatchedMapper",
    "RXNMapper",
]
