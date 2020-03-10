__all__ = [
    'Sparse',
    'point_probe',
    'Chunky',
]

from .chunky import (
    Sparse,
    point_probe
)

# compatibility aliases:
Chunky = Sparse
