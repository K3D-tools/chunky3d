__all__ = [
    'Sparse',
    'point_probe',
    'Chunky',
]

from .chunky import (
    Sparse,
    point_probe,
)

from ._version import (
    __version__,
    version_info,
)

# compatibility aliases:
Chunky = Sparse
