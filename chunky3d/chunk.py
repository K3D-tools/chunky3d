import numpy as np


class Chunk(np.ndarray):
    def __new__(cls, input_array, origin=None, spacing=None):
        obj = np.asarray(input_array).view(cls)
        obj.origin = origin
        obj.spacing = spacing
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.origin = getattr(obj, 'origin', None)
        self.spacing = getattr(obj, 'spacing', None)

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, val):
        self._spacing = val

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, val):
        self._origin = val

    def to_dict(self):
        return {
            b'origin': self.origin,
            b'spacing': self.spacing,
            b'ndarray': self
        }

    @staticmethod
    def from_dict(d):
        return Chunk(d[b'ndarray'].copy(), origin=d[b'origin'], spacing=d[b'spacing'])
