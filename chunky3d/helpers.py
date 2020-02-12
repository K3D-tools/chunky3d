import numpy as np


def slice_normalize(key, shape):
    """
    Function change None values and negative indices for values in range <0 ; shape>

    :param key: tuple of slices
    :param shape: shape of array
    :return: tuple of 'normalized' slices
    :raises: TypeError if slice start, stop or step is not an integer
    """

    assert len(key) <= len(shape), "too many indices for array"

    k = list()
    ctr = 0
    for slc in key:
        if not isinstance(slc, slice):  # when key is integer
            slc = slice(slc, slc + 1, 1)
        if not (np.issubdtype(type(slc.start), np.integer) or slc.start is None) or \
           not (np.issubdtype(type(slc.stop), np.integer) or slc.stop is None) or \
           not (np.issubdtype(type(slc.step), np.integer) or slc.step is None):
            raise TypeError('slice indices must be integers or None')
        start = slc.start
        stop = slc.stop
        step = slc.step
        if start is None:
            start = 0
        else:
            if start < 0:
                start = shape[ctr] + start
        if stop is None:
            stop = shape[ctr]
        else:
            if stop < 0:
                stop = shape[ctr] + stop
        if step is None:
            step = 1
        ctr += 1
        k.append(slice(start, stop, step))
    return tuple(k)


def pad_to_chunk(a, n):
    """
    Pad dense array to multiplicity of a number (e.g. chunk size)
    """
    to_n = lambda x, n: (0, x + (n - x % n) % n - x)
    to_ = lambda x: to_n(x, n)
    a_padded = np.pad(a, list(map(to_, a.shape)), mode='constant')
    return a_padded
