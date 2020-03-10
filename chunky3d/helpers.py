import numpy as np


def adjust_key(key, shape):
    """Prepare a slicing tuple (key) for slicing an array of specific shape."""

    if type(key) is not tuple:
        key = (key,)

    if Ellipsis in key:
        if key.count(Ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")

        ellipsis_pos = key.index(Ellipsis)

        missing = (slice(None),) * (len(shape) - len(key) + 1)
        key = key[:ellipsis_pos] + missing + key[ellipsis_pos + 1:]

    if  len(key) > len(shape):
        raise IndexError('too many indices for array')

    if len(key) < len(shape):
        missing = (slice(None),) * (len(shape) - len(key))
        key += missing

    return key


def _check_slice(slc):
    if any(
        x is not None and not np.issubdtype(type(x), np.integer)
        for x in (slc.start, slc.stop, slc.step)
    ):
        raise IndexError('slice indices must be integers or None')


def slice_normalize(key, shape):
    """change None values and negative indices to values in range <0; shape>

    :param key: tuple of slices, argument for __setitem__ or __getitem__
    :param shape: shape of array
    :return: tuple of 'normalized' slices
    :raises: IndexError if slice start, stop or step is not an integer"""

    key = adjust_key(key, shape)
    k = list()

    for ctr, slc in enumerate(key):
        if not isinstance(slc, slice):  # when key is integer
            if not np.issubdtype(type(slc), np.integer):
                raise IndexError('single indices must be integers')

            if slc < -shape[ctr] or slc >= shape[ctr]:
                raise IndexError(f'index {slc} is out of bounds for axis {ctr} with size {shape[ctr]}')

            if slc < 0:
                # nasty case for [-1]
                slc + shape[ctr]

            slc = slice(slc, slc + 1, 1)

        _check_slice(slc)

        start, stop, step = slc.start, slc.stop, slc.step

        if step is None:
            step = 1
        if step == 0:
            raise ValueError(f'slice step of axis {ctr} cannot be zero')

        if start is None:
            start = 0 if step > 0 else shape[ctr] - 1
        else:
            # bring start to (-shape[ctr], shape[ctr])
            start = min(max(start, -shape[ctr]), shape[ctr])
            if start < 0:
                start = shape[ctr] + start

        if stop is None:
            stop = shape[ctr] if step > 0 else -1
        else:
            # bring stop to (-shape[ctr], shape[ctr])
            stop = min(max(stop, -shape[ctr]), shape[ctr])
            if stop < 0:
                stop = shape[ctr] + stop


        k.append(slice(start, stop, step))
    return tuple(k)


def slice_shape(key, array_shape):
    """Determine effective shape of slicing an array

    :param key: tuple of slices, argument for __setitem__ or __getitem__
    :param shape: shape of array
    :return: resulting shape"""

    key = slice_normalize(key, array_shape)
    shape = list()

    for slc in key:
        if (slc.stop - slc.start) * slc.step <= 0:
            # range direction must match step sign
            # and slc.start != slc.stop to have a nonempty
            shape.append(0)
            continue

        span = abs(slc.stop - slc.start)  # regardless of which is greater
        step = abs(slc.step)  # dirction already compatible
        shape.append((span + step - 1) // step)  # __ceildiv__

    return tuple(shape)


def pad_to_chunk(a, n):
    """
    Pad dense array to multiplicity of a number (e.g. chunk size)
    """
    to_n = lambda x, n: (0, x + (n - x % n) % n - x)
    to_ = lambda x: to_n(x, n)
    a_padded = np.pad(a, list(map(to_, a.shape)), mode='constant')
    return a_padded


def check_start_end(start, end, shape, check_end=True):
    if (start < np.zeros(3)).any():
        raise IndexError(f"start index {start} out of range {shape}.")
    # if |step| is > 1, testing the end is not as simple
    if check_end and (end > np.array(shape)).any():
        raise IndexError(f"end index {end} out of range {shape}.")
