import platform
import zlib

from numba import jit, njit, prange
import dill
import msgpack
import msgpack_numpy as m
import numpy as np
import psutil
import scipy.ndimage

from .chunk import Chunk
from .helpers import slice_normalize, slice_shape, check_start_end
from .multiprocesses import ProcessPool


@njit
def fast_get(ijk, dense_data, grid_mask, chunks, envelope=0, fill_value=0.0, error_value=0.0):
    if ijk[0] < 0 or ijk[1] < 0 or ijk[2] < 0:
        return error_value

    bi, li = divmod(ijk[0], chunks[0])
    bj, lj = divmod(ijk[1], chunks[1])
    bk, lk = divmod(ijk[2], chunks[2])

    shp = grid_mask.shape

    if bi > shp[0] or bj > shp[1] or bk > shp[2]:
        return error_value

    idx = grid_mask[bi, bj, bk]

    if idx != 0xffffffff:
        return dense_data[envelope + idx * (chunks[0] + 2 * envelope) + li,
                          envelope + lj,
                          envelope + lk]
    else:
        return fill_value


@njit
def fast_get_interpolated(ijk, dense_data, grid_mask, shape, envelope=0):
    """
           *-----------*
          /|5         /|6           k (x)
         / |         / |            ^
        /  |        /  |            |
       *-----------*   |            |
       |8  *-------|7--*             ----> i (z)
       |  / 1      |  / 2          /
       | /         | /            /
       |/          |/            v
       *-----------*            j (y)
      4           3
    """
    ijk_base = np.array(ijk).astype(np.uint32)
    t = np.array(ijk) - ijk_base

    v1 = fast_get(ijk_base, dense_data, grid_mask, shape, envelope)
    v2 = fast_get(ijk_base + np.array([1, 0, 0]), dense_data, grid_mask, shape, envelope)
    v4 = fast_get(ijk_base + np.array([0, 1, 0]), dense_data, grid_mask, shape, envelope)
    v3 = fast_get(ijk_base + np.array([1, 1, 0]), dense_data, grid_mask, shape, envelope)
    v5 = fast_get(ijk_base + np.array([0, 0, 1]), dense_data, grid_mask, shape, envelope)
    v6 = fast_get(ijk_base + np.array([1, 0, 1]), dense_data, grid_mask, shape, envelope)
    v7 = fast_get(ijk_base + np.array([1, 1, 1]), dense_data, grid_mask, shape, envelope)
    v8 = fast_get(ijk_base + np.array([0, 1, 1]), dense_data, grid_mask, shape, envelope)

    v12 = v1 * (1 - t[0]) + v2 * t[0]
    v43 = v4 * (1 - t[0]) + v3 * t[0]
    v1243 = v12 * (1 - t[1]) + v43 * t[1]

    v56 = v5 * (1 - t[0]) + v6 * t[0]
    v87 = v8 * (1 - t[0]) + v7 * t[0]
    v5687 = v56 * (1 - t[1]) + v87 * t[1]

    return v1243 * (1 - t[2]) + v5687 * t[2]


@jit(nogil=True, parallel=True)
def fast_point_probe(xyz, dense_data, grid_mask, shape, origin, spacing, envelope=0):
    """
    Interpolates Sparse on x, y, z inside bounding box.
    Expects np array xyz, of shape [Npoints,3]
    shape, origin, spacing must be np.arrays

    The proper call for p being a Sparse array is:

        fast_point_probe(xyz, p.dense_data, p.grid_mask, np.array(p.chunks),
                         np.array(p.origin), np.array(p.spacing), envelope)

           *-----------*
          /|5         /|6           x (k)
         / |         / |            ^
        /  |        /  |            |
       *-----------*   |            |
       |8  *-------|7--*             ----> z (i)
       |  / 1      |  / 2          /
       | /         | /            /
       |/          |/            v
       *-----------*            y (j)
      4           3
    """

    ijk = ((xyz - origin) / spacing)[:, ::-1]
    ijk_base_all = np.floor(ijk).astype(np.int32)
    t_all = ijk - ijk_base_all
    output = np.empty(xyz.shape[0])

    for i in prange(xyz.shape[0]):
        t = t_all[i]
        b = ijk_base_all[i]

        v1 = fast_get(b, dense_data, grid_mask, shape, envelope)
        v2 = fast_get((b[0] + 1, b[1], b[2]), dense_data, grid_mask, shape, envelope)
        v3 = fast_get((b[0] + 1, b[1] + 1, b[2]), dense_data, grid_mask, shape, envelope)
        v4 = fast_get((b[0], b[1] + 1, b[2]), dense_data, grid_mask, shape, envelope)
        v5 = fast_get((b[0], b[1], b[2] + 1), dense_data, grid_mask, shape, envelope)
        v6 = fast_get((b[0] + 1, b[1], b[2] + 1), dense_data, grid_mask, shape, envelope)
        v7 = fast_get((b[0] + 1, b[1] + 1, b[2] + 1), dense_data, grid_mask, shape, envelope)
        v8 = fast_get((b[0], b[1] + 1, b[2] + 1), dense_data, grid_mask, shape, envelope)

        v12 = v1 * (1 - t[0]) + v2 * t[0]
        v43 = v4 * (1 - t[0]) + v3 * t[0]
        v1243 = v12 * (1 - t[1]) + v43 * t[1]

        v56 = v5 * (1 - t[0]) + v6 * t[0]
        v87 = v8 * (1 - t[0]) + v7 * t[0]
        v5687 = v56 * (1 - t[1]) + v87 * t[1]

        output[i] = v1243 * (1 - t[2]) + v5687 * t[2]

    return output


def point_probe(xyz, p, envelope=0):
    """
    Wraps numba compiled `fast_point_probe`.
    """
    if p.grid_mask is None:
        raise RuntimeError('Missing grid_mask in sparse array. Use update_grid_mask() before point_probe().')

    xyz = np.asarray(xyz)

    if xyz.shape == (3,):
        xyz = np.expand_dims(xyz, 0)

    assert len(xyz.shape) == 2 and xyz.shape[1] == 3

    return fast_point_probe(
        xyz, p.dense_data, p.grid_mask, np.array(p.chunks), np.array(p.origin), np.array(p.spacing), envelope
    )


class Sparse:
    """
    This is a class for storing data in chunks.
    All chunks have the same shape (each dimension of chunk must be a power of two).
    Chunks filled with unique value are stored as one value.
    Chunks filled with default value are not stored.


    Attributes:
        dtype: Data type stored in array
        shape (tuple): dimensions of data structure as dense matrix (i, j, k)
        chunks (tuple): dimensions of chunk (i, j, k)
        origin (tuple): position in world coordinates (x, y, z) of the voxel (0, 0, 0)
        spacing (tuple): distance between voxels in each direction (x, y, z)
        cdata_shape (tuple): dimensions of chunk grid (i, j, k)
        fill_value (self.dtype): value used for uninitialized (empty) portions of the array.
        nchunks (int): total number of chunks
        nchunks_initialized (int): numbers of chunks that have been initialized with some data

    Methods
        get_chunk: get chunk from grid
        set_chunk: set chunk in grid
        get: get dense submatrix from structure
        set: insert dense submatrix into structure
        get_k3d_voxels_group_dict: get chunks as list of dict for use in K3D voxels group
    """

    def __init__(self, shape, chunks=True, fill_value=0, dtype=np.double, origin=(0, 0, 0), spacing=(1.0, 1.0, 1.0)):
        self._shape = None
        self._dtype = None
        self._chunk_shape = None
        self._block_shape = None
        self._grid_mask = None
        self._origin = None
        self._spacing = None

        self._default_value = fill_value
        self.dtype = dtype
        self.chunks = chunks
        self.shape = shape
        self.origin = origin
        self.spacing = spacing
        self._block_shape = tuple(np.ceil(np.divide(self.shape, self.chunks)).astype(np.int_))
        self._grid = dict()

        self._memory_blocks = []
        self._memory_blocks_with_holes = 0

    def _validate_dense_data(self):
        if len(self._memory_blocks) != 1:
            raise Exception('Memory blocks have wrong len: {}, use make_dense_data() to fix.'.format(
                len(self._memory_blocks)))

        if self._memory_blocks_with_holes != 0:
            raise Exception('Memory blocks having holes: {}, use make_dense_data() to fix.'.format(
                self._memory_blocks_with_holes))

    @property
    def dense_data(self):
        self._validate_dense_data()
        return self._memory_blocks[0]

    @dense_data.setter
    def dense_data(self, val):
        self._validate_dense_data()
        self._memory_blocks[0][:] = val

    def __sizeof__(self):
        total = super().__sizeof__()

        # dicts
        total += self._grid.__sizeof__()
        for key, value in self._grid.items():
            total += key.__sizeof__()
            total += value.__sizeof__()

        # lists
        total += self._memory_blocks.__sizeof__()
        for block in self._memory_blocks:
            total += block.__sizeof__()

        # arrays
        total += self._grid_mask.__sizeof__()

        return total

    def make_dense_data(self, envelope=0):
        """Defragment self._memory_blocks into a single array with possible envelope around chunks."""

        if len(self._memory_blocks) == 1 and self._memory_blocks_with_holes == 0 \
                and self._memory_blocks[0].shape[2] == self._chunk_shape[2] + envelope * 2:
            # already defragmented
            return

        dense_data = np.zeros(((self._chunk_shape[0] + envelope * 2) * self.nchunks_initialized,
                               (self._chunk_shape[1] + envelope * 2),
                               (self._chunk_shape[2] + envelope * 2)), dtype=self.dtype)

        w = self._chunk_shape[0] + envelope * 2

        for i, k in enumerate(sorted(self._grid.keys())):
            o1, o2 = i * w + envelope, (i + 1) * w - envelope

            if envelope > 0:
                dense_data[o1:o2, envelope:-envelope, envelope:-envelope] = self._grid[k]
                self._grid[k] = self._make_chunk(dense_data[o1:o2, envelope:-envelope, envelope:-envelope], k)
            else:
                dense_data[o1:o2, :, :] = self._grid[k]
                self._grid[k] = self._make_chunk(dense_data[o1:o2, :, :], k)

        del self._memory_blocks[:]
        self._memory_blocks_with_holes = 0
        self._memory_blocks = [dense_data] if dense_data.shape[0] > 0 else []

    def _global_to_chunk_coords(self, coord):
        """
        Global coordinates to chunk id and in chunk coords
        :param coord: (tuple) global coordinates
        :return: chunk_id, coord_in_chunk
        """
        return np.divmod(coord, self.chunks)

    def _chunk_to_global_coord(self, chunk_id, coord):
        """
        Local chunk coords to global coordinates
        :param chunk_id: (tuple) id of chunk in grid
        :param coord: (tuple) local coordinates in chunk
        :return: (tuple) global coordinates
        """
        ret = np.multiply(chunk_id, self.chunks)
        ret += np.array(coord)
        return ret

    @property
    def dtype(self):
        """ The NumPy data type. """
        return self._dtype

    @dtype.setter
    def dtype(self, val):
        self._dtype = val

    @property
    def shape(self):
        """ A tuple of integers describing the length of each dimension of the array. """
        return self._shape

    @shape.setter
    def shape(self, val):
        if isinstance(val, int):
            self._shape = tuple([val] * 3)
        elif isinstance(val, np.ndarray) and len(val) == 3:
            self._shape = tuple(val)
        elif isinstance(val, tuple) and len(val) == 3:
            self._shape = val
        else:
            raise Exception("Structure shape error!")

    @property
    def size(self):
        return np.product(self._shape)

    @property
    def itemsize(self):
        return self._dtype.itemsize

    @property
    def chunks(self):
        """ A tuple of integers describing the length of each dimension of a chunk of the array. """
        return self._chunk_shape

    @chunks.setter
    def chunks(self, val):
        if isinstance(val, tuple) or isinstance(val, np.ndarray):
            if len(val) == 3:
                for v in val:
                    if v <= 0 and ((v & (v - 1)) != 0):
                        raise Exception("Chunk size must be a power of two")
                self._chunk_shape = tuple(val)
            else:
                raise Exception("Chunk size length error")
        elif isinstance(val, bool):
            self._chunk_shape = (32, 32, 32)
        elif isinstance(val, int):
            if val <= 0 and ((val & (val - 1)) != 0):
                raise Exception("Chunk size must be a power of two")
            self._chunk_shape = tuple([val] * 3)
        else:
            raise Exception("Chunk size dtype error")

    @property
    def cdata_shape(self):
        """ A tuple of integers describing the number of chunks along each dimension of the array. """
        return self._block_shape

    @property
    def fill_value(self):
        """ A value used for uninitialized (empty) portions of the array. """
        return self._default_value

    @property
    def nchunks(self):
        """ Total number of chunks. """
        return np.prod(self.cdata_shape)

    @property
    def nchunks_initialized(self):
        """ The number of chunks that have been initialized with some data. """
        return len(self._grid)

    @property
    def kchunks_initialized(self):
        """ List of keys of chunks that have been initialized with some data. """
        return set(self._grid.keys())

    @property
    def chunks_initialized(self):
        """ List of copies of chunks that have been initialized with some data. """
        return [Chunk(self._grid[k],
                      spacing=self.spacing,
                      origin=tuple(self.origin + np.multiply(k, self.spacing))) for k in self.kchunks_initialized]

    @property
    def origin(self):
        """ A tuple of floats describing the position in world coordinates (x,y,z) of the voxel (0,0,0). """
        return self._origin

    @origin.setter
    def origin(self, val):
        if isinstance(val, int) or isinstance(val, float):
            self._origin = tuple([val] * 3)
        elif isinstance(val, np.ndarray) and len(val) == 3:
            self._origin = tuple(val)
        elif isinstance(val, tuple) and len(val) == 3:
            self._origin = val
        else:
            raise Exception("Origin error!")

    @property
    def spacing(self):
        """ A tuple of floats describing the (width,height,length) of the cubical cells that compose the data set. """
        return self._spacing

    @spacing.setter
    def spacing(self, val):
        if isinstance(val, int) or isinstance(val, float):
            self._spacing = tuple([val] * 3)
        elif isinstance(val, np.ndarray) and len(val) == 3:
            self._spacing = tuple(val)
        elif isinstance(val, tuple) and len(val) == 3:
            self._spacing = val
        else:
            raise Exception("Spacing error!")

    @property
    def grid_mask(self):
        """ Grid mask is a dense array indicating indices of chunks.
        It has to be manually updated using `update_grid_mask` """
        return self._grid_mask

    def update_grid_mask(self):
        """
        An update of grid_mask attribute from dictionary of keys.
        """
        if type(self._grid_mask) is not np.ndarray or self._grid_mask.shape != self._block_shape:
            self._grid_mask = np.full(self._block_shape, 0xffffffff, dtype=np.uint32)
        else:
            self._grid_mask.fill(0xffffffff)

        self._grid_mask[tuple(np.array(sorted(self._grid.keys())).T)] = np.arange(self.nchunks_initialized)

    def get_chunk(self, idx, dtype=None):
        """
        Get chunk from the grid.
        :param:
        idx (tuple): coordinates in grid (i, j, k)
        dtype: type of the returned Chunk. Default: original dtype
        :return:
        chunk (ndarray / unique value)
        """
        if dtype is None:
            dtype = self.dtype

        # TODO: set values of chunks outside array dimensions as NaN
        if (idx < np.array([0, 0, 0])).any() or (idx >= np.array(self._block_shape)).any():
            raise Exception("Index out of range.", idx)
        if idx in self._grid.keys():
            if self._grid[idx].shape == (1,):
                arr = np.zeros(self._chunk_shape, dtype=dtype)
                arr.fill(self._grid[idx][0])
            else:
                arr = self._grid[idx].copy()
        else:
            arr = np.zeros(self._chunk_shape, dtype=dtype)
            arr.fill(self.fill_value)

        return self._make_chunk(arr, idx)

    def _make_chunk(self, arr, idx):
        chunk = arr.view(Chunk)
        chunk.origin = self.origin + (np.array(idx) * np.array(self.chunks))[::-1] * self.spacing
        chunk.spacing = self.spacing
        chunk.idx = idx

        return chunk

    def _remove_memory_block(self, val):
        for i, m in enumerate(self._memory_blocks):
            if id(m) == id(val):
                del self._memory_blocks[i]

    def _add_memory_block(self, val):
        for i, m in enumerate(self._memory_blocks):
            if id(m) == id(val):
                return

        self._memory_blocks.append(val)

    def _update_memory_block(self, idx, d):
        if idx not in self._grid.keys():
            self._add_memory_block(d)
        else:
            if self._grid[idx].shape != d.shape:
                self._remove_memory_block(self._grid[idx])
                self._add_memory_block(d)

        if idx not in self._grid.keys() or self._grid[idx].shape != d.shape:
            self._grid[idx] = d
        else:
            self._grid[idx][:] = d

    def set_chunk(self, idx, val):
        """
        Set chunk in the grid.

        :param:
        idx (tuple): coordinates in grid (i, j, k)
        val (ndarray or int/float): ndarray with chunk shape (i, j, k) or unique value
        """

        # TODO: add warrning if val.dtype != self.dtype
        if (idx < np.array([0, 0, 0])).any() or (idx >= np.array(self._block_shape)).any():
            raise IndexError(f"Index {idx} out of range {self._block_shape}.")

        if isinstance(val, np.ndarray):
            if val.shape != self._chunk_shape:
                raise ValueError("Chunk size error!")

            uniform = np.any(val != val[0, 0, 0])

            if uniform:
                d = val.astype(self.dtype)
                self._update_memory_block(idx, d)
            else:
                if np.equal(val[0, 0, 0], self.fill_value):
                    if idx in self._grid.keys():
                        if self._grid[idx].base is not None:
                            self._memory_blocks_with_holes += 1
                        else:
                            self._remove_memory_block(self._grid[idx])

                        del self._grid[idx]
                else:
                    d = val[0:1, 0:1, 0:1].astype(self.dtype).reshape((1,)).copy()
                    self._update_memory_block(idx, d)
        else:
            if np.equal(val, self.fill_value):
                if idx in self._grid.keys():
                    if self._grid[idx].base is not None:
                        self._memory_blocks_with_holes += 1
                    else:
                        self._remove_memory_block(self._grid[idx])

                    del self._grid[idx]
            else:
                d = np.array([val], dtype=self.dtype)
                self._update_memory_block(idx, d)
                self._grid[idx] = d

    def run_multivariate(self, func, sparse_list, *args):
        keys = set()

        for s in sparse_list:
            keys = keys | set(s._grid.keys())

        for k in keys:
            ret = func(self.get_chunk(k), [s.get_chunk(k) for s in sparse_list], *args)
            self.set_chunk(k, ret)

    def _run(self, *args, **kwargs):
        func = kwargs.get('sparse_func')
        envelope = kwargs.get('envelope')
        keys = kwargs.get('keys')
        prev = kwargs.get('prev')

        to_update = {}
        for k in keys:
            if envelope == (0, 0, 0):
                ret, prev = func(self.get_chunk(tuple(k)), prev, *args)
                to_update[tuple(k)] = ret
            else:
                s = self._chunk_to_global_coord(k, (-envelope[0], - envelope[1], -envelope[2]))
                e = self._chunk_to_global_coord(k, (self._chunk_shape[0] + envelope[0],
                                                    self._chunk_shape[1] + envelope[1],
                                                    self._chunk_shape[2] + envelope[2]))
                s_to_pad = s.copy()
                s_to_pad[s_to_pad > 0] = 0
                s_to_pad = np.abs(s_to_pad)
                s[s < 0] = 0

                e_to_pad = e.copy()
                e_to_pad[e_to_pad <= np.array(self._shape)] = 0
                e_to_pad = e_to_pad - np.array(self._shape)
                e_to_pad[e_to_pad < 0] = 0

                shp = np.array(self._shape)
                e[e > shp] = shp[e > shp]

                data = self.get(s, e)
                data = np.pad(data, list(zip(s_to_pad, e_to_pad)), 'constant', constant_values=self.fill_value)

                chunk = data.view(Chunk)
                chunk.spacing = self.spacing
                chunk.origin = self.origin + s[::-1] * self.spacing

                ret, prev = func(chunk, prev, *args)

                to_update[tuple(k)] = ret[envelope[0]:-envelope[0],
                                      envelope[1]:-envelope[1],
                                      envelope[2]:-envelope[2]
                                      ]
        return to_update, prev

    def _get_keys_for_run(self, envelope, skip_neighbours):
        """ Helper method to determine keys which Sparse will be iterated over during `run` method """
        if envelope == (0, 0, 0):
            keys = set(self._grid.keys())
        else:
            self.update_grid_mask()
            keys = set(self._grid.keys())
            if not skip_neighbours:
                conv_size = np.array([2, 2, 2]) + np.ceil(np.array(envelope) / np.array(self.chunks)).astype(np.uint16)
                neighbours = scipy.ndimage.filters.convolve(self.grid_mask.astype(np.uint16) != 0xFFFF,
                                                            np.ones(conv_size),
                                                            mode='constant',
                                                            cval=0.0)
                neighbours = (neighbours * ~(self.grid_mask.astype(np.uint16) != 0xFFFF)) > 0
                for idx in np.dstack(np.where(neighbours == True))[0]:
                    keys.add(tuple(idx.tolist()))
        return sorted(keys)

    def _run_multiprocess(self, func, *args, envelope=(0, 0, 0), prev=None, skip_neighbours=False, multiprocesses=1):
        pp = ProcessPool(multiprocesses)
        to_update = {}
        keys = self._get_keys_for_run(envelope, skip_neighbours)
        for k in np.array_split(keys, multiprocesses):
            pp.add_job(dill.dumps({
                'call': self._run,
                'args': args,
                'kwargs': {
                    'sparse_func': func,
                    'envelope': envelope,
                    'sparse': self,
                    'keys': k,
                    'prev': prev,
                },
            }))
        pp.finish_pool_queue()
        prev_combined = []
        for chunks, prev in pp.shared.get():
            for key, chunk in chunks.items():
                to_update[key] = chunk
            prev_combined.append(prev)

        return to_update, prev_combined

    def _run_singleprocess(self, func, *args, envelope=(0, 0, 0), prev=None, skip_neighbours=False):
        keys = self._get_keys_for_run(envelope, skip_neighbours)
        to_update, prev = self._run(*args, sparse_func=func, envelope=envelope, prev=prev, keys=keys)
        return to_update, [prev]  # list for compatibility with multiprocess

    def run(self, func, *args, envelope=(0, 0, 0), prev=None, skip_neighbours=False, multiprocesses=1):
        """
        Run given function on every initialized Chunk (pycardio.sparse.chunk).
        :param func: Function to be called on Chunk,
            always in form `func(data_in: Chunk, prev_in, *args) -> (data_out: ndarray, prev_out)` where
            `data_in` is an input Chunk,
            `prev_in` acts as an accumulator and is passed from the previous `func` execution to the next one,
            `args` is a non-keyworded variable length argument list.
            Function has to return two elements:
            `data_out` is a (possibly modified) ndarray from `data_in` which is to be inserted in-place into Sparse,
            `prev_out` is a (possibly modified) `prev_in` accumulator.
        :param args: non-keyworded variable length argument list which will be passed to every `func` call.
        :param envelope: 3-element tuple for extra margin on each direction in each axis on the `data_in` Chunk.
        :param prev: initial value for an accumulator.
        :param skip_neighbours: excludes chunks' neighbors from iterated keys list if `envelope` > 0.
        :param multiprocesses: number of processes spawned to execute `func` in parallel (chunk keys distributed evenly).
            For values below 0: (physical_cpus + multiprocesses + 1) is used, e.g. for -2, all **physical** CPUs but one are used.
            Multiprocessing is disabled if set to other values (specifically: 0, 1, None, ...).
        :return: a list of accumulated results.
            List's length is based on number of processes involved: [`prev_out_proc1`, `prev_out_proc2`, `prev_out_proc3`, ...]
        """

        to_update = {}
        if multiprocesses not in (0, 1):
            if platform.system() != 'Linux':
                raise NotImplementedError("Multiprocessing currently works only on Unix.")
            if multiprocesses < 0:
                physical_cores = psutil.cpu_count(logical = False)
                multiprocesses = max(1, physical_cores + multiprocesses + 1)
            to_update, prev = self._run_multiprocess(func, *args,
                                                     envelope=envelope, prev=prev,
                                                     skip_neighbours=skip_neighbours,
                                                     multiprocesses=multiprocesses)
        else:
            to_update, prev = self._run_singleprocess(func, *args,
                                                      envelope=envelope, prev=prev,
                                                      skip_neighbours=skip_neighbours)
        for k in to_update.keys():
            self.set_chunk(k, to_update[k])
        return prev

    def raw_run(self, func, start, end, prev=None, step=(1, 1, 1), *args):
        check_start_end(start, end, self._shape)

        s = np.array(start)
        e = np.array(end)
        cs = np.array(self._chunk_shape)
        blck = (np.floor_divide(s, self._chunk_shape), np.floor_divide(e, self._chunk_shape) + 1)

        gs = s.copy()

        ls_c, ls, le_c, le = None, None, None, None
        for i in range(blck[0][0], blck[1][0]):
            for j in range(blck[0][1], blck[1][1]):
                for k in range(blck[0][2], blck[1][2]):
                    ls_c, ls = self._global_to_chunk_coords(gs)
                    le_c, le = self._global_to_chunk_coords(e)
                    le[ls_c != le_c] = cs[ls_c != le_c]
                    le_c[ls_c != le_c] = ls_c[ls_c != le_c]

                    if np.any(ls >= le):
                        break

                    prev = func(
                        (i, j, k),
                        gs,
                        slice(ls[0], le[0], step[0]),
                        slice(ls[1], le[1], step[1]),
                        slice(ls[2], le[2], step[2]),
                        prev,
                        *args
                    )

                    gs[2] = self._chunk_to_global_coord(
                        ls_c, np.floor_divide(le - ls - 1, step) * step + ls
                    )[2] + step[2]

                gs[2] = s[2]
                gs[1] = self._chunk_to_global_coord(ls_c, np.floor_divide(le - ls - 1, step) * step + ls)[1] + step[1]

            gs[1] = s[1]
            gs[0] = self._chunk_to_global_coord(ls_c, np.floor_divide(le - ls - 1, step) * step + ls)[0] + step[0]

        return prev

    def get(self, start, end, step=(1, 1, 1), squeeze=True):
        """
        Get dense submatrix from structure
        :param start: (tuple) global coordinates of submatrix start (i,j,k)
        end: (tuple) global coordinates of submatrix end (i,j,k)
        :return: (ndarray) dense array
        """
        s = np.array(start)
        e = np.array(end)
        d = np.ceil(np.divide(e - s, step)).astype(np.int_)
        ret = np.empty(d, dtype=self.dtype)

        def process(idx, gs, z, y, x, _):
            tmp = self.get_chunk(idx)[z, y, x]
            ret_s = np.floor_divide(gs - s, step)

            ret[ret_s[0]:ret_s[0] + tmp.shape[0],
            ret_s[1]:ret_s[1] + tmp.shape[1],
            ret_s[2]:ret_s[2] + tmp.shape[2]] = tmp

        self.raw_run(process, start, end, step=step)

        if squeeze:
            return np.squeeze(ret)
        else:
            return ret

    def set(self, start, val, step=(1, 1, 1)):
        """
        Insert matrix into structure

        :param:
        start (tuple): global coordinates of submatrix start (I,J,K)
        val (ndarray): matrix to be inserted in structure
        """
        if not isinstance(val, np.ndarray):
            raise Exception("Array is expected")

        s = np.array(start)
        e = s + np.array(val.shape) * step
        cs = np.array(self._chunk_shape)
        check_start_end(s, e, self._shape, check_end=(np.abs(step) == 1).all())

        blck = (np.floor_divide(s, self._chunk_shape), np.ceil(np.divide(e, self._chunk_shape)).astype(np.uint32))

        gs = s.copy()
        idx = np.zeros(3, dtype=np.int_)
        ids = None
        ls_c, ls, le_c, le = None, None, None, None

        for i in range(blck[0][0], blck[1][0]):
            for j in range(blck[0][1], blck[1][1]):
                for k in range(blck[0][2], blck[1][2]):
                    ls_c, ls = self._global_to_chunk_coords(gs)
                    le_c, le = self._global_to_chunk_coords(e)
                    le[ls_c != le_c] = cs[ls_c != le_c]
                    le_c[ls_c != le_c] = ls_c[ls_c != le_c]
                    if np.any(ls >= le):
                        break

                    ids = np.floor_divide(le - ls, step)
                    tmp = val[idx[0]:idx[0] + ids[0], idx[1]:idx[1] + ids[1], idx[2]:idx[2] + ids[2]]

                    # special case speedup - skip fill_value->fill_value
                    if (i, j, k) in self._grid.keys() or np.any(tmp != self.fill_value):
                        ret = self.get_chunk((i, j, k))
                        ret[ls[0]:le[0]:step[0], ls[1]:le[1]:step[1], ls[2]:le[2]:step[2]] = tmp

                        self.set_chunk((i, j, k), ret)

                    idx[2] += ids[2]
                    gs[2] = self._chunk_to_global_coord(ls_c, np.floor_divide(le - ls - 1, step) * step + ls)[2] + step[2]
                idx[2] = 0
                idx[1] += ids[1]
                gs[2] = s[2]
                gs[1] = self._chunk_to_global_coord(ls_c, np.floor_divide(le - ls - 1, step) * step + ls)[1] + step[1]
            idx[1] = 0
            idx[0] += ids[0]
            gs[1] = s[1]
            gs[0] = self._chunk_to_global_coord(ls_c, np.floor_divide(le - ls - 1, step) * step + ls)[0] + step[0]

    def _simple_get1(self, key):
        bi, li = divmod(key[0], self._chunk_shape[0])
        bj, lj = divmod(key[1], self._chunk_shape[1])
        bk, lk = divmod(key[2], self._chunk_shape[2])

        chunk = self._grid.get((bi, bj, bk))
        if chunk is not None:
            if chunk.shape == (1,):
                return chunk[0]
            else:
                return chunk[li, lj, lk]
        else:
            if (key < np.array([0, 0, 0])).any() or (key >= np.array(self._shape)).any():
                raise IndexError("index out of range.")
            return self.fill_value

    def __getitem__(self, key):
        if np.shape(key) == (3,) and all(isinstance(k, int) for k in key):
            return self._simple_get1(key)

        key = slice_normalize(key, self.shape)
        start = (key[0].start, key[1].start, key[2].start)
        end = (key[0].stop, key[1].stop, key[2].stop)
        step = (key[0].step, key[1].step, key[2].step)

        return self.get(start, end, step)

    def _simple_set1(self, key, val):
        bi, li = divmod(key[0], self._chunk_shape[0])
        bj, lj = divmod(key[1], self._chunk_shape[1])
        bk, lk = divmod(key[2], self._chunk_shape[2])

        chunk = self._grid.get((bi, bj, bk))
        if chunk is not None:
            if chunk.shape == (1,):
                if val != chunk[0]:
                    new_val = np.full(self._chunk_shape, chunk[0])
                    new_val[li, lj, lk] = val
                    self.set_chunk((bi, bj, bk), new_val)
                return
            else:
                chunk[li, lj, lk] = val
                return

        # not in grid
        if (key < np.array([0, 0, 0])).any() or (key >= np.array(self._shape)).any():
            raise IndexError("index out of range.")

        # new chunk
        # don't do this:
        # new_val = np.full(self._chunk_shape, self._default_value)
        new_val = self.get_chunk((bi, bj, bk))
        new_val[li, lj, lk] = val
        self.set_chunk((bi, bj, bk), new_val)

    def __setitem__(self, key, val):
        # simple case (single element)
        if (np.shape(val) == () and np.shape(key) == (3,) and all(isinstance(k, int) for k in key)):
            self._simple_set1(key, val)
            return

        # general case
        key = slice_normalize(key, self.shape)
        shape = slice_shape(key, self.shape)

        if np.product(shape) == 0:
            return

        if np.shape(val) != shape:
            # enable broadcasting, e.g. a single value
            val = np.broadcast_to(val, shape)

        start = tuple(k.start for k in key)
        step = tuple(k.step for k in key)
        return self.set(start, val, step)

    def crop_chunks(self, crop=((0, 0), (0, 0), (0, 0))):
        """
        Crops chunks around Sparse
        :param: number of full chunks to crop ((z_before, z_after), (y_before, y_after), (x_before, x_after))
        :returns: new Sparse object
        """
        crop = np.array(crop)
        shape = tuple(
            [self.chunks[dim] * (((self.shape[dim] - 1) // self.chunks[dim] + 1) - crop[dim][0] - crop[dim][1])
             for dim in range(3)]
        )
        origin = tuple(
            [self.origin[dim] + crop[{0: 2, 1: 1, 2: 0}[dim]][0] * self.chunks[dim] * self.spacing[dim] for dim in
             range(3)])
        sparse_new = Sparse.empty_like(self, shape=shape, origin=origin)

        chunk_min = crop[:, 0]
        chunk_max = sparse_new.cdata_shape + crop[:, 0]
        for k in self._grid.keys():
            if not np.any(np.array(k) < chunk_min) and not np.any(chunk_max < np.array(k)):
                sparse_new.set_chunk(tuple(k - crop[:, 0]), self.get_chunk(k))
        return sparse_new

    def pad_chunks(self, pad=((0, 0), (0, 0), (0, 0))):
        """
        Pads chunks around Sparse
        :param: number of full chunks to pad ((z_before, z_after),(y_before, y_after), (x_before, x_after)).
                shape is *always* padded to the nearest chunk size multiple.
        :returns: new Sparse object
        """
        pad = np.array(pad)
        shape = tuple(
            [self.chunks[dim] * ((self.shape[dim] - 1) // self.chunks[dim] + 1 + pad[dim][0] + pad[dim][1]) for dim in
             range(3)])
        origin = tuple(
            [self.origin[dim] - pad[{0: 2, 1: 1, 2: 0}[dim]][0] * self.chunks[dim] * self.spacing[dim] for dim in
             range(3)])
        sparse_new = Sparse.empty_like(self, shape=shape, origin=origin)
        for k in self._grid.keys():
            sparse_new.set_chunk(tuple(pad[:, 0] + k), self.get_chunk(k))
        return sparse_new

    def get_k3d_voxels_group_dict(self, dtype=None):
        """
        Generator returns chunks as dict for use in K3D voxels group.

        :param:
        dtype (numpy dtype): returned data dtype

        :return:
        list of dicts
        """
        if dtype is None:
            dtype = self.dtype
        ret = []
        for k in self._grid.keys():
            coord = np.array(k, dtype=np.int) * self._chunk_shape
            mask = (coord + self._chunk_shape) > self._shape
            end = self._chunk_shape - np.multiply(coord + self._chunk_shape - self._shape, mask.astype(np.int))
            ret.append({'voxels': self.get_chunk(k)[:end[0], :end[1], :end[2]].astype(dtype),
                        'coord': coord[::-1],
                        'multiple': 1})
        return ret

    def get_voxels_bounds(self):
        dx, dy, dz = self.spacing[0] / 2.0, self.spacing[1] / 2.0, self.spacing[2] / 2.0

        return [self.origin[0] - dx, self.origin[0] + self.spacing[0] * (self._shape[2] - 1) + dx,
                self.origin[1] - dy, self.origin[1] + self.spacing[1] * (self._shape[1] - 1) + dy,
                self.origin[2] - dz, self.origin[2] + self.spacing[2] * (self._shape[0] - 1) + dz]

    def get_bounds(self):
        return [self.origin[0], self.origin[0] + self.spacing[0] * (self._shape[2] - 1),
                self.origin[1], self.origin[1] + self.spacing[1] * (self._shape[1] - 1),
                self.origin[2], self.origin[2] + self.spacing[2] * (self._shape[0] - 1)]

    def copy_from(self, sparse):
        """
        Copy **data only** from another Sparse.
        :param sparse:
        :return:
        """
        for k in sparse._grid.keys():
            coord = sparse._chunk_to_global_coord(k, (0, 0, 0))
            chunk = sparse.get_chunk(k, dtype=self.dtype)
            s = chunk.shape

            self.set(coord, chunk[
                            0:min(s[0], self.shape[0] - coord[0]),
                            0:min(s[1], self.shape[1] - coord[1]),
                            0:min(s[2], self.shape[2] - coord[2])
                            ])

    @staticmethod
    def empty_like(sparse, shape=None, chunks=None, fill_value=None, dtype=None, origin=None, spacing=None):
        return Sparse(shape if shape is not None else sparse.shape,
                      chunks if chunks is not None else sparse.chunks,
                      fill_value if fill_value is not None else sparse.fill_value,
                      dtype if dtype is not None else sparse.dtype,
                      origin if origin is not None else sparse.origin,
                      spacing if spacing is not None else sparse.spacing)

    @staticmethod
    def create_from_k3d_voxels_group_dict(k3dvg, shape=None):
        """
        Create Sparse object from K3D voxel group

        :param k3dvg: voxel group list
        :param shape: shape of the full original geometry (computed with chunk-size precision if None)
        :returns: Sparse object
        """
        geom_shape = chunk_shape = (0, 0, 0)
        for chunk in k3dvg:
            geom_shape = np.maximum(geom_shape, chunk['coord'][::-1])
            chunk_shape = np.maximum(chunk_shape, chunk['voxels'].shape)
        geom_shape += chunk_shape
        if shape is not None:
            geom_shape = shape

        s = Sparse(geom_shape, chunks=chunk_shape)

        for chunk in k3dvg:
            filler = np.zeros(chunk_shape)
            filler[:chunk['voxels'].shape[0], :chunk['voxels'].shape[1], :chunk['voxels'].shape[2]] = chunk['voxels']
            s.set_chunk(tuple(chunk['coord'][::-1] // chunk_shape), filler)

        return s

    def save(self, filename, compression_level=0):
        with open(filename, "wb") as file:
            d = {
                "chunks": self.chunks,
                "shape": self.shape,
                "origin": self.origin,
                "spacing": self.spacing,
                "dtype": np.dtype(self.dtype).str,
                "fill_value": self.fill_value,
                "_grid": {k: v.to_dict() for k, v in self._grid.items()}
            }

            data = msgpack.packb(d, default=m.encode)

            if compression_level > 0:
                data = zlib.compress(data, compression_level)

            file.write(data)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as file:
            d = file.read()

            msgpack_args = {
                'object_hook': m.decode,
                'use_list': False,
                'strict_map_key': False,  # tuple as "map key" (msgpack 1.0)
                'raw': True,  # return keys as b'shape' (msgpack 1.0)
            }

            try:
                data = msgpack.unpackb(d, **msgpack_args)
            except msgpack.exceptions.ExtraData:
                # FIXME: not a great way to detect compression, to say the least.
                data = msgpack.unpackb(zlib.decompress(d), **msgpack_args)

            s = Sparse(data[b'shape'], chunks=data[b'chunks'], fill_value=data[b'fill_value'],
                       dtype=np.dtype(data[b'dtype']), origin=data[b'origin'], spacing=data[b'spacing'])

            s._grid = {k: Chunk.from_dict(v) for k, v in data[b'_grid'].items()}

            return s

    def copy(self):
        s = Sparse(self.shape, chunks=self.chunks, fill_value=self.fill_value,
                   dtype=self.dtype, origin=self.origin, spacing=self.spacing)

        s._grid = {k: Chunk.from_dict({
            b'origin': v.origin,
            b'spacing': v.spacing,
            b'ndarray': np.copy(v)
        }) for k, v in self._grid.items()}

        return s
