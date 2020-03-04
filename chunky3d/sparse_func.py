import math
import warnings
from collections import Counter

import SimpleITK as sitk
import itk
import networkx as nx
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from .helpers import slice_normalize
from .chunky import Sparse
from .vtk_utils import add_np_to_vti
from .geometry.extract import triangles

dilate_filter = sitk.BinaryDilateImageFilter()
erode_filter = sitk.BinaryErodeImageFilter()
closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
opening_filter = sitk.BinaryMorphologicalOpeningImageFilter()
connected_compontent_filter = sitk.ConnectedComponentImageFilter()


def min_dtype(t):
    try:
        t = t.type
    except:
        pass

    if issubclass(t, np.inexact):
        return np.finfo(t).min
    else:
        return np.iinfo(t).min


def max_dtype(t):
    try:
        t = t.type
    except:
        pass

    if issubclass(t, np.inexact):
        return np.finfo(t).max
    else:
        return np.iinfo(t).max


def _unique_pairs(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    return np.unique(b).view(a.dtype).reshape(-1, a.shape[1])


def unique(sparse, return_counts=False, multiprocesses=1):
    """ see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html """
    def process(data, prev):
        if return_counts:
            values, counts = np.unique(data, return_counts=True)
            x = dict(zip(values, counts))

            return data, dict(Counter(x) + Counter(prev))
        else:
            return data, prev | set(np.unique(data))

    result = sparse.run(process, prev={} if return_counts else set(), multiprocesses=multiprocesses)

    result_merged = {} if return_counts else set()
    for process in result:
        if return_counts:
            result_merged = dict(Counter(result_merged) + Counter(process))
        else:
            result_merged |= process
    result = result_merged
    return result


def downscale(sparse, stride=(2, 2, 2)):
    ret = Sparse.empty_like(sparse,
                            chunks=(sparse.chunks[0] // stride[0],
                                    sparse.chunks[1] // stride[1],
                                    sparse.chunks[2] // stride[2]),
                            shape=(sparse.shape[0] // stride[0],
                                   sparse.shape[1] // stride[1],
                                   sparse.shape[2] // stride[2]),
                            spacing=(sparse.spacing[0] * stride[0],
                                     sparse.spacing[1] * stride[1],
                                     sparse.spacing[2] * stride[2]))

    for k in sparse._grid.keys():
        value = sparse._grid[k][0] if sparse._grid[k].shape == (1,) else sparse._grid[k][::stride[0], ::stride[1],
                                                                         ::stride[2]]
        ret.set_chunk(k, value)

    return ret


def sum(sparse, multiprocesses=1):
    return np.sum(sparse.run(lambda data, prev: (data, prev + np.sum(data)), prev=0, multiprocesses=multiprocesses))


def downsample(sparse, stride=(3, 3, 3)):
    sparse_copy = Sparse.empty_like(sparse, chunks=np.array(stride) * 16)
    sparse_copy.copy_from(sparse)

    sparse_downscaled = Sparse(np.array(sparse.shape) // np.array(stride), dtype=np.float32)
    sparse_downscaled.origin = sparse.origin
    sparse_downscaled.spacing = np.array(sparse.spacing) * np.array(stride)

    def reduce(chunk, prev):
        accu = np.zeros((16,) * 3, dtype=np.float32)

        for i in range(stride[0]):
            for j in range(stride[1]):
                for k in range(stride[2]):
                    accu += chunk[i::stride[0], j::stride[1], k::stride[2]]

        coord = sparse_copy._chunk_to_global_coord(chunk.idx, (0, 0, 0)) // stride

        prev[coord[0]:coord[0] + 16,
        coord[1]:coord[1] + 16,
        coord[2]:coord[2] + 16] = accu / (stride[0] * stride[1] * stride[2])

        return chunk, prev

    return sparse_copy.run(reduce, (0, 0, 0), sparse_downscaled)


def sum_slice(sparse, z, y, x):
    def process(idx, gs, z, y, x, prev):
        chunk_size = abs(abs(x.stop - x.start) // x.step) * \
                     abs(abs(y.stop - y.start) // y.step) * \
                     abs(abs(z.stop - z.start) // z.step)

        if idx in sparse._grid.keys():
            if sparse._grid[idx].shape == (1,):
                prev += sparse._grid[idx][0] * chunk_size
            else:
                prev += np.sum(sparse._grid[idx][z, y, x])
        else:
            prev += sparse.fill_value * chunk_size

        return prev

    z, y, x = slice_normalize((z, y, x), sparse.shape)

    return sparse.raw_run(process,
                          (z.start, y.start, x.start),
                          (z.stop, y.stop, x.stop),
                          step=(z.step, y.step, x.step),
                          prev=0)


def where(sparse, func):
    ret_i, ret_j, ret_k = [], [], []

    for key in sorted(sparse._grid.keys()):
        i, j, k = np.where(func(sparse.get_chunk(key)))

        offset = sparse._chunk_to_global_coord(key, (0, 0, 0))
        ret_i.append(i + offset[0])
        ret_j.append(j + offset[1])
        ret_k.append(k + offset[2])

    return np.hstack(ret_i), np.hstack(ret_j), np.hstack(ret_k)


def max(sparse, multiprocesses=1):
    f = np.max
    return f(sparse.run(
                        lambda data, prev: (data, f([prev, f(data)])),
                        prev=min_dtype(sparse.dtype),
                        multiprocesses=multiprocesses,
                        )
            )


def nanmax(sparse, multiprocesses=1):
    f = np.nanmax
    return f(sparse.run(
                        lambda data, prev: (data, f([prev, f(data)])),
                        prev=min_dtype(sparse.dtype),
                        multiprocesses=multiprocesses,
                        )
            )


def max_slice(sparse, z, y, x):
    z, y, x = slice_normalize((z, y, x), sparse.shape)

    return sparse.raw_run(lambda idx, gs, z, y, x, prev: np.max([prev, np.max(sparse.get_chunk(idx)[x, y, x])]),
                          (z.start, y.start, x.start),
                          (z.stop, y.stop, x.stop),
                          step=(z.step, y.step, x.step),
                          prev=-math.inf)


def min(sparse, multiprocesses=1):
    f = np.min
    return f(sparse.run(
                        lambda data, prev: (data, f([prev, f(data)])),
                        prev=max_dtype(sparse.dtype),
                        multiprocesses=multiprocesses,
                        )
            )


def nanmin(sparse, multiprocesses=1):
    f = np.nanmin
    return f(sparse.run(
                        lambda data, prev: (data, f([prev, f(data)])),
                        prev=max_dtype(sparse.dtype),
                        multiprocesses=multiprocesses,
                        )
            )


def min_slice(sparse, z, y, x):
    z, y, x = slice_normalize((z, y, x), sparse.shape)

    return sparse.raw_run(lambda idx, gs, z, y, x, prev: np.min([prev, np.min(sparse.get_chunk(idx)[x, y, x])]),
                          (z.start, y.start, x.start),
                          (z.stop, y.stop, x.stop),
                          step=(z.step, y.step, x.step),
                          prev=math.inf)


def mul(sparse_a, sparse_b):
    return sparse_a.run_multivariate(lambda a, b: a * b[0], [sparse_b])


def add(sparse_a, sparse_b):
    return sparse_a.run_multivariate(lambda a, b: a + b[0], [sparse_b])


def subtract(sparse_a, sparse_b):
    return sparse_a.run_multivariate(lambda a, b: a - b[0], [sparse_b])


def add_scalar(sparse_a, val, multiprocesses=1):
    sparse_a.run(lambda data, prev: (data + val, prev), multiprocesses=multiprocesses)


def dilate(sparse, kernel_radius, foreground_value=1, multiprocesses=1):
    """ see: http://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm """
    dilate_filter.SetKernelRadius(kernel_radius)
    dilate_filter.SetForegroundValue(foreground_value)
    sparse.run(lambda data, prev:
               (sitk.GetArrayFromImage(
                   dilate_filter.Execute(
                       sitk.GetImageFromArray(data)
                   )
               ), prev), envelope=kernel_radius, multiprocesses=multiprocesses)


def erode(sparse, kernel_radius, foreground_value=1, multiprocesses=1):
    """ see: http://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm """
    erode_filter.SetKernelRadius(kernel_radius)
    erode_filter.SetForegroundValue(foreground_value)
    sparse.run(lambda data, prev:
               (sitk.GetArrayFromImage(
                   erode_filter.Execute(
                       sitk.GetImageFromArray(data)
                   )
               ), prev), envelope=kernel_radius, skip_neighbours=True, multiprocesses=multiprocesses)


def closing(sparse, kernel_radius, foreground_value=1, multiprocesses=1):
    """ see: http://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm """
    closing_filter.SetKernelRadius(kernel_radius)
    closing_filter.SetForegroundValue(foreground_value)
    sparse.run(lambda data, prev:
               (sitk.GetArrayFromImage(
                   closing_filter.Execute(
                       sitk.GetImageFromArray(data)
                   )
               ), prev), envelope=kernel_radius, multiprocesses=multiprocesses)


def opening(sparse, kernel_radius, foreground_value=1, multiprocesses=1):
    """ see: http://homepages.inf.ed.ac.uk/rbf/HIPR2/morops.htm """
    opening_filter.SetKernelRadius(kernel_radius)
    opening_filter.SetForegroundValue(foreground_value)
    sparse.run(lambda data, prev: (sitk.GetArrayFromImage(
        opening_filter.Execute(
            sitk.GetImageFromArray(data)
        )
    ), prev), envelope=kernel_radius, skip_neighbours=True, multiprocesses=multiprocesses)


def thinning(sparse, envelope, multiprocesses=1):
    """
    1 pixel-thin wire skeletonization
    """
    sparse.run(lambda data, prev: (itk.GetArrayFromImage(
        itk.BinaryThinningImageFilter3D.New(
            itk.GetImageFromArray(data)
        )
    ), prev), envelope=envelope, skip_neighbours=True, multiprocesses=multiprocesses)


def thinning_diameter(sparse, envelope, multiprocesses=2):
    """
    Computes twice the shortest distance to the outter shell along the medial axis of the object
    (i.e. diameter of the local maximal fitting sphere)
    """
    if multiprocesses == 1:
        warnings.warn(
            'Disabled multiprocessing creates a pool of threads which does not get deallocated. '
            'Subsequent calls with multiprocessing **enabled** will be locked indefinitely.'
        )
    sparse.run(lambda data, prev: (itk.GetArrayFromImage(
        itk.MedialThicknessImageFilter3D.New(
            itk.GetImageFromArray(data)
        )
    ), prev), envelope=envelope, skip_neighbours=True, multiprocesses=multiprocesses)


def to_indices_value(sparse):
    coords = []

    for k in sparse._grid.keys():
        for val in np.unique(sparse._grid[k]):
            if val != sparse.fill_value:
                z, y, x = np.where(sparse._grid[k] == val)
                offset = sparse._chunk_to_global_coord(k, (0, 0, 0))
                x += offset[2]
                y += offset[1]
                z += offset[0]
                coords.append(np.dstack((z, y, x, np.full(x.shape, val))).reshape(-1, 4))

    return np.vstack(coords)


def label(sparse, multiprocesses=1, fully_connected=False):
    """
    Label the objects in a binary image.
    ConnectedComponentImageFilter labels the objects in a binary image (non-zero pixels are considered to be objects, zero-valued pixels are considered to be background). Each distinct object is assigned a unique label. The final object labels start with 1 and are consecutive.
    :param sparse: Sparse instance
    :param multiprocesses: number of processes
    :param fully_connected: whether the connected components are defined strictly by face connectivity or by face+edge+vertex connectivity. Set `True` for objects that are 1 pixel wide.
    """
    if multiprocesses != 1:
        raise NotImplementedError()

    if sparse.dtype != np.uint32:
        warnings.warn('label in most cases require uint32 data to work properly')
        
    connected_compontent_filter.SetFullyConnected(fully_connected)

    G = nx.Graph()

    def process(a, b):
        mask = np.dstack([a, b])
        relations = _unique_pairs(mask[np.sum(mask != 0, axis=2) == 2])

        for r in relations:
            G.add_edge(r[0], r[1])

    def initialize(data, prev):
        result = sitk.GetArrayFromImage(
            connected_compontent_filter.Execute(
                sitk.GetImageFromArray(data)
            )
        )

        obj_count = connected_compontent_filter.GetObjectCount()
        result[result > 0] += prev[1]
        prev[0].append(obj_count)

        return result, (prev[0], prev[1] + obj_count)

    component_count_per_chunk, component_sum = sparse.run(initialize, prev=([], 0), multiprocesses=multiprocesses)[0]

    for k in sparse._grid.keys():
        # Z axis
        nk = (k[0] - 1, k[1], k[2])
        if nk in sparse._grid.keys():
            process(sparse.get_chunk(k)[0, :, :], sparse.get_chunk(nk)[-1, :, :])

        nk = (k[0] + 1, k[1], k[2])
        if nk in sparse._grid.keys():
            process(sparse.get_chunk(k)[-1, :, :], sparse.get_chunk(nk)[0, :, :])

        # Y axis
        nk = (k[0], k[1] - 1, k[2])
        if nk in sparse._grid.keys():
            process(sparse.get_chunk(k)[:, 0, :], sparse.get_chunk(nk)[:, -1, :])

        nk = (k[0], k[1] + 1, k[2])
        if nk in sparse._grid.keys():
            process(sparse.get_chunk(k)[:, -1, :], sparse.get_chunk(nk)[:, 0, :])

        # X axis
        nk = (k[0], k[1], k[2] - 1)
        if nk in sparse._grid.keys():
            process(sparse.get_chunk(k)[:, :, 0], sparse.get_chunk(nk)[:, :, -1])

        nk = (k[0], k[1], k[2] + 1)
        if nk in sparse._grid.keys():
            process(sparse.get_chunk(k)[:, :, -1], sparse.get_chunk(nk)[:, :, 0])

    cutted_G = list(G.subgraph(c) for c in nx.connected_components(G))
    val_map = {v: k + 1 for k, g in enumerate(cutted_G) for v in g}

    for idx, k in enumerate(set(range(1, component_sum + 1)) - val_map.keys()):
        val_map[k] = idx + len(cutted_G) + 1

    add_scalar(sparse, component_sum)
    component_index = 1

    for k, component_count in zip(sorted(sparse._grid.keys()), component_count_per_chunk):
        chunk = sparse.get_chunk(k)

        for _ in range(component_count):
            if component_index in val_map:
                chunk[chunk == (component_index + component_sum)] = val_map[component_index]

            component_index += 1

        chunk[chunk >= component_sum] -= component_sum

        sparse.set_chunk(k, chunk)


def contour(sparse, values, envelope=(1,1,1), quantize_points_factor=0.0, multiprocesses=1):
    """
    Generate isosurface(s) from a Sparse volume
    """
    def chunk_contour(data, prev, *args):
        envelope, values = args

        vti = vtk.vtkImageData()
        vti.SetOrigin(data.origin)
        vti.SetDimensions(np.array(data.shape) - np.array(envelope))
        vti.SetSpacing(data.spacing)
        series_name = "sparse"
        add_np_to_vti(vti, data[:-envelope[0], :-envelope[1], :-envelope[2]], series_name)
        vti.GetPointData().SetActiveScalars(series_name)
        contour = vtk.vtkImageMarchingCubes()
        contour.SetInputData(vti)
        contour.SetNumberOfContours(values.shape[0])
        for i, v in enumerate(values):
            contour.SetValue(i, v)
        contour.ComputeScalarsOn()
        contour.Update()

        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(contour.GetOutput())
        writer.WriteToOutputStringOn()
        writer.Update()
        prev.append(writer.GetOutputString())

        return data, prev

    out = sparse.run(chunk_contour,
                     envelope,
                     np.array([values]).flatten(),
                     envelope=envelope,
                     multiprocesses=multiprocesses,
                     prev=[],
              )

    appendFilter = vtk.vtkAppendPolyData()
    for contour in out:
        for c in contour:
            reader = vtk.vtkPolyDataReader()
            reader.ReadFromInputStringOn()
            reader.SetInputString(c)
            reader.Update()
            appendFilter.AddInputData(reader.GetOutput())

    quantize = vtk.vtkQuantizePolyDataPoints()
    quantize.SetQFactor(quantize_points_factor)

    quantize.SetInputConnection(appendFilter.GetOutputPort())
    quantize.Update()

    return triangles(quantize.GetOutput())
