import os.path
import logging

import vtk
from vtk.util import numpy_support
import numpy as np

logging.debug(("VTK version:", vtk.VTK_VERSION))


# region VTK I/O

# Great explanation of VTK file and data formats are in
# "VTK Textbook" and "VTK User Guide" chapter 12: "Readers" and 19: "VTK File Formats".


def read_vtk(path, return_reader=False):
    path = os.path.abspath(path)
    logging.debug('Reading: "{}".'.format(path))

    _, file_extension = os.path.splitext(path)

    if file_extension.endswith(".vti"):
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(path)

    elif file_extension.endswith(".stl"):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)

    elif file_extension.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)

    elif file_extension.endswith(".vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(path)

    elif file_extension.endswith(".mhd"):
        reader = vtk.vtkMetaImageReader()
        reader.SetFileName(path)

    elif file_extension.endswith(".vtk"):
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(path)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
    else:
        raise Exception("Unsupported file extension.")

    reader.Update()

    if not return_reader:
        return reader.GetOutput()
    return reader.GetOutput(), reader


def save_vti(vti, path):
    _save(vti, path, vtk.vtkXMLImageDataWriter)


def save_vtp(vtp, path):
    _save(vtp, path, vtk.vtkXMLPolyDataWriter)


def save_vtu(vtu, path):
    _save(vtu, path, vtk.vtkXMLUnstructuredGridWriter)


def save_stl(vtp, path):
    _save(vtp, path, vtk.vtkSTLWriter)


def _save(input_data, path, writer_type):
    path = os.path.abspath(path)
    logging.debug('Saving: "{}".'.format(path))
    writer = writer_type()
    writer.SetFileName(path)
    writer.SetInputData(input_data)
    writer.Write()

# endregion

# region Conversions


def add_np_to_vti(vti, arr_np, arr_name, arr_type=None):
    arr = numpy_support.numpy_to_vtk(num_array=arr_np.ravel(), deep=True, array_type=arr_type)
    arr.SetName(arr_name)
    vti.GetPointData().AddArray(arr)


def vti_to_np(vti, array, components=1):
    x, y, z = vti.GetDimensions()
    arr_np = numpy_support.vtk_to_numpy(vti.GetPointData().GetArray(array))
    if components == 1:
        return arr_np.reshape(z, y, x)
    else:
        return arr_np.reshape(z, y, x, components)



def vtp_to_np(vtp, arrays):
    if vtp.GetPolys().GetMaxCellSize() > 3:
        cut_triangles = vtk.vtkTriangleFilter()
        cut_triangles.SetInputData(vtp)
        cut_triangles.Update()
        vtp = cut_triangles.GetOutput()

    result = [
        ('vertices', numpy_support.vtk_to_numpy(vtp.GetPoints().GetData()).astype(np.float32)),
        ('triangles', numpy_support.vtk_to_numpy(vtp.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]),
    ]
    result.extend([(arr, numpy_support.vtk_to_numpy(vtp.GetPointData().GetArray(arr))) for arr in arrays])
    return dict(result)


def vti_to_string(vti):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(vti)
    writer.WriteToOutputStringOn()  # must be before Update call
    writer.Update()
    return writer.GetOutputString()


def vtp_to_string(vtp):
    polydata_writer = vtk.vtkPolyDataWriter()
    polydata_writer.WriteToOutputStringOn()
    polydata_writer.SetInputData(vtp)
    polydata_writer.Update()
    return polydata_writer.GetOutputString()


def string_to_vtp(s):
    reader = vtk.vtkPolyDataReader()
    reader.ReadFromInputStringOn()
    reader.SetInputString(s)
    reader.Update()
    return reader.GetOutput()


def sailfish_vti_to_npy(vti_file, verbose=False, rho_name='rho', v_name='v'):
    """
    Read vti files from sailfish and return
    gets only rho and v
    """
    vti, reader = read_vtk(vti_file, return_reader=True)
    # info = reader.GetInformation()

    field_names = [reader.GetPointArrayName(i) for i in range(reader.GetNumberOfPointArrays())]
    logging.getLogger(vti_file).debug(("fields:", field_names))
    assert (rho_name in field_names and v_name in field_names)

    data_rho = vti_to_np(vti, rho_name)
    data_v = vti_to_np(vti, v_name, components=3)
    data_v = np.rollaxis(data_v, 3, 0)
    # alt.  data = data.transpose(2,1,0)

    return data_rho, data_v

# endregion

# region SimpleITK interop.


def read_sitk(path):
    import SimpleITK as sitk
    path = os.path.abspath(path)
    logging.debug('Reading "{}".'.format(path))
    return sitk.ReadImage(path)


def save_sitk(img_sitk, path):
    import SimpleITK as sitk
    path = os.path.abspath(path)
    logging.debug('Saving "{}".'.format(path))
    sitk.WriteImage(img_sitk, path, False)


def save_sitk_as_vti(img_sitk, path, array_name='data'):
    vti = sitk_to_vti(img_sitk, array_name)
    save_vti(vti, path)


def sitk_to_vti(img, array_name, array_type=None):
    import SimpleITK as sitk
    vti = vtk.vtkImageData()
    vti.SetSpacing(img.GetSpacing())
    vti.SetOrigin(img.GetOrigin())
    vti.SetDimensions(img.GetSize())

    voxels = sitk.GetArrayFromImage(img)
    arr = numpy_support.numpy_to_vtk(num_array=voxels.ravel(), deep=True, array_type=array_type)
    arr.SetName(array_name)
    vti.GetPointData().SetScalars(arr)

    return vti


def vti_to_sitk(vti, array):
    import SimpleITK as sitk
    arr_np = vti_to_np(vti, array)
    arr_sitk = sitk.GetImageFromArray(arr_np)
    arr_sitk.SetOrigin(vti.GetOrigin())
    arr_sitk.SetSpacing(vti.GetSpacing())
    return arr_sitk

# endregion

# region Some strange functions


def probe_vtu(vtu_file='output.vtu', point_data=[[0.050640, 0.027959, 0.05213]], fname=None, shape2d=None,
              verbose=False):
    """
    get values of interpolated from vtu mesh on the set of points (N,3)
    """
    vtu = read_vtk(vtu_file)

    points = vtk.vtkPoints()
    for point in point_data:
        points.InsertNextPoint(*point)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    probe = vtk.vtkProbeFilter()
    probe.SetSourceData(vtu)
    probe.SetInputData(polydata)
    probe.Update()

    out = probe.GetOutput()
    # out.GetBounds()
    # to get points: numpy_support.vtk_to_numpy(out.GetPoints().GetData()).shape
    pd = out.GetAttributesAsFieldData(0)
    log = logging.getLogger(vtu_file)
    if fname is None:
        # all fields
        output = dict()
        for i in range(pd.GetNumberOfArrays()):
            v_interp_on_grid = numpy_support.vtk_to_numpy(pd.GetArray(i))
            if shape2d:
                v_interp_on_grid = v_interp_on_grid.reshape(shape2d)
            log.debug(("appending in output:", pd.GetArrayName(i)))
            output[pd.GetArrayName(i)] = v_interp_on_grid
        assert (len(output) > 0)
        return output
    else:
        field_numbers = [i for i in range(pd.GetNumberOfArrays()) if pd.GetArrayName(i) == fname]
        assert (len(field_numbers) == 1)
        log.debug(("output:", pd.GetArrayName(field_numbers[0])))
        v_interp_on_grid = numpy_support.vtk_to_numpy(pd.GetArray(field_numbers[0]))
        if shape2d:
            return v_interp_on_grid.reshape(shape2d)
        return v_interp_on_grid


def probe_vti(vti_file='output.vti', point_data=[[0.050640, 0.027959, 0.05213]], fname=None, shape2d=None,
              verbose=False):
    """
    get values of interpolated from vti file mesh on the set of points (N,3)
    """
    vti = read_vtk(vti_file)

    points = vtk.vtkPoints()
    for point in point_data:
        points.InsertNextPoint(*point)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    probe = vtk.vtkProbeFilter()
    probe.SetSourceData(vti)
    probe.SetInputData(polydata)
    probe.Update()

    out = probe.GetOutput()
    # out.GetBounds()
    # to get points: numpy_support.vtk_to_numpy(out.GetPoints().GetData()).shape
    pd = out.GetAttributesAsFieldData(0)
    log = logging.getLogger(vti_file)
    if fname is None:
        # all fields
        output = dict()
        for i in range(pd.GetNumberOfArrays()):
            v_interp_on_grid = numpy_support.vtk_to_numpy(pd.GetArray(i))
            if shape2d:
                v_interp_on_grid = v_interp_on_grid.reshape(shape2d)
            log.debug(("appending in output:", pd.GetArrayName(i)))
            output[pd.GetArrayName(i)] = v_interp_on_grid
        assert (len(output) > 0)
        return output
    else:
        field_numbers = [i for i in range(pd.GetNumberOfArrays()) if pd.GetArrayName(i) == fname]
        assert (len(field_numbers) == 1)
        log.debug(("output:", pd.GetArrayName(field_numbers[0])))
        v_interp_on_grid = numpy_support.vtk_to_numpy(pd.GetArray(field_numbers[0]))
        if shape2d:
            return v_interp_on_grid.reshape(shape2d)
        return v_interp_on_grid


def mod_mesh(du, stlfile='c0006.stl', output_fn='surface.vtp',
             write=False, sign=1):
    """
    Move stl in normal direction
    """
    stl = read_vtk(stlfile)
    vertices = numpy_support.vtk_to_numpy(stl.GetPoints().GetData())
    # indices = numpy_support.vtk_to_numpy(stl.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]

    merged = vtk.vtkPolyData()
    merged.DeepCopy(stl)

    # Compute normals to vertices
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputData(merged)
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ComputeCellNormalsOff()
    normalGenerator.SetSplitting(0)
    normalGenerator.SetConsistency(0)
    normalGenerator.Update()

    merged = normalGenerator.GetOutput()
    normals = numpy_support.vtk_to_numpy(merged.GetPointData().GetNormals())

    points = vtk.vtkPoints()

    for normal, pos in zip(normals, vertices):
        points.InsertNextPoint(pos + normal * (-sign * du))

    merged.SetPoints(points)

    return merged


def probe_at_dx(du=0.001, velocity_file=None, stlfile='c0006.stl', output_fn='surface.vtp',
                velocity_name='v [m/s]',
                write=False, mu=1.0, move_mesh=False, sign=1):
    """
    Equidistant points from stl in normal direction
    """
    stl = read_vtk(stlfile)
    vertices = numpy_support.vtk_to_numpy(stl.GetPoints().GetData())
    # indices = numpy_support.vtk_to_numpy(stl.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]
    merged = vtk.vtkPolyData()
    merged.DeepCopy(stl)

    vel_data = read_vtk(velocity_file)

    # Compute normals to vertices
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputData(merged)
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ComputeCellNormalsOff()
    normalGenerator.SetSplitting(0)
    normalGenerator.SetConsistency(0)
    normalGenerator.Update()

    merged = normalGenerator.GetOutput()
    normals = numpy_support.vtk_to_numpy(merged.GetPointData().GetNormals())

    points = vtk.vtkPoints()
    pointsPolyData = vtk.vtkPolyData()

    for normal, pos in zip(normals, vertices):
        points.InsertNextPoint(pos + normal * (-sign * du))

    pointsPolyData.SetPoints(points)
    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetInputData(pointsPolyData)
    probe_filter.SetSourceData(vel_data)
    probe_filter.GetOutputPort()
    probe_filter.Update()

    probed_data = probe_filter.GetOutput().GetPointData()

    if isinstance(velocity_name, str):
        v_vec = numpy_support.vtk_to_numpy(probed_data.GetArray(velocity_name))
    elif isinstance(velocity_name, list):
        vx = numpy_support.vtk_to_numpy(probed_data.GetArray(velocity_name[0]))
        vy = numpy_support.vtk_to_numpy(probed_data.GetArray(velocity_name[1]))
        vz = numpy_support.vtk_to_numpy(probed_data.GetArray(velocity_name[2]))
        v_vec = np.stack((vx, vy, vz), 1).reshape((-1, 3))
    else:
        raise NotImplementedError

    logging.debug(v_vec.shape)

    velocity = vtk.vtkFloatArray()
    velocity.SetNumberOfComponents(1)
    velocity.SetName("X_at_epsilon")

    for v in v_vec:
        # if np.max(v) > 1e33 or np.min(v) < -1e33:
        #     s = np.array([0.00])

        velocity.InsertNextTypedTuple([v])

    merged.GetPointData().AddArray(velocity)

    if move_mesh:
        merged.SetPoints(points)
        logging.debug('moving point by dx inside')

    merged.Modified()
    if write:
        save_vtp(merged, output_fn)

    return merged


def scale_and_trans(vtk_data=None, output=None,
                    scale=1000.0,
                    deltaxyz=[-14.52326308, 180.637182, 161.81502267]):
    """
    Performs scaling (e.g from meters to mm) and translation of the dataset.
    Note that `vtk_data` is reader.GetOutput()
    """
    transform = vtk.vtkTransform()
    transform.Scale(scale, scale, scale)
    transform.Translate(*deltaxyz)
    transformFilter = vtk.vtkTransformFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInputData(vtk_data)
    transformFilter.Update()
    if output is None:
        return transformFilter.GetOutput()
    else:
        save_vtu(transformFilter.GetOutput(), output)

# endregion
