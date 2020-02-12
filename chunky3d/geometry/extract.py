import vtk

def _extract(polydata, celltypes):
    
    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfComponents(1)

    unique_points = []
    for i in range(polydata.GetNumberOfCells()):
        c = polydata.GetCell(i)
        if c.GetCellType() in celltypes:
            ids.InsertNextValue(i)
    sn = vtk.vtkSelectionNode()
    sn.SetFieldType(vtk.vtkSelectionNode.CELL)
    sn.SetContentType(vtk.vtkSelectionNode.INDICES)
    sn.SetSelectionList(ids)

    sel = vtk.vtkSelection()
    sel.AddNode(sn)

    es = vtk.vtkExtractSelection()
    es.SetInputData(0, polydata)
    es.SetInputData(1, sel)
    es.Update()

    gf = vtk.vtkGeometryFilter()
    gf.SetInputConnection(es.GetOutputPort())
    return gf


def triangles(polydata):
    gf = _extract(polydata, celltypes=[vtk.VTK_TRIANGLE,])
    gf.Update()
    return gf.GetOutput()

