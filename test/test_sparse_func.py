import os
import unittest

import numpy as np

from chunky3d import Sparse, point_probe
from chunky3d.sparse_func import contour
from chunky3d.vtk_utils import read_vtk, save_stl, save_vtp, save_vti, vti_to_np, add_np_to_vti


class TestContour(unittest.TestCase):
    def test_contour(self):
        # import warnings
        # warnings.filterwarnings('error')
        shape = (100, 100, 100)
        sp = Sparse(shape=shape)
        step = 0

        step += 1
        sp[50, 50, 50] = 100
        c = contour(sp, 100)
        self.assertEqual(c.GetNumberOfCells(), 0)
        self.assertEqual(c.GetNumberOfPoints(), 0)
        #save_vtp(c,f"test_contour_step{step}_{'_'.join([str(s) for s in shape])}.vtp")

        step+=1
        sp[50, 50, 50] = 101
        c = contour(sp, 100)
        self.assertEqual(c.GetNumberOfPoints(), 6)
        self.assertEqual(c.GetNumberOfCells(), 8)
        #save_vtp(c,f"test_contour_step{step}_{'_'.join([str(s) for s in shape])}.vtp")


if __name__ == '__main__':
    unittest.main()