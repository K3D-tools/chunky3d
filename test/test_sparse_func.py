import os
import unittest

import numpy as np

from chunky3d import Sparse
from chunky3d.sparse_func import (
    contour,
    dilate,
    label,
    thinning,
    to_indices_value,
    unique,
    _have_itk,
    _have_itk_thickness,
    _have_nx,
    _have_sitk,
    _have_vtk,
)
import chunky3d.sparse_func as sf


@unittest.skipUnless(_have_vtk, 'this test needs VTK')
class TestContour(unittest.TestCase):
    def test_contour(self):
        shape = (100, 100, 100)
        sp = Sparse(shape=shape)
        step = 0

        step += 1
        sp[50, 50, 50] = 100
        c = contour(sp, 100)
        self.assertEqual(c.GetNumberOfCells(), 0)
        self.assertEqual(c.GetNumberOfPoints(), 0)

        step += 1
        sp[50, 50, 50] = 101
        c = contour(sp, 100)
        self.assertEqual(c.GetNumberOfPoints(), 6)
        self.assertEqual(c.GetNumberOfCells(), 8)


class TestFunctions(unittest.TestCase):
    def test_to_indices_value(self):
        sp = Sparse(shape=(100, 100, 100))
        sp[1, 2, 3] = 4
        sp[50, 60, 70] = 80
        result = to_indices_value(sp)
        result.sort(axis=0)
        np.testing.assert_array_equal(
            result,
            [[1, 2, 3, 4], [50, 60, 70, 80]],
            verbose=True,
        )

    def test_sum(self):
        sp = Sparse(shape=(10, 10, 10))
        sp[0, 2, 1] = 2
        sp[4, 3, 5] = 4
        self.assertEqual(sf.sum(sp), 6)

    def test_max(self):
        sp = Sparse(shape=(10, 10, 10))
        sp[0, 2, 1] = 2
        sp[4, 3, 5] = 4
        self.assertEqual(sf.max(sp), 4)
        sp[4, 4, 3] = np.nan
        self.assertTrue(np.isnan(sf.max(sp)))
        self.assertEqual(sf.nanmax(sp), 4)

    def test_min(self):
        sp = Sparse(shape=(10, 10, 10))
        sp[0, 2, 1] = 2
        sp[4, 3, 5] = 4
        self.assertEqual(sf.min(sp), 0)
        sp[0, 2, 2] = -2
        sp[4, 3, 3] = -4
        self.assertEqual(sf.min(sp), -4)
        sp[4, 4, 3] = np.nan
        self.assertTrue(np.isnan(sf.min(sp)))
        self.assertEqual(sf.nanmin(sp), -4)

    def test_unique(self):
        sp = Sparse(shape=(10, 10, 10))
        sp[0, 2, 1] = 2
        sp[4, 3, 5] = 4
        self.assertSetEqual(unique(sp), {0, 2, 4})

    @unittest.skipUnless(_have_sitk and _have_nx, 'this test needs SimpleITK and NetworkX')
    def test_label(self):
        s = Sparse((30, 30, 30), dtype=np.uint32)
        s[2:5, 2:5, 2:5] = 1
        s[7:9, 7:9, 7:9] = 1
        s[15:18, 15:18, 15:18] = 1
        s[25:28, 25:28, 25:28] = 1
        label(s)
        self.assertSetEqual(unique(s), set(range(5)))

    @unittest.skipUnless(_have_sitk, 'this test needs SimpleITK')
    def test_dilate(self):
        s = Sparse((15, 15, 15), dtype=np.uint8)
        s[3, 3, 3] = 1
        dilate(s, [1])  # also: dilate(s, 1) and dilate(s, (1, 1, 1))
        expected = np.array([
            # as you see, corners are still empty, because this is
            # itk::BinaryBallStructuringElement (a ball kernel)
           [[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],

           [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]],

           [[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]]
        ], dtype=np.uint8)
        np.testing.assert_array_equal(s[2:5, 2:5, 2:5], expected)
        self.assertEqual(sf.sum(s), 3**3 - 8)

    @unittest.skipUnless(_have_itk and _have_itk_thickness, 'this test needs ITK and itk-thickness3d')
    def test_thinning(self):
        s = Sparse((5, 5, 5), dtype=np.uint16)
        s[...] = 1
        s[2, 2] = 0
        expected_slice = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)
        ss = {1: s, 2: s.copy()}
        for mp in (1, 2):
            with self.subTest(multiprocesses=mp):
                ss = s.copy()
                thinning(ss, (0, 0, 0), multiprocesses=mp)
                np.testing.assert_array_equal(ss[..., 2], expected_slice)


if __name__ == '__main__':
    unittest.main()
