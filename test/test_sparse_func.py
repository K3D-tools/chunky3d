import os
import unittest

import numpy as np

from chunky3d import Sparse, point_probe
from chunky3d.sparse_func import contour, to_indices_value, unique, label


@unittest.skip
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

    def test_unique(self):
        sp = Sparse(shape=(10, 10, 10))
        sp[0, 2, 1] = 2
        sp[4, 3, 5] = 4
        self.assertSetEqual(unique(sp), {0, 2, 4})

    def test_label(self):
        s = Sparse((30, 30, 30), dtype=np.uint32)
        s[2:5, 2:5, 2:5] = 1
        s[7:9, 7:9, 7:9] = 1
        s[15:18, 15:18, 15:18] = 1
        s[25:28, 25:28, 25:28] = 1
        label(s)
        self.assertSetEqual(unique(s), set(range(5)))


if __name__ == '__main__':
    unittest.main()
