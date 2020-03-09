from chunky3d import Sparse, point_probe

import unittest
import os

import numpy as np

class TestImage(unittest.TestCase):
    def test_zero(self):
        shp = (10, 10, 10)

        chunk_shape = 32

        s = Sparse(shape=np.multiply(shp, chunk_shape), chunks=chunk_shape)

        zeros = np.zeros(shape=(chunk_shape, chunk_shape, chunk_shape))

        for i in range(shp[0]):
            for j in range(shp[1]):
                for k in range(shp[2]):
                    self.assertTrue(np.array_equal(s.get_chunk((i, j, k)), zeros))

    def test_set_get(self):
        shp = (10, 10, 10)
        chunk_shape = 32

        dims = np.multiply(shp, chunk_shape)
        print(dims)

        s = Sparse(shape=dims, chunks=chunk_shape)

        test_data = np.zeros(shape=dims)

        test_data[:dims[0] // 3, :dims[1] // 4, 32] = 3.14

        test_data[128:160, 128:161, 128:159] = np.random.rand(32, 33, 31)

        s.set((0, 0, 0), test_data)

        for i in range(shp[0]):
            for j in range(shp[1]):
                for k in range(shp[2]):
                    self.assertTrue(
                        np.array_equal(s.get_chunk((i, j, k)), test_data[i * chunk_shape:(i + 1) * chunk_shape,
                                                               j * chunk_shape:(j + 1) * chunk_shape,
                                                               k * chunk_shape:(k + 1) * chunk_shape]))

    def test_getitem(self):
        shp = (234, 231, 128)
        mga = np.random.rand(shp[0], shp[1], shp[2])
        s = Sparse(shp, dtype=np.float_, chunks=(16, 32, 8), fill_value=0)
        s.set((0, 0, 0), mga)

        w2 = s[-138:-10:2, -223:-39:5, 120:128:1]
        w3 = mga[-138:-10:2, -223:-39:5, 120:128:1]

        self.assertTrue(np.all(np.equal(w2, w3)))

    def test_getitem_simple_vs_slice(self):
        shp = (234, 231, 128)
        mga = np.random.rand(*shp)
        s = Sparse(shp, dtype=np.float_, chunks=(16, 32, 8), fill_value=0)
        s.set((0, 0, 0), mga)

        for _ in range(10):
            i, j, k = map(np.random.randint, shp)
            v_slice = s[i:i+1, j:j+1, k:k+1]
            v_simple = s[i, j, k]
            self.assertEqual(v_simple, v_slice)

    def test_setitem_simple_vs_slice(self):
        shp = (23, 12, 128)
        s_point = Sparse(shp, dtype=np.float_, chunks=(16, 32, 8), fill_value=0)
        s_slice = Sparse(shp, dtype=np.float_, chunks=(16, 32, 8), fill_value=0)

        s_point[0, 1, 2] = 3
        s_slice[0:1, 1:2, 2:3] = 3

        s_point[22, 11, 127] = 4
        s_slice[22:23, 11:12, 127:128] = 4

        for _ in range(200):
            i, j, k = map(np.random.randint, shp)
            val = np.random.random()
            s_slice[i:i+1, j:j+1, k:k+1] = val
            s_point[i, j, k] = val

        self.assertTrue((s_slice[:, :, :] == s_point[:, :, :]).all())

    def save_load(self, compression_level):
        shp = (67, 87, 33)
        mga = np.random.randint(0, 100, shp, dtype=np.uint8)
        s = Sparse(shp, dtype=np.uint8, chunks=(16, 32, 8), fill_value=0)
        s.set((0, 0, 0), mga)

        s.save('save.msgpack', compression_level)
        loaded_s = Sparse.load('save.msgpack')

        for (k1, v1), (k2, v2) in zip(vars(s).items(), vars(loaded_s).items()):
            self.assertEqual(k1, k2)

            if k1 not in ['_grid', '_memory_blocks']:
                self.assertEqual(v1, v2)

            if k1 == '_grid':
                for (_, d1), (_, d2) in zip(v1.items(), v2.items()):
                    self.assertTrue(np.all(np.equal(d1, d2)))

            if k1 == '_memory_blocks':
                for (d1, d2) in zip(v1, v2):
                    self.assertTrue(np.all(np.equal(d1, d2)))

        os.remove('save.msgpack')


    def test_save_compress(self):
        self.save_load(6)

    def test_save_without_compress(self):
        self.save_load(0)

    def test_memory_usage(self):
        s = Sparse(shape=(64, 64, 64), chunks=(2, 2, 2))
        empty_memory = s.__sizeof__()

        s[0, 0, 0] = 1
        single_chunk = s.__sizeof__()

        s[1, 1, 1] = 1
        single_chunk2 = s.__sizeof__()

        self.assertEqual(single_chunk, single_chunk2)

        s[2, 2, 2] = 1
        self.assertEqual(len(s._memory_blocks), 2)
        two_chunks = s.__sizeof__()

        self.assertAlmostEqual(two_chunks - single_chunk, single_chunk - empty_memory, delta=32)

        s.make_dense_data()
        self.assertEqual(len(s._memory_blocks), 1)
        defragmented = s.__sizeof__()

        self.assertLess(defragmented, two_chunks)

        s.update_grid_mask()
        mask_size = np.zeros(s._block_shape, dtype=np.int32).__sizeof__()
        after_grid_mask = s.__sizeof__()
        self.assertEqual(after_grid_mask, defragmented + mask_size - None.__sizeof__())


class TestBroadcasting(unittest.TestCase):
    def test_too_small(self):
        # base behavior
        n = np.zeros((4, 4, 4))
        with self.assertRaises(ValueError) as cm:
            n[:3, :3, :3] = np.ones((2, 2, 2))
        self.assertTrue(str(cm.exception).startswith('could not broadcast'))

        # chunky behavior
        s = Sparse(shape=(4, 4, 4))

        with self.assertRaises(ValueError) as cm:
            s[:3, :3, :3] = np.ones((2, 2, 2))
        # time will show if this message is constant
        self.assertTrue(str(cm.exception).startswith('operands could not be broadcast together'))

    def test_too_big(self):
        # base behavior
        n = np.zeros((4, 4, 4))
        with self.assertRaises(ValueError) as cm:
            n[:3, :3, :3] = np.ones((4, 4, 4))
        self.assertTrue(str(cm.exception).startswith('could not broadcast'))

        # chunky behavior
        s = Sparse(shape=(4, 4, 4))

        with self.assertRaises(ValueError) as cm:
            s[:3, :3, :3] = np.ones((4, 4, 4))
        # time will show if this message is constant
        self.assertTrue(str(cm.exception).startswith('operands could not be broadcast together'))

    def test_uniform_value(self):
        s = Sparse(shape=(4, 4, 4))
        s[...] = 3
        self.assertTrue((s[...] == np.full((4, 4, 4), 3)).all())
        s[...] = 9
        self.assertTrue((s[...] == np.full((4, 4, 4), 9)).all())

    def test_step_broadcast(self):
        expected = np.zeros((3, 3, 3))
        expected[:2] = [3, 4, 5]
        s = Sparse(shape=(3, 3, 3))
        s[:2] = [3, 4, 5]
        self.assertTrue((s[...] == expected).all())

        expected = np.zeros((3, 3, 3))
        expected[::2] = [3, 4, 5]
        s = Sparse(shape=(3, 3, 3))
        s[::2] = [3, 4, 5]
        self.assertTrue((s[...] == expected).all())


class TestInterpolation(unittest.TestCase):
    def setUp(self):
        self.s = Sparse(shape=(3, 3, 3), chunks=(2, 2, 2))
        self.s[1, 1, 1] = 1.0
        self.s[2, 2, 2] = 2.0
        self.s.make_dense_data()
        self.s.update_grid_mask()

    def test_grid_mask_exception(self):
        s = Sparse(shape=(2, 2, 2), chunks=(2, 2, 2))
        s[1, 1, 1] = 1.0
        # RuntimeError: Missing grid_mask in sparse array. Use update_grid_mask() before point_probe().
        with self.assertRaises(RuntimeError):
            point_probe(np.zeros((1, 3)), s)

    def test_non_contiguous_memory_blocks(self):
        s = Sparse(shape=(3, 3, 3), chunks=(2, 2, 2))
        s[1, 1, 1] = 1.0
        s[2, 2, 2] = 1.0
        # Exception: Memory blocks have wrong len: 2, use make_dense_data() to fix.
        with self.assertRaises(Exception):
            point_probe(np.zeros((1, 3)), s)

    def test_empty_zeros(self):
        self.assertEqual(point_probe(np.zeros((1, 3)), self.s), 0.0)

    def test_flat_xyz(self):
        self.assertEqual(point_probe(np.ones(3), self.s), 1.0)

    def test_middle_of_voxel(self):
        self.assertEqual(point_probe(np.ones((1, 3)) * 0.5, self.s), 0.125)

    def test_list_xyz(self):
        self.assertEqual(point_probe([0.5, 0.5, 0.5], self.s), 0.125)

    def test_unequal_spacing(self):
        s = Sparse(shape=(2, 2, 2), chunks=(2, 2, 2), spacing=(1, 2, 4))
        s[:2, :2, :2] = np.arange(8).reshape((2, 2, 2))
        s.update_grid_mask()
        self.assertEqual(point_probe([0.5, 0.5, 0.5], s), 1.5)

    def test_shape_corner_probe(self):
        self.assertEqual(point_probe(np.full(3, 2), self.s), 2.0)


if __name__ == '__main__':
    unittest.main()
