import unittest

from chunky3d.helpers import slice_normalize, slice_shape


class SliceHelper(unittest.TestCase):
    def __getitem__(self, key):
        """Helper magic to easily simulate slicing in [] operator."""
        return key

    def test_hack(self):
        """Testing the behavior of __getitem__."""
        self.assertEqual(self[2, 3], (2, 3))
        self.assertEqual(self[1::3, ...], (slice(1, None, 3), ...))
        self.assertEqual(self[..., :5], (Ellipsis, slice(None, 5)))


class TestSliceNormalize(SliceHelper):
    def test_simple(self):
        self.assertEqual(slice_normalize(self[1, 2], (2, 3)), (slice(1, 2, 1), slice(2, 3, 1)))

    def test_shape_extension(self):
        self.assertEqual(slice_normalize(self[0, 0, ...], (2, 2, 2)), (slice(0, 1, 1), slice(0, 1, 1), slice(0, 2, 1)))
        self.assertEqual(slice_normalize(self[0, 0], (2, 2, 2)), (slice(0, 1, 1), slice(0, 1, 1), slice(0, 2, 1)))
        self.assertEqual(slice_normalize(self[0, 0, 0, ...], (2, 2, 2)), (slice(0, 1, 1),) * 3)
        self.assertEqual(slice_normalize(self[...], (2, 2, 2)), (slice(0, 2, 1),) * 3)
        self.assertEqual(slice_normalize(self[:], (2, 2, 2)), (slice(0, 2, 1),) * 3)

    def test_overindexing(self):
        with self.assertRaises(IndexError):
            slice_normalize(self[1, 2, 3, 4], (1, 2, 3))

        # ellipsis doesn't overindex
        slice_normalize(self[0, 1, 2, ...], (1, 2, 3))

    def test_ellipse_error(self):
        with self.assertRaises(IndexError):
            slice_normalize(self[..., ...], (1, 2, 3))

    def test_wraparound(self):
        self.assertEqual(slice_normalize(-3, (10,)), (slice(7, 8, 1),))

    def test_out_of_bound(self):
        with self.assertRaises(IndexError):
            slice_normalize(self[3], (3,))

    def test_bad_slice_type(self):
        # slices
        with self.assertRaises(IndexError):
            slice_normalize(self['a':'b'], (1, 2, 3))

        with self.assertRaises(IndexError):
            slice_normalize(self[0:1:'b'], (1, 2, 3))

        # lone indices
        with self.assertRaises(IndexError):
            slice_normalize(self[None], (1, 2, 3))

        with self.assertRaises(IndexError):
            slice_normalize(self['hello'], (1, 2, 3))

        # no exceptions:
        slice_normalize(self[0:None], (1, 2, 3))

    def test_step_prefix(self):
        self.assertEqual(
            slice_normalize(self[::2], (3, 3, 3)),
            (slice(0, 3, 2), slice(0, 3, 1), slice(0, 3, 1)),
        )

    def test_backward_slice(self):
        self.assertEqual(
            slice_normalize(self[::-1], (3, 3, 3)),
            (slice(2, -1, -1), slice(0, 3, 1), slice(0, 3, 1)),
        )


class TestSliceShape(SliceHelper):
    def test_single(self):
        self.assertEqual(slice_shape(self[3], (5,)), (1,))

    def test_empty(self):
        self.assertEqual(slice_shape(self[4:3], (5,)), (0,))

    def test_backward(self):
        self.assertEqual(slice_shape(self[4:3:-1], (5,)), (1,))

    def test_step(self):
        self.assertEqual(slice_shape(self[1:3:2], (5,)), (1,))
        self.assertEqual(slice_shape(self[1:4:2], (5,)), (2,))
        self.assertEqual(slice_shape(self[4:1:-3], (5,)), (1,))

    def test_3d(self):
        self.assertEqual(slice_shape(self[::2], (3, 3, 3)), (2, 3, 3,))


if __name__ == '__main__':
    unittest.main()
