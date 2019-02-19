import numpy as np
import unittest
import KB_python.coordinate_manipulation.angles as angles


class test_vectorNorm(unittest.TestCase):
    def test_sumsCorrectDefaultAxis(self):
        array = np.zeros((3, 2, 4))
        result = angles.vectorNorm(array)
        self.assertEqual(result.shape, (3, 2, 1))

    def test_sumsCorrectSpecifiedAxis(self):
        array = np.zeros((3, 2, 4))
        result = angles.vectorNorm(array, 1)
        self.assertEqual(result.shape, (3, 1, 4))

    def test_raisesIndexException(self):
        array = np.zeros(((3, 5, 2)))
        with self.assertRaises(IndexError):
            angles.vectorNorm(array, 3)

    def test_producesCorrectOutput(self):
        array = np.array(((3, 4), (-3, 4), (4, -3)))
        result = angles.vectorNorm(array)
        self.assertAlmostEqual(result[0], 5)
        self.assertAlmostEqual(result[1], 5)
        self.assertAlmostEqual(result[2], 5)


class test_anglesFromVectors(unittest.TestCase):

    def test_produces_correct_output(self):
        v1 = np.array((0, 0, 1))
        v2 = np.array((0, 1, 0))
        self.assertAlmostEqual(angles.angleFromVectors(v1, v2), np.pi / 2)
        v2 = np.array((0, 0, -1))
        self.assertAlmostEqual(angles.angleFromVectors(v1, v2), np.pi)
        v2 = np.array((-2, 0, 0))
        self.assertAlmostEqual(angles.angleFromVectors(v1, v2), np.pi / 2)


class test_dihedralFromVectors(unittest.TestCase):

    def test_producesCorrectOutput(self):
        # simple unit case, 180 degrees
        v1 = np.array((0, 1, 0))
        v2 = np.array((1, 0, 0))
        v3 = np.array((0, 1, 0))
        self.assertAlmostEqual(angles.dihedralFromVectors(v1, v2, v3), np.pi)

        # simple unit case, 0 degree
        v3 = np.array((0, -1, 0))
        self.assertAlmostEqual(angles.dihedralFromVectors(v1, v2, v3), 0)

        # simple unit case, 90 degrees
        v3 = np.array((0, 0, 1))
        self.assertAlmostEqual(angles.dihedralFromVectors(v1, v2, v3), np.pi / 2)
        v3 = np.array((0, 0, -1))
        self.assertAlmostEqual(angles.dihedralFromVectors(v1, v2, v3), - np.pi / 2)


if __name__ == '__main__':
    unittest.main()
