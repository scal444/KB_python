import numpy as np
import unittest
import KB_python.coordinate_manipulation.angles as angles


class test_normalizeVector(unittest.TestCase):
    def test_sumsCorrectDefaultAxis(self):
        array = np.zeros((3, 2, 4))
        result = angles.normalizeVector(array)
        self.assertEqual(result.shape, (3, 2))

    def test_sumsCorrectSpecifiedAxis(self):
        array = np.zeros((3, 2, 4))
        result = angles.normalizeVector(array, 1)
        self.assertEqual(result.shape, (3, 4))

    def test_raisesIndexException(self):
        array = np.zeros(((3, 5, 2)))
        with self.assertRaises(IndexError):
            angles.normalizeVector(array, 3)

    def test_producesCorrectOutput(self):
        array = np.array(((3, 4), (-3, 4), (4, -3)))
        result = angles.normalizeVector(array)
        self.assertAlmostEqual(result[0], 5)
        self.assertAlmostEqual(result[1], 5)
        self.assertAlmostEqual(result[2], 5)


if __name__ == '__main__':
    unittest.main()
