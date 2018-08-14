import unittest
import numpy as np
import KB_python.coordinate_manipulation.transformations as transformations


class test_cart2pol(unittest.TestCase):

    def test_dimension_exceptions(self):
        coords_bad_shape_1 = np.zeros((3, 3, 3))
        coords_bad_shape_2 = np.ones((4, 4))
        self.assertRaises(ValueError, transformations.cart2pol, coords_bad_shape_1)
        self.assertRaises(ValueError, transformations.cart2pol, coords_bad_shape_2)

    def test_output_shape(self):
        nparts = 20
        cart_coords = np.zeros((nparts, 3))
        theta, rho, z = transformations.cart2pol(cart_coords)
        self.assertEqual(theta.size, nparts)
        self.assertEqual(theta.shape[0], nparts)
        self.assertEqual(rho.size, nparts)
        self.assertEqual(z.size, nparts)

    def test_output_values(self):
        coords = np.array([-4, 3, 2])[np.newaxis, :]
        theta, rho, z = transformations.cart2pol(coords)
        self.assertEqual(z[0], 2)
        self.assertEqual(rho[0], 5.0)  # 3 4 5 triangle

        thetacoords = np.array([[0, 0, 0], [1, 1, 0], [-1, 0, 0], [0, -1, 0]])
        theta, rho, z = transformations.cart2pol(thetacoords)
        self.assertEqual(theta[0], 0)
        self.assertAlmostEqual(theta[1], np.pi / 4)
        self.assertAlmostEqual(theta[2], np.pi)
        self.assertAlmostEqual(theta[3], - np.pi / 2)  # goes negative after pi


if __name__ == '__main__':
    unittest.main()
