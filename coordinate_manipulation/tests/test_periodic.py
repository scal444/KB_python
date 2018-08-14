import numpy as np
import unittest
import KB_python.coordinate_manipulation.periodic as periodic


class test_calc_vectors(unittest.TestCase):
    def test_coordinate_mismatch_exceptions(self):
        cp = np.ones((10, 20, 3))
        cd1 = np.ones((9, 20, 3))    # bad frame number
        cd2 = np.ones((10, 10, 3))   # bad particle number
        cd3 = np.ones((10, 20, 2))   # bad dimension number
        boxdims = np.ones((10, 3))   # correct
        self.assertRaises(ValueError, periodic.calc_vectors, cp, cd1, boxdims)
        self.assertRaises(ValueError, periodic.calc_vectors, cp, cd2, boxdims)
        self.assertRaises(ValueError, periodic.calc_vectors, cp, cd3, boxdims)

    def test_coordinate_boxdim_mismatch_exceptions(self):
        cp = np.ones((10, 20, 3))
        boxdims = np.ones((9, 3))    # not enough frames
        boxdims2 = np.ones((10, 2))  # not enough dimensions
        self.assertRaises(ValueError, periodic.calc_vectors, cp, cp, boxdims)
        self.assertRaises(ValueError, periodic.calc_vectors, cp, cp, boxdims2)

    def test_bad_input_dimension_exceptions(self):
        cp_too_few       = np.ones((100, 3))
        cp_too_many      = np.ones((100, 20, 3, 4))
        cp_good          = np.ones((100, 20, 3))
        boxdims_too_few  = np.ones(100)
        boxdims_too_many = np.ones((100, 3, 10))
        boxdims_good     = np.ones((100, 3))
        self.assertRaises(ValueError, periodic.calc_vectors, cp_too_few, cp_too_few, boxdims_good)
        self.assertRaises(ValueError, periodic.calc_vectors, cp_too_many, cp_too_many, boxdims_good)
        self.assertRaises(ValueError, periodic.calc_vectors, cp_good, cp_good, boxdims_too_few)
        self.assertRaises(ValueError, periodic.calc_vectors, cp_good, cp_good, boxdims_too_many)

    def test_for_correct_vectors(self):
        boxdims = np.array([[10, 10, 10], [9.5, 9.5, 9.5]])   # 2 frames, 3D
        p_low  = np.ones((2, 2, 3))          # = 1
        p_high = np.ones((2, 2, 3)) + 8.4    # = 9.4
        p_mid  = np.ones((2, 2, 3)) + 4      # = 5
        vecs_pos_no_pi   = periodic.calc_vectors(p_mid, p_high, boxdims)
        vecs_neg_no_pi   = periodic.calc_vectors(p_mid,  p_low, boxdims)
        vecs_prev_pi     = periodic.calc_vectors(p_low, p_high, boxdims)
        vecs_next_pi     = periodic.calc_vectors(p_high, p_low, boxdims)
        self.assertAlmostEqual(vecs_pos_no_pi[0, 0, 2], 4.4)
        self.assertAlmostEqual(vecs_neg_no_pi[1, 1, 2], -4)
        self.assertAlmostEqual(vecs_prev_pi[0, 0, 0], -1.6)
        self.assertAlmostEqual(vecs_next_pi[1, 1, 1], 1.1)   # 2nd frame, 9.5 size


if __name__ == '__main__':
    unittest.main()
