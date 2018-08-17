import unittest
import numpy as np  # noqa
import KB_python.file_io as file_io

file_prefix = './test_ref_data/file_io'


# ----------------------------------------------
# io tests
# ----------------------------------------------
class test_load_xvg(unittest.TestCase):

    def test_xvg_1D(self):
        data = file_io.load_xvg(file_prefix + '/data_1D.xvg', dims=1)
        self.assertEqual(data.shape[0], 6)
        self.assertEqual(data.shape[1], 10)
        self.assertEqual(data.shape[2], 1)

    def test_xvg_2D(self):
        data = file_io.load_xvg(file_prefix + '/data_2D.xvg', dims=2)
        self.assertEqual(data.shape[0], 6)
        self.assertEqual(data.shape[1], 10)
        self.assertEqual(data.shape[2], 2)

    def test_xvg_3D(self):
        data = file_io.load_xvg(file_prefix + '/data_3D.xvg', dims=3)
        self.assertEqual(data.shape[0], 6)
        self.assertEqual(data.shape[1], 10)
        self.assertEqual(data.shape[2], 3)

    def test_fakedata_3D(self):
        #  make sure reordering is correct, values actually right
        data = file_io.load_xvg(file_prefix + '/fake_3D_data.xvg', dims=3)
        self.assertEqual(data[1, 1, 0], 10)
        self.assertEqual(data[1, 1, 1], 11)
        self.assertEqual(data[1, 1, 2], 12)

    def test_xvg_return_time(self):
        data, time = file_io.load_xvg(file_prefix + '/data_3D.xvg', dims=3, return_time_data=True)
        self.assertEqual(time.size, 6)

    def test_xvg_comments(self):
        data = file_io.load_xvg(file_prefix + '/data_&comments.xvg', dims=3, comments=('#', '@', '&'))
        self.assertEqual(data.shape[0], 6)
        self.assertEqual(data.shape[1], 10)
        self.assertEqual(data.shape[2], 3)

    def test_xvg_column_mismatch_error(self):
        self.assertRaises(ValueError, file_io.load_xvg, file_prefix + '/data_1D.xvg', dims=3)


class test_load_large_text_file(unittest.TestCase):
    def test_loads_data(self):
        data = file_io.load_large_text_file(file_prefix + '/fake_3D_data.xvg', verbose=False)
        self.assertEqual(data[1, 3], 9)
        self.assertEqual(data.shape, (3, 7))

    def test_raises_mismatch_error(self):
        self.assertRaises(Exception, file_io.load_large_text_file, file_prefix + '/data_missing_columns.xvg')

    def test_ignores_trailing_whitespace(self):
        data = file_io.load_large_text_file(file_prefix + '/data_trailing_whitespace.xvg', verbose=False)
        self.assertEqual(data.shape, (3, 7))


class test_load_gromacs_index(unittest.TestCase):
    def test_load_index(self):
        indices = file_io.load_gromacs_index('test_ref_data/file_io/gromacs_index.ndx')
        self.assertEqual(indices['data1'], [0, 1, 2] )   # index is one less than found in index file
        self.assertEqual(indices['data2'], [3, 4, 5, 6, 7, 8])
        self.assertEqual(indices['data3'], [9])
        self.assertNotIn( 'data4', indices.keys())   # data4 is empty, don't make a key


if __name__ == '__main__':
    unittest.main()
