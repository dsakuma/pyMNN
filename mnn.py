import numba.cuda as cuda
import numpy as np
import math


@cuda.jit
def mat_morph_mul_max_plus_impl(a, b, c, stmp, w, q, h):
    row, col = cuda.grid(2)

    tmp = stmp
    if row < h and col < w:
        for i in range(q):
            val = a[row, i] + b[i, col]
            tmp = max(tmp, val)

        c[row, col] = tmp


@cuda.jit
def mat_morph_mul_max_minus_impl(a, b, c, stmp, w, q, h):
    row, col = cuda.grid(2)

    tmp = stmp
    if row < h and col < w:
        for i in range(q):
            val = a[row, i] - b[i, col]
            tmp = max(tmp, val)

        c[row, col] = tmp


@cuda.jit
def mat_morph_mul_min_plus_impl(a, b, c, stmp, w, q, h):
    row, col = cuda.grid(2)

    tmp = stmp
    if row < h and col < w:
        for i in range(q):
            val = a[row, i] + b[i, col]
            tmp = min(tmp, val)

        c[row, col] = tmp


@cuda.jit
def mat_morph_mul_min_minus_impl(a, b, c, stmp, w, q, h):
    row, col = cuda.grid(2)

    tmp = stmp
    if row < h and col < w:
        for i in range(q):
            val = a[row, i] - b[i, col]
            tmp = min(tmp, val)

        c[row, col] = tmp


def mat_dot(fn, stmp, a, b, c, stream=0):
    """
                         m
                     ---------
                    |         |
                  q |    b    |
                    |         |
             q       ---------
         ---------   ---------
        |         | |         |
      n |    a    | |    c    | n
        |         | |         |
         ---------   ---------
                         m
    """
    n = a.shape[0]
    m = b.shape[1]

    assert c.shape == (n, m)
    assert a.shape[1] == b.shape[0]

    q = a.shape[1]

    if n*m > 1024:
        threads_per_block = [32, 32]

        bpg_n = int(math.ceil(n / threads_per_block[0]))
        bpg_m = int(math.ceil(m / threads_per_block[1]))
        blocks_per_grid = [bpg_n, bpg_m]
    else:
        threads_per_block = [n, m]
        blocks_per_grid = [1, 1]

    fn[blocks_per_grid, threads_per_block, stream](a, b, c, stmp, m, q, n)


def mat_morph_mul_max_plus(a, b, c, stream=0):
    mat_dot(mat_morph_mul_max_plus_impl, float('-inf'), a, b, c, stream)


def mat_morph_mul_max_minus(a, b, c, stream=0):
    mat_dot(mat_morph_mul_max_minus_impl, float('-inf'), a, b, c, stream)


def mat_morph_mul_min_plus(a, b, c, stream=0):
    mat_dot(mat_morph_mul_min_plus_impl, float('+inf'), a, b, c, stream)


def mat_morph_mul_min_minus(a, b, c, stream=0):
    mat_dot(mat_morph_mul_min_minus_impl, float('+inf'), a, b, c, stream)


################################################################################
import unittest


@cuda.jit
def _test_mat_mul_impl(a, b, c, stmp, w, q, h):
    row, col = cuda.grid(2)

    tmp = stmp
    if row < h and col < w:
        for i in range(q):
            tmp += a[row, i] * b[i, col]

    c[row, col] = tmp


def _test_mat_mul(a, b, c):
    mat_dot(_test_mat_mul_impl, 0, a, b, c)


class TestMnnPackage(unittest.TestCase):
    def setUp(self):
        self.mat_a = np.matrix(
            [[10, 2, -1, 0],
             [4, -12, 4, 2]])

        self.mat_b = np.matrix(
            [[-1, 7],
             [9, 12],
             [0, 0],
             [3, -5]])

    def test_mat_dot_simple(self):
        '''Smoke test'''
        actual = np.zeros([2, 2])
        expected = np.matrix([[   8,   94],
                              [-106, -126]])

        _test_mat_mul(self.mat_a, self.mat_b, actual)
        np.testing.assert_array_equal(expected, actual)

    def test_cuda_mat_dot(self):
        '''Check whether CUDA kernel is invoked correctly for big matrices'''
        n = 1024
        m = 512

        a = np.random.randint(0, 10, size=n*m).reshape([n, m])
        b = np.random.randint(0, 10, size=n*m).reshape([m, n])

        actual = np.zeros([n, n])
        _test_mat_mul(a, b, actual)
        expected = a @ b
        np.testing.assert_array_equal(expected, actual)

    def test_mat_morph_mul_max_plus(self):
        '''Sanity test'''
        actual = np.zeros([2, 2])
        expected = np.matrix([[11, 17],
                              [ 5, 11]])

        mat_morph_mul_max_plus(self.mat_a, self.mat_b, actual)
        np.testing.assert_array_equal(expected, actual)


    def test_mat_morph_mul_max_minus(self):
        '''Sanity test'''
        actual = np.zeros([2, 2])
        expected = np.matrix([[11, 5],
                              [ 5, 7]])

        mat_morph_mul_max_minus(self.mat_a, self.mat_b, actual)
        np.testing.assert_array_equal(expected, actual)

    def test_mat_morph_mul_min_plus(self):
        '''Sanity test'''
        actual = np.zeros([2, 2])
        expected = np.matrix([[-1, -5],
                              [-3, -3]])

        mat_morph_mul_min_plus(self.mat_a, self.mat_b, actual)
        np.testing.assert_array_equal(expected, actual)

    def test_mat_morph_mul_min_minus(self):
        '''Sanity test'''
        actual = np.zeros([2, 2])
        expected = np.matrix([[ -7, -10],
                              [-21, -24]])

        mat_morph_mul_min_minus(self.mat_a, self.mat_b, actual)
        np.testing.assert_array_equal(expected, actual)

    def test_vector_outer_product(self):
        '''Use case: morphological outer product of two vectors'''
        a = np.array([4, 2, 5, 6]).reshape(1, 4)

        actual = np.zeros([4, 4])
        expected = np.matrix([[ 0,  2, -1, -2],
                              [-2,  0, -3, -4],
                              [ 1,  3,  0, -1],
                              [ 2,  4,  1,  0]])

        mat_morph_mul_max_minus(a.T, a, actual)
        np.testing.assert_array_equal(expected, actual)

    def test_vector_mat_dot(self):
        '''Use case: morphological mat*vec multiplication'''
        a = np.array([4, 2, 5, 6]).reshape(1, 4)
        mat = np.matrix([[-1,  7, 10,   4],
                         [ 9, 12,  2, -12],
                         [ 0,  0, -1,   4],
                         [ 3, -5,  0,   2]])

        out = np.zeros([4, 1])
        expected = np.array([[15,14,10,8]])

        mat_morph_mul_max_plus(mat, a.T, out)
        np.testing.assert_array_equal(expected, out.T)

    def test_bug_mat_mul_10x400(self):
        '''
        Bug test case:
        using mat_morph_mul_max_minus to multiply 10x400 matrix by itself
        '''

        a = np.random.randint(0, 10, size=(10, 400))
        c = np.empty((400, 400))

        mat_morph_mul_max_minus(a.T, a, c)


if __name__ == '__main__':
    unittest.main()
