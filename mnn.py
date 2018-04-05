import numba.cuda as cuda
import numpy as np
import math


@cuda.jit
def mat_morph_max_mul_impl(a, b, c, w, q, h):
    row, col = cuda.grid(2)

    tmp = 0
    if row < h and col < w:
        for i in range(q):
            val = a[row, i] + b[i, col]
            tmp = max(tmp, val)

    c[row, col] = tmp


@cuda.jit
def mat_morph_min_mul_impl(a, b, c, w, q, h):
    row, col = cuda.grid(2)

    tmp = 0
    if row < h and col < w:
        for i in range(q):
            val = a[row, i] + b[i, col]
            tmp = min(tmp, val)

    c[row, col] = tmp


def mat_dot(fn, a, b, c):
    assert a.shape[1] == b.shape[0]
    assert c.shape == (a.shape[0], b.shape[1])

    q = b.shape[0]
    w = b.shape[1]
    h = a.shape[0]

    if w*h > 1024:
        threads_per_block = [32, 32]

        bpg_w = int(math.ceil(w / threads_per_block[0]))
        bpg_h = int(math.ceil(h / threads_per_block[1]))
        blocks_per_grid = [bpg_w, bpg_h]
    else:
        threads_per_block = [w, h]
        blocks_per_grid = [1, 1]

    fn[blocks_per_grid, threads_per_block](a, b, c, w, q, h)


def mat_morph_max_mul(a, b, c):
    mat_dot(mat_morph_max_mul_impl, a, b, c)


def mat_morph_min_mul(a, b, c):
    mat_dot(mat_morph_min_mul_impl, a, b, c)


################################################################################
import unittest


@cuda.jit
def _test_mat_mul_impl(a, b, c, w, q, h):
    row, col = cuda.grid(2)

    tmp = 0
    if row < h and col < w:
        for i in range(q):
            tmp += a[row, i] * b[i, col]

    c[row, col] = tmp


def _test_mat_mul(a, b, c):
    mat_dot(_test_mat_mul_impl, a, b, c)


class TestMnnPackage(unittest.TestCase):
    def test_cuda_mat_dot(self):
        n = 1024
        m = 512

        a = np.random.randint(0, 10, size=n*m).reshape([n, m])
        b = np.random.randint(0, 10, size=n*m).reshape([m, n])

        actual = np.zeros([n, n])
        _test_mat_mul(a, b, actual)

        expected = a @ b

        np.testing.assert_array_equal(expected, actual)

    def test_morph_max_mat_mul(self):
        a = np.matrix([[10, 2, -1, 0],
                       [4, -12, 4, 2]])
        b = np.matrix([[-1, 7],
                       [9, 12],
                       [0, 0],
                       [3, -5]])

        actual = np.zeros([2, 2])
        expected = np.matrix([[11, 17],
                              [5, 11]])

        mat_morph_max_mul(a, b, actual)

        np.testing.assert_array_equal(expected, actual)

    def test_morph_min_mat_mul(self):
        a = np.matrix([[10, 2, -1, 0],
                       [4, -12, 4, 2]])
        b = np.matrix([[-1, 7],
                       [9, 12],
                       [0, 0],
                       [3, -5]])

        actual = np.zeros([2, 2])
        expected = np.matrix([[-1, -5],
                              [-3, -3]])

        mat_morph_min_mul(a, b, actual)

        np.testing.assert_array_equal(expected, actual)


if __name__ == '__main__':
    mat = np.matrix([[4, 2, 2],
                     [2, 3, 5],
                     [5, 8, 1],
                     [6, 4, 4]])

    mat_t = np.transpose(mat)

    out = np.zeros((4, 4))

    mat_morph_max_mul(mat, -mat_t, out)
    print(out)

    unittest.main()
