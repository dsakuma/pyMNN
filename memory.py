from numba import cuda
import mnn
import numpy as np


class AutoMemoryBuilder:
    def __init__(self, x_shape, n_samples):
        if len(x_shape) == 1:
            self._x_size = x_shape[0]
        else:
            assert len(x_shape) == 2
            self._x_size = x_shape[0] * x_shape[1]

        assert self._x_size >= n_samples

        self._n_samples = n_samples
        self._xmem_shape = (n_samples, self._x_size)

        self._i = 0
        self._pinned_mem = cuda.pinned_array(self._xmem_shape)
        self._stream = cuda.stream()


    def append(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if self.is_full():
            raise RuntimeError('Memory is full')

        if np.size(x) != self._x_size:
            raise ValueError('Wrong sample size')

        self._pinned_mem[self._i] = x.reshape(-1)
        self._i += 1

    def is_full(self):
        assert self._i <= self._n_samples
        return self._i == self._n_samples

    def build(self):
        if not self.is_full():
            raise RuntimeError('Memory is not full')

        dev_mem = cuda.to_device(self._pinned_mem)
        mem_shape = (self._x_size, self._x_size)

        morph_mem_m = cuda.device_array(mem_shape)
        morph_mem_w = cuda.device_array(mem_shape)

        dev_mem_t = dev_mem.T

        mnn.mat_morph_mul_max_minus(dev_mem_t, dev_mem, morph_mem_m)
        mnn.mat_morph_mul_min_minus(dev_mem_t, dev_mem, morph_mem_w)

        return Memory(morph_mem_m, morph_mem_w, self._x_size)


class Memory:
    def __init__(self, dev_mem_m, dev_mem_w, x_size):
        assert dev_mem_m.shape == (x_size, x_size)
        assert dev_mem_w.shape == (x_size, x_size)

        self._dev_mem_m = dev_mem_m
        self._dev_mem_w = dev_mem_w
        self._x_size = x_size

    def recall(self, x, how):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        assert np.size(x) == self._x_size

        orig_shape = x.shape
        x = x.reshape((self._x_size, 1))
        out = np.empty_like(x)

        if how == 'm':
            mem = self._dev_mem_m
            op = mnn.mat_morph_mul_min_plus
        elif how == 'w':
            mem = self._dev_mem_w
            op = mnn.mat_morph_mul_max_plus
        else:
            raise ValueError('Expected how=\'m\' or how=\'w\'')

        stream = cuda.stream()

        dev_x = cuda.to_device(x, stream=stream)
        out_x = cuda.device_array_like(out, stream=stream)

        op(mem, x, out_x, stream=stream)
        out_x.copy_to_host(out, stream=stream)

        stream.synchronize()

        return out.T.reshape(orig_shape)


################################################################################
import unittest


class AutoMemoryBuilderTest(unittest.TestCase):
    def test_smoke(self):
        builder = AutoMemoryBuilder(x_shape=(1, 4), n_samples=3)
        self.assertFalse(builder.is_full())

        builder.append([4, 2, 5, 6])
        builder.append([2, 3, 8, 4])
        builder.append([2, 5, 1, 4])

        self.assertTrue(builder.is_full())

        mem = builder.build()
        self.assertIsInstance(mem, Memory)

        np.testing.assert_array_equal(
            np.array([4, 2, 5, 6]),
            mem.recall([4, 2, 5, 6], how='w'))

        np.testing.assert_array_equal(
            np.array([4, 2, 5, 6]),
            mem.recall([0, 2, 5, 6], how='w'))

        np.testing.assert_array_equal(
            np.array([4, 2, 5, 6]),
            mem.recall([4, 2, 5, 6], how='m'))

        np.testing.assert_array_equal(
            np.array([4, 2, 5, 6]),
            mem.recall([255, 2, 5, 6], how='m'))


if __name__ == '__main__':
    unittest.main()
