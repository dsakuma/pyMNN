import memory as mem
import numpy as np
import os

from functools import lru_cache
from os import path
from PIL import Image
from timeit import default_timer as timer


data_dir = 'data_gen'


class InputData:
    def __init__(self, size):
        p = 'data_dir/{0}'.format(size)
        if not path.exists(p):
            raise ValueError('invalid size or data does not exist')

        self._path = p
        self._files = os.listdir(self._path)
        self._size = size

        if not all(f.endswith('png') for f in self._files):
            raise RuntimeError('there are some non-png files in the folder')

    def load_n_random(self, n):
        to_load = np.random.choice(self._files, replace=False, size=n)
        return _load_img(f) for f in to_load

    @lru_cache
    def _load_img(self, path):
        im = Image.open(path)
        bands = im.getbands()
        s = self._size

        assert len(bands) <= 2
        return np.reshape(np.array(im.getdata(0)) / 255, (s, s))


def test_word_length(size, repeats, n_samples):
    print('test_word_length')
    data = InputData(size)

    for n_samples_inst in n_samples:
        sample_list = list(data.load_n_random(n_samples_inst))
        builder = mem.AutoMemoryBuilder((size, size), n_samples_inst)

        start = timer()
        for s in sample_list:
            builder.append(s)


if __name__ == '__main__':
    repeats = 100
    n_samples_in = [10, 20, 50]

    test_word_length()
