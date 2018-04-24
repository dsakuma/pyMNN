from functools import lru_cache
from PIL import Image
from timeit import default_timer as timer

import logging
import matplotlib.pyplot as plt
import memory as mem
import numpy as np
import os
import pickle


data_dir = 'data_gen'
data_out = 'data_out'


class InputData:
    def __init__(self, size):
        p = '{0}/{1}'.format(data_dir, size)
        if not os.path.exists(p):
            raise ValueError('invalid size {0} or data does not exist'.format(size))

        self._path = p
        self._files = os.listdir(self._path)
        self._size = size

        if not all(f.endswith('png') for f in self._files):
            raise RuntimeError('there are some non-png files in the folder')

    def load_n_random(self, n):
        to_load = np.random.choice(self._files, replace=False, size=n)
        return [self._load_img(f) for f in to_load]

    @lru_cache()
    def _load_img(self, path):
        im = Image.open('{0}/{1}'.format(self._path, path))
        bands = im.getbands()
        s = self._size

        assert len(bands) <= 2

        data = np.array(im.getdata(0)) / 255
        return np.reshape(data, (s, s))

    @staticmethod
    def sizes():
        a = [int(s.name) for s in os.scandir(data_dir) if s.is_dir()]
        a.sort()

        return a


def _perform_feeding(mem_builder, samples):
    for s in samples:
        mem_builder.append(s)


def _perform_build(mem_builder):
    return mem_builder.build()


def perform_build(data, size, nsamples, repeat_i, logger):
    print(repeat_i, nsamples)

    sample_list = list(data.load_n_random(nsamples))
    builder = mem.AutoMemoryBuilder((size, size), nsamples, logger)

    start = timer()
    _perform_feeding(builder, sample_list)
    _perform_build(builder)
    end = timer()

    return end - start


def test_word_length(size, repeats, n_samples, logger):
    print('test_word_length({0})'.format(size))
    data = InputData(size)

    times = np.empty((len(n_samples), 2, repeats))
    for si, n_samples_inst in enumerate(n_samples):
        print('\tn_samples={0}'.format(n_samples_inst))

        for i in range(repeats):
            t = perform_build(data, size, n_samples_inst, i, logger)
            times[si, :, i] = t

    return times


def test_nsamples(nsamples, repeats, sizes, logger):
    print('test_nsamples({0})'.format(nsamples))

    times = np.empty((len(sizes), 2, repeats))
    for si, size_inst in enumerate(sizes):
        print('\tsize={0}'.format(size_inst))

        data = InputData(size_inst)
        for i in range(repeats):
            t = perform_build(data, size_inst, nsamples, i, logger)
            times[si, :, i] = t

    return times


def test_recall_size(size, repeats, n_samples, logger):
    print('test_recall_size({0})'.format(size))

    data = InputData(size)
    times = np.empty((len(n_samples), repeats))

    for si, ns in enumerate(n_samples):
        sample_list = list(data.load_n_random(ns))
        builder = mem.AutoMemoryBuilder((size, size), ns, logger)

        _perform_feeding(builder, sample_list)
        memory = _perform_build(builder)

        for i in range(repeats):
            idx = np.random.randint(len(sample_list))
            sample = sample_list[idx]

            start = timer()
            memory.recall(sample, how='w')
            end = timer()

            times[si, i] = end - start

    return times


def test_recall_nsamples(n_samples, repeats, sizes, logger):
    print('test_recall_nsamples({0})'.format(n_samples))
    times = np.empty((len(sizes), repeats))

    for si, size in enumerate(sizes):
        data = InputData(size)
        sample_list = list(data.load_n_random(n_samples))
        builder = mem.AutoMemoryBuilder((size, size), n_samples, logger)

        _perform_feeding(builder, sample_list)
        memory = _perform_build(builder)

        for i in range(repeats):
            idx = np.random.randint(len(sample_list))
            sample = sample_list[idx]

            start = timer()
            memory.recall(sample, how='w')
            end = timer()

            times[si, i] = end - start

    return times


def get_logger(level = 'notset'):
    logger = logging.getLogger('perf')
    ch = logging.StreamHandler()

    logger.setLevel(logging.NOTSET)
    logger.addHandler(ch)

    return logger


def plot_from_pickles(what, param_name, params,
                      transform_xs=None, reduce_ys=None):
    if transform_xs is None:
        transform_xs = lambda xs: xs

    if reduce_ys is None:
        reduce_ys = lambda ys: np.mean(ys)

    in_path = os.path.join(data_out, 'test_perf', what)
    if not os.path.exists(in_path):
        raise ValueError()

    def _parse_f(n):
        return int(os.path.splitext(n)[0]), n

    names = dict(_parse_f(f.name) for f in os.scandir(in_path) if f.is_file())
    len_params = len(params)
    xs = np.array(list(names.keys()))

    # %%
    X = transform_xs(xs)
    Y = np.empty([len_params, len(X)])

    for k, (size, name) in enumerate(names.items()):
        with open(os.path.join(in_path, name), mode='rb') as f:
            data = pickle.load(f)

        for (j, param), times_list in zip(enumerate(params), data):
            Y[j, k] = reduce_ys(times_list)

    # %%
    plt.plot(X, Y.T, '.')
    plt.legend(['{0}={1}'.format(param_name, i) for i in params])
    plt.grid()


def set_random_elems(a, val, prob):
    assert prob >= 0 and prob <= 1
    noise = np.random.choice([True, False], a.shape, p=[prob, 1-prob])

    c = np.copy(a)
    c[noise] = val

    return c


def dilative_bool_noise(im, amount):
    return set_random_elems(im, 1, amount)


def erosive_bool_noise(im, amount):
    return set_random_elems(im, 0, amount)
