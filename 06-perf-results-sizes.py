# %%
import matplotlib.pyplot as plt
import numpy as np
import test_util

if __name__ == '__main__':
    test_util.plot_from_pickles('build-insize',
                                param_name='n(samples)',
                                params=[10, 40, 80],
                                transform_xs=lambda xs: np.power(xs, 2))
    plt.xlabel('Input sample size')
    plt.ylabel('Network building time [s]')
    plt.show()

    test_util.plot_from_pickles('build-nsamples',
                                param_name='size',
                                params=['20x20', '50x50', '80x80'])
    plt.xlabel('Number of input samples')
    plt.ylabel('Network building time [s]')
    plt.show()

    test_util.plot_from_pickles('recall-insize',
                                param_name='n(samples)',
                                params=[10, 40, 80],
                                transform_xs=lambda xs: np.power(xs, 2))
    plt.xlabel('Input sample size')
    plt.ylabel('Average sample recall time [s]')
    plt.show()

    test_util.plot_from_pickles('recall-nsamples',
                                param_name='size',
                                params=['20x20', '50x50', '80x80'])
    plt.xlabel('Number of input samples')
    plt.ylabel('Average sample recall time [s]')
    plt.show()
