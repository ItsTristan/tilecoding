#!/usr/bin/env python
from __future__ import division,print_function
import numpy as np
from math import pi, sqrt, exp

from tilecoder import Tilecoder, rescale
from nnswarm import nnswarm as NNSwarm
import pdb

class TileSwarm:
    def __init__(self, input_ranges, input_densities, num_tilings, num_outputs, \
            lr=.1, momentum=.707):
        self._upper = [r[1] for r in input_ranges]
        self._lower = [r[0] for r in input_ranges]
        self._widths = [u-l for (u,l) in zip(self._upper, self._lower)]
        self._densities = input_densities
        self._offsets = [w/d/num_tilings for w,d in zip(self._widths,self._densities)]

        self._inputs = len(input_ranges)
        self._outputs = num_outputs
        self._num_tilings = num_tilings

        self._weights = [NNSwarm(input_densities, # [x+1 for x in input_densities],
            [np.array(x) + i*self._offsets[j] for j,x in enumerate(input_ranges)],
            lr, momentum) for i in xrange(num_tilings)]

    def _index_f(self, i, index):
        return [index[j] for j in xrange(len(index))]

    def __setitem__(self, index, new_value):
        delta = self[index] - new_value
        for i in xrange(self._num_tilings):
            self._weights[i][self._index_f(i,index)] -= delta / self._num_tilings

    def __getitem__(self, index):
        s = 0.
        for i in range(self._num_tilings):
            s += np.sum(self._weights[i][self._index_f(i,index)])
        return s


def example():
    from pylab import sin, cos, normal, random
    def targetFunction(*X):
        return sin(X[0]) + cos(X[1]) + normal(0., 0.1)
#        return X[0] + X[1] - 2*X[0]*X[1] + normal(0., 0.1)
#        return (int(X[0]) ^ int(X[1])) & 1

    bounds = [(0, 8)] * 2
    dimensions = [8] * len(bounds)
    num_tilings = 10

    T = TileSwarm(bounds, dimensions, num_tilings, 1)
    consecutive = 0

    try:
        batch_size = 100
        for n in xrange(1000):
            mean_sq_err = 0.0
            for i in xrange(batch_size):
                X = [0] * len(bounds)
                for j in xrange(len(bounds)):
                    X[j] = rescale(random(), 0, 1, *bounds[j])
                Z = targetFunction(*X)
                err = np.linalg.norm(T[X] - Z) ** 2
                T[X] = Z
                mean_sq_err += (err - mean_sq_err) / (i+1)
            print("MSE = {}".format(mean_sq_err))
            if mean_sq_err < 0.01 * 1.05:
                consecutive += 1
                if consecutive >= 3:
                    print("Converged after {} iterations".format(n*batch_size))
                    raise KeyboardInterrupt
            else:
                consecutive = 0
    except KeyboardInterrupt:
        pass

    import matplotlib.pyplot as plt
    nX = 100
    nY = 100
    bwidth = [(x2-x1) for (x1,x2) in bounds]
    if len(bounds) == 1:
        X = [bwidth[0]*i/nX+bounds[0][0] for i in xrange(nX)]
        Z = np.zeros((nX,))
        for i,x in enumerate(X):
            Z[i] = T[(x,)]
        plt.plot(X, Z)
        plt.show()
    if len(bounds) == 2:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X = [bwidth[0]*i/nX+bounds[0][0] for i in xrange(nX)]
        Y = [bwidth[1]*i/nY+bounds[1][0] for i in xrange(nY)]

        Z = np.zeros((nX, nY))
        for i,x in enumerate(X):
            for j,y in enumerate(Y):
                Z[i,j] = T[x,y]
        [X,Y] = np.meshgrid(X, Y)

        ax.plot_surface(X,Y,Z, cmap=plt.get_cmap('hot'))
        plt.show()

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    example()




