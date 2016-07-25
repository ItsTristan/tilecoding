#!/usr/bin/env python
from __future__ import division
import numpy as np
from math import pi, sqrt, exp


class Tilecoder:
    def __init__(self, input_ranges, input_densities, num_tilings, num_outputs, \
            init=lambda d: 0.01*np.random.rand(*d)):
        self._upper = [r[1] for r in input_ranges]
        self._lower = [r[0] for r in input_ranges]
        self._widths = [u-l for (u,l) in zip(self._upper, self._lower)]
        self._densities = input_densities
        self._offsets = [w/d/num_tilings for w,d in zip(self._widths,self._densities)]

        self._inputs = len(input_ranges)
        self._outputs = num_outputs
        self._num_tilings = num_tilings

        self._weights = init([num_tilings] + [d+1 for d in input_densities] + [num_outputs])

    def _index_f(self, X):
        return [range(self._num_tilings)] + \
            [
                [int(rescale(X[i],
                        self._lower[i]-t*self._offsets[i],
                        self._upper[i]-t*self._offsets[i],
                        0, self._densities[i]))
                for t in xrange(self._num_tilings)]
            for i in xrange(self._inputs)]

    def __getitem__(self, index):
        return sum(self._weights[self._index_f(index)])

    def __setitem__(self, index, new_value):
        delta = self[index] - new_value
        self._weights[self._index_f(index)] -= delta / self._num_tilings

def rescale(X, lo, hi, new_lo, new_hi):
    """
    >>> rescale(0, 0, 1, 0, 1)
    0.0
    >>> rescale(1, 0, 1, 0, 1)
    1.0
    >>> rescale(0, -.5, .5, 0, 1)
    0.5
    >>> rescale(0.0, -1, 1, 1, 10)
    5.5
    """
    return (X-lo) / (hi-lo) * (new_hi - new_lo) + new_lo

def example():
    from pylab import sin, cos, normal, random
    def targetFunction(*X):
        return sin(X[0]) + cos(X[1]) + normal(0., 0.1)

    bounds = [(0, 2*pi)] * 2
    dimensions = [8] * len(bounds)
    num_tilings = 10

    T = Tilecoder(bounds, dimensions, num_tilings, 1)

    try:
        batch_size = 50
        twice = False
        for n in xrange(1000):
            mean_sq_err = 0.0
            for i in xrange(batch_size):
                X = [0] * len(bounds)
                for j in xrange(len(bounds)):
                    X[j] = rescale(random(), 0, 1, *bounds[j])
                Z = targetFunction(*X)
                err = np.linalg.norm(T[X] - Z) ** 2
                T[X] += (Z - T[X]) * 0.05
                mean_sq_err += (err - mean_sq_err) / (i+1)
            print "MSE = {}".format(mean_sq_err)
            # Stop if MSE is less than 5% away from target error
            # Require two consecutive batches to count as converged
            if mean_sq_err < .01 * 1.05: # At most 5% of the target MSE
                if twice:
                    print "Converged after {} iterations".format(n*batch_size)
                    raise KeyboardInterrupt
                else:
                    twice = True
            else:
                twice = False
    except KeyboardInterrupt:
        pass

    import matplotlib.pyplot as plt
    nX = 200
    nY = 200
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




