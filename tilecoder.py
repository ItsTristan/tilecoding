#!/usr/bin/env python
from __future__ import division
import numpy as np


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

		self._weights = init([num_tilings] + [d+1 for d in input_densities])

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
	def targetFunction(in1,in2):
		return sin(in1-3.0)*cos(in2) + normal(0,0.1)

	bounds = [(-1.,2.), (-1.,2.)]
	width = 5
	height = 5
	num_tilings = 10

	T = Tilecoder(bounds, (width, height), num_tilings, 1)

	for n in xrange(10):
		mean_sq_err = 0.0
		for i in xrange(1000):
			X = rescale(random(), 0, 1, *bounds[0])
			Y = rescale(random(), 0, 1, *bounds[1])
			Z = targetFunction(X,Y)
			err = (T[X,Y] - Z) ** 2
			T[X,Y] += (Z - T[X,Y])*0.05
			mean_sq_err += (err - mean_sq_err) / (i+1)
		print "MSE = {}".format(mean_sq_err)

	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

	nX = 100
	nY = 100
	Z = np.zeros((nX, nY))
	bwidth = [(x2-x1) for (x1,x2) in bounds]
	for i,x in enumerate(bwidth[0]*i/nX+bounds[0][0] for i in xrange(nX)):
		for j,y in enumerate(bwidth[1]*i/nY+bounds[1][0] for i in xrange(nY)):
			Z[i,j] = T[x,y]
	plt.contour(Z)
	plt.show()

if __name__ == '__main__':
	import doctest
	doctest.testmod()

	example()