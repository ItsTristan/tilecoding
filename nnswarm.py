#!/usr/bin/env python
import numpy as np

class nnswarm:
  def __init__(self, dims, limits, learning_rate=0.1, momentum=0.707):
    self._in_size = len(dims)
    self._hash_vec = np.array([1])
    for i in range(len(dims) - 1):
      self._hash_vec = np.hstack([self._hash_vec, dims[i] * self._hash_vec[-1]])
    self._dims = np.array(dims)
    self._limits = np.array(limits)
    self._ranges = self._limits[:, 1] - self._limits[:, 0]
    self._grid = np.array([[0.0] * (self._in_size + 1)] * np.prod(dims))
    self._dw = np.array([[0.0] * (self._in_size + 1)] * np.prod(dims))
    self._rho = momentum
    self._alpha = learning_rate

  def _coords(self, x):
    return 0.999 * ((x - self._limits[:, 0]) / self._ranges) * self._dims

  def _input(self, x):
    return np.hstack([self._coords(x) % 1, 1])

  def _hash_id(self, x):
    return int(np.dot(np.floor(self._coords(x)), self._hash_vec))

  def __getitem__(self, x):
    return np.dot(self._input(x), self._grid[self._hash_id(x)])

  def __setitem__(self, x, val):
    h_id = self._hash_id(x)
    self._dw[h_id] = self._rho * self._dw[h_id] - self._alpha * self._input(x) * (self[x] - val)
    self._grid[h_id] += self._dw[h_id]

def example():
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  # swarm dimensions and value limits
  dims = [8, 8]
  lims = [(0, 2.0 * np.pi)] * 2

  # create swarm
  g = nnswarm(dims, lims)

  # target function with gaussian noise
  def target_ftn(x, y):
    return np.sin(x) + np.cos(y) + np.random.randn() * 0.1

  # randomly sample target function until convergence
  batch_size = 50
  for iters in range(200):
    mse = 0.0
    for b in range(batch_size):
      xi = lims[0][0] + np.random.random() * (lims[0][1] - lims[0][0])
      yi = lims[1][0] + np.random.random() * (lims[1][1] - lims[1][0])
      zi = target_ftn(xi, yi)
      g[xi, yi] = zi
      mse += (g[xi, yi] - zi) ** 2
    mse /= batch_size
    print 'samples:', (iters + 1) * batch_size, 'batch_mse:', mse

  # get learned function
  print 'mapping function...'
  x = np.arange(lims[0][0], lims[0][1], 0.025)
  y = np.arange(lims[1][0], lims[1][1], 0.025)
  z = np.zeros([len(y), len(x)])
  for i in range(len(x)):
    for j in range(len(y)):
      z[j, i] = g[x[i], y[j]]

  # plot
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  X, Y = np.meshgrid(x, y)
  surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
  plt.show()

if __name__ == '__main__':
  example()
