import numpy as np
from nnswarm import nnswarm
from tilecoder import Tilecoder
from tileswarm import TileSwarm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adadelta

# swarm dimensions and value limits
dims = [8, 8]
lims = [(-2, 2)] * 2
#lims = [(0, 2.0 * np.pi)] * 2

# create swarm
g = nnswarm(dims, lims)

# create tilecoder
T = Tilecoder(lims, dims, 10, 1)

# create tile swarm
S = TileSwarm(lims, dims, 100, 1)

# create nn
nn = Sequential()
nn.add(Dense(100, input_dim=2))
nn.add(Activation('relu'))
nn.add(Dense(100))
nn.add(Activation('tanh'))
nn.add(Dense(1))
nn.compile(optimizer='adadelta', loss='mse')

# target function with gaussian noise
def target_ftn(x, y, noise=True):
  # f = np.sin(x) + np.cos(y)
  f = x + y
  return f + noise * np.random.randn() * 0.1

# randomly sample target function until convergence
batch_size = 1
try:
  for iters in range(200):
    mse = 0.0
    for b in range(batch_size):
      xi = lims[0][0] + np.random.random() * (lims[0][1] - lims[0][0])
      yi = lims[1][0] + np.random.random() * (lims[1][1] - lims[1][0])
      zi = target_ftn(xi, yi)
      g[xi, yi] = zi
      S[xi, yi] = zi
      T[xi, yi] += 0.1 * (zi - T[xi, yi])
      nn.fit(np.array([[xi, yi]]), np.array([[zi]]), batch_size=1, nb_epoch=1, verbose=0)
      mse += (g[xi, yi] - zi) ** 2
    mse /= batch_size
    print 'samples:', (iters + 1) * batch_size#, 'batch_mse:', mse
except KeyboardInterrupt:
    print 'Stop'

# get learned function
print 'mapping function...'
res = 100.0
x = np.arange(lims[0][0], lims[0][1], (lims[0][1] - lims[0][0]) / res)
y = np.arange(lims[1][0], lims[1][1], (lims[1][1] - lims[1][0]) / res)
z = np.zeros([len(y), len(x)])
zg = np.zeros([len(y), len(x)])
zt = np.zeros([len(y), len(x)])
zs = np.zeros([len(y), len(x)])
znn = np.zeros([len(y), len(x)])
for i in range(len(x)):
  for j in range(len(y)):
    z[j, i] = target_ftn(x[i], y[j], noise=False)
    zg[j, i] = g[x[i], y[j]]
    zt[j, i] = T[x[i], y[j]]
    zs[j, i] = S[x[i], y[j]]
    znn[j, i] = nn.predict(np.array([[x[i], y[j]]]))

# plot
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
plt.title('target ftn')

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, zg, cmap=plt.get_cmap('hot'))
plt.title('nnswarm')

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, zs, cmap=plt.get_cmap('hot'))
plt.title('tileswarm')

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, zt, cmap=plt.get_cmap('hot'))
plt.title('tilecoder')

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, znn, cmap=plt.get_cmap('hot'))
plt.title('nn')

plt.show()
