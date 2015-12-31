from matplotlib import pyplot as pt
import numpy as np
from scipy.stats import triang
from scipy.stats import linregress 

import sampen

def q(x, M, N, S):
  ''' triangle distribution in the point x with numeric parameters M, N, S '''
  a = N * (N + M) / 2.
  b = M * (N + M) / 2.
  ans = np.copy(x)
  ans[x <= S - N] = 0
  ans[x >= S + M] = 0
  ans[(x > S) * (x <= S + M)] = (S + M - x[(x > S) * (x <= S + M)]) / b
  ans[(x <= S) * (x > S - N)] = (x[(x <= S) * (x > S - N)] - S + N) / a
  return ans

def F(x, h, p):
  ''' Goal function to minimize '''
  f = np.sum((q(x, p[0], p[1], p[2]) - h)**2)
  return f

def gradF(x, h, p):
  ''' Gradient of the goal function F '''
  d = 1.e-4
  f = F(x, h, p)
  g = np.zeros(3)
  for i in xrange(3):
    if i == 0:
      p1 = (p[0] + d, p[1], p[2])
    elif i == 1:
      p1 = (p[0], p[1] + d, p[2])
    elif i == 2:
      p1 = (p[0], p[1], p[2] + d)
    g[i] = (F(x, h, p1) - F(x, h, p)) / d
  return g

def grad_descent(iter_number, step, alpha, p0, x, h):
  ''' gradient descent of minimizing F with some parameters'''
  P = (p0[0], p0[1], p0[2])
  for i in xrange(iter_number):
    g = gradF(x, h, P)

    P0 = P[0] - step * g[0]
    P1 = P[1] - step * g[1]
    P2 = P[2] - step * g[2]
    P = (P0, P1, P2)

    if np.linalg.norm(g)*step < 1.e-3:
      break;

    if i%10 == 0:
      step /= alpha
  return P0, P1, P2

def hist(x, bins):

  step = (np.max(x) - np.min(x)) / bins
  idx = ((x - np.min(x)) / step).astype(np.int32)
  idx[idx == bins] -= 1

  histogram = np.histogram(x, bins=bins)[0] / float(x.shape[0]) / step

  return histogram[idx]


def apply_grad_descent(x):
  ''' Main function to minimize F '''
  std = np.std(x)
  mean = np.mean(x)
  x_n = (x - mean) / std

  bins = int(round(x.shape[0]**0.4))
  H_n = hist(x_n, bins)
  H = hist(x, bins)

  M = np.std(x_n) * 2.5
  N = np.std(x_n) * 2.5
  S = 0
  p0 = (M, N, S)

  n_iter = 100
  step = 0.004
  alpha = 1.2

  M1, N1, S1 = grad_descent(n_iter, step, alpha, p0, x_n, H_n)

  t_n = np.array([q(xi, M1, N1, S1) for xi in x_n])

  M2 = M1 * std
  N2 = N1 * std
  S2 = S1 * std + mean
  t = np.array([q(xi, M2, N2, S2) for xi in x])

  return M2, N2, S2



if __name__ == '__main__':
  pass