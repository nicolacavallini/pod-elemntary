from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

from local_gaussian_process import DataBase
from local_gaussian_process import GaussianProcess



""" This is code for simple GP regression. It assumes a zero mean GP Prior """


# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()


N = 10         # number of training points.
n = 50         # number of test points.

# Sample some input points and noisy versions of the function evaluated at
# these points.
#X = np.random.uniform(-5, 5, size=(N,1))
X = np.linspace(-5, 5, N).reshape(-1,1)
print(X)
y = f(X)

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

db = DataBase(X,y)

gp = GaussianProcess("squared-exp",1.)
gp.fit(db)
mu, s = gp.predict(Xtest)



# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-5, 5, -3, 3])
pl.show()

# draw samples from the prior at our test points.
#L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
#f_prior = np.dot(L, np.random.normal(size=(n,10)))
#pl.figure(2)
#pl.clf()
#pl.plot(Xtest, f_prior)
#pl.title('Ten samples from the GP prior')
#pl.axis([-5, 5, -3, 3])
#pl.savefig('prior.png', bbox_inches='tight')
#
## draw samples from the posterior at our test points.
#L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
#f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
#pl.figure(3)
#pl.clf()
#pl.plot(Xtest, f_post)
#pl.title('Ten samples from the GP posterior')
#pl.axis([-5, 5, -3, 3])
#pl.savefig('post.png', bbox_inches='tight')
#
#pl.show()
#
