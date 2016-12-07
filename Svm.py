import cvxopt
import numpy as np
from numpy import linalg


class Svm(object):
    features = []
    labels = []
    n_samples= 0
    b = 0
    w = 0
    n_features = 0
    sv_y = []
    C= None
    sv = []

    def __init__(self, features, labels, C=None):
        self.features = list(features)
        self.labels = labels
        self.n_samples = len(self.features)
        self.n_features = len(self.features[0])
        if self.C is not None: self.C = float(self.C)
        # Remove:
        self.p = 10
        self.sigma = 1

    def kernel(self, x, y, k = 2):
        if k == 0:
            return (1 + np.dot(x, y)) ** self.p

        elif k == 1:
            return np.exp(-linalg.norm(x - y) ** 2 / (2 * (self.sigma ** 2)))

        else: # Linear kernel
            return np.dot(x,y)

    def train(self):
        X = np.array(self.features)
        y = np.array(self.labels)

        # Gram matrix
        XiXj = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                XiXj[i, j] = self.kernel(X[i],  X[j])

        YiYj = np.outer(y, y)
        XiXjYiYj = XiXj * YiYj

        P = cvxopt.matrix(XiXjYiYj)
        q = cvxopt.matrix(np.ones(self.n_samples) * -1)
        A = cvxopt.matrix(y, (1, self.n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(self.n_samples) * -1))
            h = cvxopt.matrix(np.zeros(self.n_samples))
        else:
            tmp1 = np.diag(np.ones(self.n_samples) * -1)
            tmp2 = np.identity(self.n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(self.n_samples)
            tmp2 = np.ones(self.n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Solve the quadratic optimization
        alpha_values = cvxopt.solvers.qp(P, q, G, h, A, b)

        # get the alpha values
        # a = np.ravel(alpha_values['x'])
        a = np.ravel(alpha_values['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), self.n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * XiXj[ind[n], sv])
        self.b /= len(self.a)

        self.w = np.zeros(self.n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]

    def predict(self, X):
        value = np.dot(X, self.w) + self.b
        return 1 if value > 0  else -1