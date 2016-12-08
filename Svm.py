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

    def __init__(self, features, labels, C=1, power=2, sigma = 1):
        self.features = list(features)
        self.labels = labels
        self.n_samples = len(self.features)
        self.n_features = len(self.features[0])
        self.C= C

        self.p = power
        self.sigma = 1

    def kernel(self, x, y, k = 0):
        if k == 0:
            return np.dot(x, y)

        elif k == 1:
            return (1 + np.dot(x, y)) ** self.p

        else: # Linear kernel
            return np.exp(-linalg.norm(x - y) ** 2 / (2 * (self.sigma ** 2)))

    def train(self):
        X = np.array(self.features)
        y = np.array(self.labels)

        n_samples, n_features = X.shape

        # Compute the double summation.
        # Store the kernel computations
        p_matrix = np.zeros((n_samples, n_samples))
        XiXj = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                xixj = self.kernel(X[i], X[j])
                XiXj[i, j] = xixj
                yiyj = y[i] * y[j]
                p_matrix[i, j] = xixj * yiyj

        # Compute the constraint that Summation(aiyi) = 0
        a_matrix = []
        for i in range(0, n_samples):
            a_matrix.append([y[i]])

        # Compute the q term i.e. summation(alphai)
        q_matrix = [-1.0] * n_samples

        # Create LHS matrix for constraint ai >= 0 and ai <=C
        # Constraint 1: ai >= 0 do this for all n_samples
        constraints = []
        for i in range(0, n_samples):
            constraint = [0.0] * n_samples
            constraint[i] = -1.0
            constraints.append(np.array(constraint))

        if self.C != 0:
            # Constraint 2: ai <= C
            constraints2 = []
            for i in range(0, n_samples):
                constraint = [0.0] * n_samples
                constraint[i] = 1.0
                constraints2.append(np.array(constraint))

            constraints.extend(constraints2)

        constraint_lhs = np.array(constraints)

        constraint_rhs = [0.0] * n_samples
        if self.C != 0:
            constraint_c = [1.0 * self.C] * n_samples
            constraint_rhs.extend(constraint_c)

        constraint_rhs = np.array(constraint_rhs)

        P = cvxopt.matrix(p_matrix)
        q = cvxopt.matrix(q_matrix)
        A = cvxopt.matrix(a_matrix)
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(constraint_lhs)
        h = cvxopt.matrix(constraint_rhs)

        # Solve It!
        alpha_values = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Get the alpha values
        support_vecs = []
        i = 0
        for a in alpha_values['x']:
            value = a if a > 1e-5 else 0
            tup = (value, y[i], X[i], XiXj[i], i)
            if value > 0:
                support_vecs.append(tup)

            i+=1

        self.b = 0
        alphas = np.array([vec[0] for vec in support_vecs])
        svy = np.array([vec[1] for vec in support_vecs])
        sv_indices = np.array([vec[4] for vec in support_vecs])

        for i in range(0, len(support_vecs)):
            x = support_vecs[i][3]
            x = [x[svi] for svi in sv_indices]

            self.b += (y[i] - np.sum(alphas * svy * x))

        self.b /= len(support_vecs)

        self.w = [0.0] * n_features
        for sv in support_vecs:
            w = sv[0] * sv[1] * sv[2]
            self.w += w

    def predict(self, X):
        value = np.dot(X, self.w) + self.b
        return 1 if value > 0  else -1