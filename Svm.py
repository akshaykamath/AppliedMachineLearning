import cvxopt
import numpy as np
from numpy import linalg


class Svm(object):
    features = []
    labels = []
    number_of_samples= 0
    b = 0
    w = 0
    number_of_features = 0
    C= None

    kernel_method = 0

    def __init__(self, features, labels, C=0, kernel = 0, power=2, sigma = 1):
        self.features = np.array(list(features))
        self.labels = labels
        self.number_of_samples = len(self.features)
        self.number_of_features = len(self.features[0])
        self.C= C
        self.kernel_method = kernel
        self.p = power
        self.sigma = sigma

    def kernel(self, x, y):
        # linear
        if self.kernel_method == 0:
            return np.dot(x, y)

        # polynomial
        elif self.kernel_method  == 1:
            return pow(np.dot(x, y) + 1, self.p)

        else: # gaussian
            norm = pow(linalg.norm(x - y), 2)
            function = - norm / self.sigma

            return np.exp(function)

    def train(self):
        # Compute the double summation.
        # Store the kernel computations
        p_matrix = np.zeros((self.number_of_samples, self.number_of_samples))
        XiXj = np.zeros((self.number_of_samples, self.number_of_samples))
        for i in range(self.number_of_samples):
            for j in range(self.number_of_samples):
                xixj = self.kernel(self.features[i], self.features[j])
                XiXj[i, j] = xixj
                yiyj = self.labels[i] * self.labels[j]
                p_matrix[i, j] = xixj * yiyj

        # Compute the constraint that Summation(aiyi) = 0
        a_matrix = []
        for i in range(0, self.number_of_samples):
            a_matrix.append([self.labels[i]])

        # Compute the q term i.e. summation(alphai)
        q_matrix = [-1.0] * self.number_of_samples

        # Create LHS matrix for constraint ai >= 0 and ai <=C
        # Constraint 1: ai >= 0 do this for all n_samples
        constraints = []
        for i in range(0, self.number_of_samples):
            constraint = [0.0] * self.number_of_samples
            constraint[i] = -1.0
            constraints.append(np.array(constraint))

        if self.C != 0:
            # Constraint 2: ai <= C
            constraints2 = []
            for i in range(0, self.number_of_samples):
                constraint = [0.0] * self.number_of_samples
                constraint[i] = 1.0
                constraints2.append(np.array(constraint))

            constraints.extend(constraints2)

        constraint_lhs = np.array(constraints)

        constraint_rhs = [0.0] * self.number_of_samples
        if self.C != 0:
            constraint_c = [1.0 * self.C] * self.number_of_samples
            constraint_rhs.extend(constraint_c)

        constraint_rhs = np.array(constraint_rhs)

        p = cvxopt.matrix(p_matrix)
        q = cvxopt.matrix(q_matrix)
        a = cvxopt.matrix(a_matrix)
        b = cvxopt.matrix(0.0)
        g = cvxopt.matrix(constraint_lhs)
        h = cvxopt.matrix(constraint_rhs)

        # Solve It!
        alpha_values = cvxopt.solvers.qp(p, q, g, h, a, b)

        # Get the alpha values
        support_vecs = []
        i = 0
        for a in alpha_values['x']:
            value = a if a > 1e-5 else 0
            tup = (value, self.labels[i], self.features[i], XiXj[i], i)
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

            self.b += (self.labels[i] - np.sum(alphas * svy * x))

        self.b /= len(support_vecs)

        self.w = [0.0] * self.number_of_features
        for sv in support_vecs:
            w = sv[0] * sv[1] * sv[2]
            self.w += w

    def predict(self, X):
        value = np.dot(X, self.w) + self.b
        return 1 if value > 0  else -1