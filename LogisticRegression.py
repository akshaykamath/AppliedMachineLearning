# coding=utf-8
from __future__ import division
import math
import numpy

class LogisticRegression():
    train_data = []
    train_labels = None
    test_data = []
    test_labels = None
    alpha = 0.3
    feature_len = None
    weights = []
    weight_vector = []

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        # self.train_labels = train_labels
        self.train_labels = self.convert_labels(train_labels)
        self.test_data = test_data
        # self.test_labels = test_labels
        self.test_labels = self.convert_labels(test_labels)
        self.feature_len = len(train_data[0])
        self.perform_logistic_regression()
        print(self.weight_vector)
        # self.printAccuracy()

    # assuming our data would have 1.0 for pos and -1.0 for neg
    def convert_labels(self, labels):
        d = {1.0: 1, -1.0: 0}
        res = []
        for x in labels:
            res.append(d.get(x))
        return res

    def sigmoid(self, z):
        return long(1) / long((1 + numpy.exp(-z)))

    def calculate_prediction(self, weight, data_point):
        """
        prediction = 1 / (1 + e^(-(w0 + w1*data_point1 + w2*data_point2+ .....)))
        :param weight: weight vector
        :param data_point: data point
        :return: the computed prediction
        """
        res = weight[0]
        for idx, x in enumerate(data_point):
            res += float(weight[idx + 1]) * float(x)
        print("without sigmoid {}".format(res))
        yhat = self.sigmoid(res)
        return yhat

    def calculate_coefficient(self, prediction, weight, data_point, y):
        """
        w = w + alpha * (y – prediction) * prediction * (1 – prediction) * x
        :param prediction: the calculated prediction
        :param weight: the weight as a list vector
        :param data_point: the data point as a list vector
        :param y:
        :return:
        """
        updated_weight = []
        for w, x in zip(weight, data_point):
            uw = w + self.alpha * (y - prediction) * prediction * (1 - prediction) * x
            updated_weight.append(uw)
        return updated_weight

    def perform_logistic_regression(self):
        # initialize weight vector
        weight_vector = []
        for i in range(0, self.feature_len + 1):
            weight_vector.append(0)
        self.weights.append(weight_vector)

        epoch = 1000
        idx = 0
        while epoch > 0 and idx < len(self.train_data)-1:
            # print("epoch number {}".format(epoch))
            data_point = self.train_data[idx]

            # predict
            yhat = self.calculate_prediction(weight_vector, data_point)

            # calculate coefficients
            intercept = (1.0,)
            weight_vector = self.calculate_coefficient(yhat, weight_vector, intercept + data_point, self.train_labels[idx])
            # print(weight_vector)
            self.weights.append(weight_vector)
            idx += 1
            epoch -= 1

        self.weight_vector = self.weights[-1]

    def predict(self, data_point):
        d = {1: 1.0, 0: -1.0}
        res = self.calculate_prediction(self.weight_vector, data_point)
        if res <= 0.5:
            p_label = 0
        else:
            p_label = 1
        return d[p_label]

if __name__ == "__main__":
    train_data = [(2.7810836, 2.550537003),
                  (1.465489372, 2.362125076),
                  (3.396561688, 4.400293529),
                  (1.38807019, 1.850220317),
                  (3.06407232,	3.005305973),
                  (7.627531214, 2.759262235),
                  (5.332441248,	2.088626775),
                  (6.922596716, 1.77106367),
                  (8.675418651,	-0.2420686549),
                  (7.673756466,	3.508563011)]
    train_label = [0,0,0,0,0,1,1,1,1,1]

    test_data = [(2.7810836, 2.550537003),
                  (1.465489372, 2.362125076),
                  (3.396561688, 4.400293529),
                  (1.38807019, 1.850220317),
                  (3.06407232,	3.005305973),
                  (7.627531214, 2.759262235),
                  (5.332441248,	2.088626775),
                  (6.922596716, 1.77106367),
                  (8.675418651,	-0.2420686549),
                  (7.673756466,	3.508563011)]
    test_label = [0,0,0,0,0,1,1,1,1,1]

    lr = LogisticRegression(train_data, train_label, test_data, test_label)





