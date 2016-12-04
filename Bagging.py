import numpy as np
from KNearestNeighbour import KNearestNeighbour


class Bagging(object):
    train_features = []
    S = []
    S_PREDICTIONS = []
    num_bags = 0
    k = 1
    prediction_labels = []
    KNN= []

    def __init__(self, data_set, prediction_labels, num_bags, k):
        self.train_features = data_set
        self.num_bags = num_bags
        self.k = k
        self.prediction_labels = prediction_labels
        self.bootstrap_sampling()
        self.learn_bagged()

    def bootstrap_sampling(self):
        """
        Randomly picks an element from the list and creates a new bag
        :param data:
        """
        for i in range(0, self.num_bags):
            bag = []
            bag_preds = []

            for r in range(0, len(self.train_features)):
                idx = np.random.randint(0, len(self.train_features))
                bag.append(self.train_features[idx])
                bag_preds.append(self.prediction_labels[idx])

            self.S.append(bag)
            self.S_PREDICTIONS.append(bag_preds)

    def learn_bagged(self):

        for i in range(0, len(self.S)):
            bag = self.S[i]
            bag_predictions = self.S_PREDICTIONS[i]

            knn = KNearestNeighbour(bag, bag_predictions, self.k)
            self.KNN.append(knn)

    def predict(self, data_point):
        # determine the label by majority vote
        result = 0

        for knn in self.KNN:
            classification_result = knn.predict(data_point)
            result+= classification_result

        prediction = self.sign(result)
        return prediction

    def sign(self, prediction):
        return 1 if (prediction >= 0) else -1
