from scipy.spatial.distance import cdist

class KNearestNeighbour(object):
    features = []
    prediction_labels = []
    k = 1
    train_set_len = 0

    def __init__(self, data_set, prediction_labels, k = 1):
        self.features = data_set
        self.prediction_labels = prediction_labels

        self.train_set_len = len(self.features)

        self.k = self.train_set_len if k > self.train_set_len else k

    def predict(self, data_point):
        distances = {}

        for n in range(0, self.train_set_len):
            distance_from_test_point = self.get_distance(data_point, self.features[n])
            label_for_test_point = self.prediction_labels[n]
            distances[n] = (label_for_test_point, distance_from_test_point)

        sorted_data_points = sorted(distances.items(), key = lambda x: x[1][1])

        prediction = 0
        print sorted_data_points

        for k in range(0, self.k):
            prediction += sorted_data_points[k][1][0]

        return self.sign(prediction)

    def get_distance(self, data_point_1, data_point_2):
        return cdist([data_point_1], [data_point_2])[0][0]

    def sign(self, prediction):
        return 1 if (prediction >= 0) else -1

    def print_set_len(self):
        dslen = len(self.features)
        print dslen
        print self.k

