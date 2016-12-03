from KNearestNeighbour import KNearestNeighbour

def test_stub():
    data_set = [(3, 2), (1,1), (4,4), (7,9), (11, 11)]
    prediction_labels = [1,-1,1,-1,1]
    knn = KNearestNeighbour(data_set, prediction_labels, 1)

    print knn.predict((0,0))

test_stub()