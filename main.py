from KNearestNeighbour import KNearestNeighbour
from DataHandler import DataHandler


def test_stub():
    data_set = [(3, 2), (1,1), (4,4), (7,9), (11, 11)]
    prediction_labels = [1,-1,1,-1,1]
    knn = KNearestNeighbour(data_set, prediction_labels, 1)

    print knn.predict((0,0))


def data_numeric_stub():
    dh = DataHandler('data/total-test.csv', 'prediction_label')
    headers, features, prediction_labels = dh.get_numeric_data_set()
    knn = KNearestNeighbour(features, prediction_labels, 1)
    print knn.predict((0, 0))


def data_numeric__cols_stub():
    dh = DataHandler('data/total-test.csv', 'prediction_label', ['label2'])
    headers, features, prediction_labels = dh.get_numeric_data_set()
    knn = KNearestNeighbour(features, prediction_labels, 1)
    print knn.predict((0, ))


def data_text_stub():
    dh = DataHandler('data/train-set.csv', 'sentiment')
    headers, features, prediction_labels = dh.get_textual_data_set()
    review_text_index = headers.index('review_text')
    review_text_list = [feature[1] for feature in features]
    train_features = dh.convert_docs_to_bow(review_text_list)


#data_numeric__cols_stub()
data_text_stub()