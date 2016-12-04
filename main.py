from KNearestNeighbour import KNearestNeighbour
from DataHandler import DataHandler
from EvaluationMetrics import EvaluationMetrics
from Bagging import Bagging

# Method to test basic KNN implementation with a dummy feature set and predictions set
def test_stub():
    data_set = [(3, 2), (1,1), (4,4), (7,9), (11, 11)]
    prediction_labels = [1,-1,1,-1,1]
    knn = KNearestNeighbour(data_set, prediction_labels, 1)

    print knn.predict((0,0))


# Method to test reading of numerical columns
def data_numeric_stub():
    dh = DataHandler('data/total-test.csv', 'prediction_label')
    headers, features, prediction_labels = dh.get_numeric_data_set()
    knn = KNearestNeighbour(features, prediction_labels, 1)
    print knn.predict((0, 0))


# Method to test reading of numerical columns with some columns that need to be ignored
def data_numeric__cols_stub():
    dh = DataHandler('data/total-test.csv', 'prediction_label')
    headers, features, prediction_labels = dh.get_numeric_data_set()
    print(len(headers))
    print (len(features[0]))

    print headers
    print features[0]
    knn = KNearestNeighbour(features, prediction_labels, 1)
    print knn.predict((0, ))


# Method to test tf-idf
def data_text_stub():
    dh = DataHandler('data/train-set.csv', 'sentiment')
    headers, features, prediction_labels = dh.get_textual_data_set()
    review_text_index = headers.index('review_text')
    review_text_list = [feature[review_text_index] for feature in features]
    bow_headers, train_features = dh.convert_docs_to_bow(review_text_list)


# Method to write tf-idf features and prediction labels into a file for subsequent reading.
def data_write_train_stub():
    dh = DataHandler('data/train-set.csv', 'sentiment')

    headers, features, prediction_labels = dh.get_textual_data_set()
    review_text_index = headers.index('review_text')

    review_text_list = [feature[review_text_index] for feature in features]
    bow_feature_names = dh.get_feature_set_for_documents(review_text_list)
    print bow_feature_names
    bow_features = dh.convert_docs_to_bow_for_features(review_text_list, bow_feature_names)
    train_prediction_labels = dh.convert_sentiment_list_to_number(prediction_labels)

    print len(bow_feature_names)
    print len(bow_features[0])
    print train_prediction_labels[0]

    bow_feature_names.append("prediction_label")
    dh.write_to_file('data/train-set-feature-engineered.csv', bow_features, bow_feature_names, train_prediction_labels)


# Method to write tf-idf features and prediction labels into a file for subsequent reading.
def data_write_test_stub():
    dh = DataHandler('data/train-set.csv', 'sentiment')

    headers, features, prediction_labels = dh.get_textual_data_set()
    review_text_index = headers.index('review_text')

    review_text_list = [feature[review_text_index] for feature in features]
    bow_feature_names = dh.get_feature_set_for_documents(review_text_list)

    dh = DataHandler('data/test-set.csv', 'sentiment')

    headers, features, prediction_labels = dh.get_textual_data_set()
    review_text_index = headers.index('review_text')

    review_text_list = [feature[review_text_index] for feature in features]

    bow_features = dh.convert_docs_to_bow_for_features(review_text_list, bow_feature_names)
    test_prediction_labels = dh.convert_sentiment_list_to_number(prediction_labels)

    print len(bow_feature_names)
    print len(bow_features[0])
    print test_prediction_labels[0]

    bow_feature_names.append("prediction_label")
    dh.write_to_file('data/test-set-feature-engineered.csv', bow_features, bow_feature_names, test_prediction_labels)


# Method to read tf-idf feature engineered files and perform KNN classification
def test_knn_on_review_data_set():
    dh = DataHandler('data/train-set-feature-engineered.csv', 'prediction_label')
    headers, train_features, train_prediction_labels = dh.get_numeric_data_set()

    knn = KNearestNeighbour(train_features, train_prediction_labels, 1)

    dh_test = DataHandler('data/test-set-feature-engineered.csv', 'prediction_label')
    headers, test_features, test_prediction_labels = dh_test.get_numeric_data_set()

    print knn.predict(test_features[1])
    print test_prediction_labels[1]


# This is the main method to evaluate knn on the data set.
def evaluate_knn():
    dh = DataHandler('data/train-set-feature-engineered.csv', 'prediction_label')
    headers, train_features, train_prediction_labels = dh.get_numeric_data_set()

    knn = KNearestNeighbour(train_features, train_prediction_labels, 5)

    dh_test = DataHandler('data/test-set-feature-engineered.csv', 'prediction_label')
    headers, test_features, test_prediction_labels = dh_test.get_numeric_data_set()

    eval_metrics = EvaluationMetrics(knn, test_features, test_prediction_labels)
    eval_metrics.evaluate()


# This is the main method to evaluate knn on the data set.
def evaluate_bagged_knn():
    dh = DataHandler('data/train-set-feature-engineered.csv', 'prediction_label')
    headers, train_features, train_prediction_labels = dh.get_numeric_data_set()

    bagged_knn = Bagging(train_features, train_prediction_labels, 5, 1)

    dh_test = DataHandler('data/test-set-feature-engineered.csv', 'prediction_label')
    headers, test_features, test_prediction_labels = dh_test.get_numeric_data_set()

    eval_metrics = EvaluationMetrics(bagged_knn, test_features, test_prediction_labels)
    eval_metrics.evaluate()

#test_knn_on_review_data_set()
#data_text_stub()

#data_write_train_stub()
#data_write_test_stub()
#test_knn_on_review_data_set()
#evaluate_knn()

evaluate_bagged_knn()