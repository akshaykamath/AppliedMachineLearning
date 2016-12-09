from __future__ import division
import numpy as np
from copy import deepcopy
from sklearn import metrics
import matplotlib.pyplot as plt


class EvaluationMetrics(object):
    test_data_points = []
    test_true_labels = []
    target_attribute_value_count = {}
    classifier = None

    def __init__(self, classifier, test_features, test_true_labels):
        self.test_data_points = test_features
        self.test_true_labels = test_true_labels
        self.classifier = classifier
        self.positive_result = 1
        self.negative_result = -1
        self.count_uniqueness()

    def count_uniqueness(self):
        total_count = len(self.test_true_labels)
        self.target_attribute_value_count[self.positive_result] = self.test_true_labels.count(self.positive_result)
        self.target_attribute_value_count[self.negative_result] = self.test_true_labels.count(self.negative_result)

    def evaluate(self):

        # Get the total number of data points in the test set.
        total_num = len(self.test_data_points)

        correct_classification = 0
        true_positive = 0
        true_negative = 0
        total_positive = self.target_attribute_value_count[self.positive_result]
        total_negative = self.target_attribute_value_count[self.negative_result]
        count = 0
        predicted = []
        for i in range(0, len(self.test_data_points)):
            print "Datapoint prediction:", i
            test_data_point = self.test_data_points[i]

            count += 1

            classification_result = self.classifier.predict(test_data_point)
            predicted.append(classification_result)

            test_set_result = self.test_true_labels[i]

            if classification_result == test_set_result:
                correct_classification += 1

                if test_set_result == self.positive_result:
                    true_positive += 1

                if test_set_result == self.negative_result:
                    true_negative += 1

            false_negative = total_positive - true_positive
            false_positive = total_negative - true_negative

            accuracy = (correct_classification / total_num)
            error = 1 - accuracy
            precision = true_positive / (true_positive  + false_positive)
            recall = true_positive / (true_positive + false_negative)

        result = {"accuracy": accuracy, "error": error, "TP": true_positive, "FN":false_negative,
                  "TN":true_negative, "FP": false_positive, "precision": precision, "recall": recall, "predicted": predicted}

        print result

        return result


    def compute_au_roc(self, prediction, y):
        print len(y)
        print len(prediction)

        y = np.array(y)

        n_classes = 2
        pred = np.array(prediction)
        auc = metrics.roc_auc_score(y, pred)
        # auc = metrics.auc(fpr, tpr)
        print "auc", auc


    def compute_and_plot_auc(self, prediction, y):
        y = np.array(y)
        pred = np.array(prediction)

        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y, pred)
        roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()