from __future__ import division

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
        """
        :return:
        """

        # Get the total number of data points in the test set.
        total_num = len(self.test_data_points)

        # Get the distribution of + and - data points in the test data set.
        #test_attribute_value_count = self.count_uniqueness(self.test_data_points)

        # Assume that the first taget value is positive and the other one is negative.
        # This is because the possible values of target values are calculated by the program.
        #print(self.attribute_possible_values.get(self.target_attribute))
        #positive_result = self.target_attribute_values[0]
        #negative_result = self.target_attribute_values[1]

        correct_classification = 0
        true_positive = 0
        true_negative = 0
        total_positive = self.target_attribute_value_count[self.positive_result]
        total_negative = self.target_attribute_value_count[self.negative_result]
        count = 0

        for i in range(0, len(self.test_data_points)):
            print "Datapoint prediction:", i
            test_data_point = self.test_data_points[i]

            count += 1

            classification_result = self.classifier.predict(test_data_point)
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

        """print("Accuracy is {}".format(accuracy))
        print("Error is {}".format(error))

        print "#################################################"

        print "True Positive", positive_result, true_positive
        print "False Negative", negative_result, false_negative
        print "True Negative", negative_result, true_negative
        print "False Positive", positive_result, false_positive"""

        result = {"accuracy": accuracy, "error": error, "TP": true_positive, "FN":false_negative,
                  "TN":true_negative, "FP": false_positive}

        print result
        # table = PrettyTable(["Variable", "Value"])
        # for k, v in result.items():
        #     table.add_row([k, v])
        # print(table)

        return result
