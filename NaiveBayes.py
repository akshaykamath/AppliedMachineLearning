import copy

class NaiveBayes():
    train_data = []
    train_labels = None
    test_data = []
    test_labels = None
    feature_len = None
    cond_prob = {}
    headers = []

    def __init__(self, train_data, train_labels, test_data, test_labels, headers):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.feature_len = len(train_data[0])
        self.headers = headers
        self.perform_naive_bayes()

    def perform_naive_bayes(self):
        self.generate_prob_table()
        self.count()

    def generate_prob_table(self):
        # generate class labels
        labels = set(self.train_labels)

        # prob for class labels
        for l in labels:
          self.cond_prob[str(l)] = 0

        # for features and label combo..
        for l in labels:
            for f in self.headers:
                key = str(f) + "AND" + str(l)
                self.cond_prob[key] = 0

    def count(self):
        count_d = copy.deepcopy(self.cond_prob)

        # add n1 add n-1 as well
        labels = set(self.train_labels)
        for l in labels:
            key = "n" + str(l)
            count_d[key] = 0

        for idx, data_point in enumerate(self.train_data):
            # add the label
            label = str(self.train_labels[idx])
            count_d[label] = count_d.get(label) + 1

            # count each feature
            for idx, f in enumerate(data_point):
                if f > 0:
                    featureName = self.headers[idx]
                    key = featureName + "AND" + str(label)
                    count_d[key] = count_d.get(key) + 1

                    k = "n" + str(label)
                    count_d[k] = count_d.get(k) + 1

        # fill up the prob probability table
        for key in self.cond_prob.keys():
            if "AND" in key:
                f, l = key.split("AND")
                num = count_d.get(key) + 1
                den = count_d.get("n" + l) + self.feature_len

                self.cond_prob[key] = float(num) / float(den)

        # add p("1") and p("-1")
        for l in labels:
            k = str(l)
            self.cond_prob[k] = float(count_d.get(k))/ len(self.train_data)
        # print(count_d)
        # print(self.cond_prob)

    def predict(self, data_point):
        # compute case 1
        labels = list(set(self.train_labels))

        case1 = 1
        for idx, word in enumerate(data_point):
            if word > 0:
                key = self.headers[idx] + "AND" + str(labels[0])
                case1 = case1 * self.cond_prob.get(key)
        case1 = case1 * self.cond_prob.get(str(labels[0]))

        # compute case 2
        case2 = 1
        for idx, word in enumerate(data_point):
            if word > 0:
                key = self.headers[idx] + "AND" + str(labels[1])
                case2 = case2 * self.cond_prob.get(key)
        case2 = case2 * self.cond_prob.get(str(labels[1]))

        if case1 > case2:
            return labels[0]
        else:
            return labels[1]


if __name__ == "__main__":
    train_data = [(1,0,1),
                  (0,0,1),
                  (1,1,0),
                  (0,0,1),
                  (1,1,0)]

    train_label = [1,1,0,0,0]

    nb = NaiveBayes(train_data, train_label, [], [], ["HELLO", "YOU", "SEND"])
    print(nb.predict((1,1,0)))








