from __future__ import division
from nltk.stem.porter import PorterStemmer
import csv
from copy import deepcopy
import math
import numpy as np
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class DataHandler(object):
    data_set = []
    prediction_labels = []
    data_set_file_name = ""
    prediction_label = ""
    cos_to_ignore = []
    #http://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html
    #stop_words = ['i', 'he', 'she', 'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for' ]

    def __init__(self, data_set_file_name, prediction_label, cols_to_ignore=[]):
        self.data_set_file_name = data_set_file_name
        self.prediction_label = prediction_label
        self.cos_to_ignore = cols_to_ignore

    def get_numeric_data_set(self):
        csv_file = open(self.data_set_file_name, 'r')

        reader = csv.reader(csv_file, delimiter=',')
        all_lines = []

        for line in reader:
            all_lines.append(line)

        first_line = all_lines[0]
        target_index = first_line.index(self.prediction_label)

        all_lines.remove(first_line)

        features = []
        predictions = []

        for line in all_lines:
            target_value = line[target_index]
            line.remove(line[target_index])
            for col in self.cos_to_ignore:
                col_index = first_line.index(col)
                line.remove(line[col_index])

            numeric_line = [float(word) for word in line]
            predictions.append(float(target_value))
            features.append(tuple(numeric_line))

        print "features and prediction array"
        #print features
        #print predictions
        print "###########"

        return first_line, features, predictions

    def get_textual_data_set(self):
        csv_file = open(self.data_set_file_name, 'r')

        reader = csv.reader(csv_file, delimiter=',')
        all_lines = []

        for line in reader:
            all_lines.append(line)

        first_line = all_lines[0]
        target_index = first_line.index(self.prediction_label)

        all_lines.remove(first_line)

        features = []
        predictions = []

        for line in all_lines:
            target_value = line[target_index]
            line.remove(target_value)

            for col in self.cos_to_ignore:
                col_index = first_line.index(col)
                line.remove(line[col_index])

            split_line = [word for word in line]
            predictions.append(target_value)
            features.append(split_line)

        print "features and prediction array"
        #print features
        #print predictions
        print "###########"

        return first_line, features, predictions

    def convert_numeric_text_to_numbers(self, lines):
        numeric_lines = [float(num) for num in lines]
        return numeric_lines

    def tokenize(self, document, clean_strategy = 0):
        if clean_strategy == 1:
            return document.lower().split()
        else:
            return document.lower().split()

    def get_feature_set_for_documents(self, documents):
        terms_in_documents = {}
        for document in documents:
            words = self.tokenize(document)
            for word in words:
                if word not in terms_in_documents:
                    terms_in_documents[word] = 0

        return terms_in_documents.keys()

    def get_cross_validation_data_sets(self, k, X, Y):

        cross_val_sets = {}

        NPX = np.array(X)
        NPY = np.array(Y)

        # split the data into k folds
        train_list_k_folds = np.array_split(NPX, k)

        # split the labels into k folds
        train_labels_k_folds = np.array_split(NPY, k)

        for i in range(0, k):
            # Make a deep copy first then extract the training set and tuning set for this fold
            k_fold_copy = deepcopy(train_list_k_folds)
            k_fold_label_copy = deepcopy(train_labels_k_folds)

            tuning_set_object = {}

            tuning_set = k_fold_copy[i].tolist()
            tuning_set_labels = k_fold_label_copy[i].tolist()

            # create the tuning set object for this fold
            tuning_set_object["data_points"] = tuning_set
            tuning_set_object["labels"] = tuning_set_labels

            del(k_fold_copy[i])
            del (k_fold_label_copy[i])

            train_set = []
            train_set_label = []

            training_set_object = {}
            for j in range(0, len(k_fold_copy)):
                train_set.extend(k_fold_copy[j].tolist())
                train_set_label.extend(k_fold_label_copy[j].tolist())

            # create the training set object for this iteration
            training_set_object["data_points"] = train_set
            training_set_object["labels"] = train_set_label

            # store the training set and tuning set into this fold dictionary
            cross_val_sets[i] = [training_set_object, tuning_set_object]

        return cross_val_sets

    def convert_docs_to_bow_for_features(self, documents, all_terms):
        terms_in_documents = {}

        for term in all_terms:
            terms_in_documents[term] = 0

        tf_documents = []
        count = 0

        # calculate all the term frequencies in all the documents
        for document in documents:
            tf_document = deepcopy(terms_in_documents)

            for term in all_terms:
                doc_words = self.tokenize(document)
                number_of_words_in_current_document = len(doc_words)
                tf_document[term] = doc_words.count(term) / number_of_words_in_current_document

            count+=1

            print "Tf calculated for document: ", count
            tf_documents.append(tf_document)

        idf_terms = deepcopy(terms_in_documents)
        number_of_documents = len(tf_documents)

        # compute the idf per term.
        # idf = log(N/dft),
        # where N = Total number of documents
        # dft = Total number of document where the term appears
        for term in all_terms:
            docs_that_contain_term = [tf_doc for tf_doc in tf_documents if tf_doc[term] > 0]
            number_of_documents_that_contain_word = len(docs_that_contain_term)
            number_of_documents_that_contain_word = 1 if number_of_documents_that_contain_word == 0 else number_of_documents_that_contain_word

            idf_terms[term] = math.log(number_of_documents / number_of_documents_that_contain_word)

        # final list containing the tf*idfs per feature i.e. word
        bow_tfidfs = []

        # iterate through tf of each document,
        # compute the tf*idf for each term of the document,
        # put them in an array of tuples
        # and return the list

        for tf_doc in tf_documents:
            feature_vec = []
            for term in all_terms:
                tf_doc[term] =  tf_doc[term] * idf_terms[term]
                feature_vec.append(tf_doc[term])

            bow_tfidfs.append(tuple(feature_vec))

        return bow_tfidfs

    def convert_sentiment_list_to_number(self, predictions):
        pred_arr = []
        for prediction in predictions:
            pred_val = 1 if prediction == 'POSITIVE' else -1
            pred_arr.append(pred_val)

        return pred_arr

    def write_to_file(self, file_name, features, headers, predictions):

        csvfile = codecs.open(file_name, 'w', 'utf-8')

        writer = csv.writer(csvfile)
        writer.writerow(headers)
        print headers

        for i in range(0, len(features)):
            feature = list(features[i])
            feature.append(predictions[i])
            print len(features[i])
            #print features
            writer.writerow(feature)

        csvfile.close()

    def print_set_len(self):
        dslen = len(self.features)
        print dslen
        print self.k
