from __future__ import division
import csv
from copy import deepcopy
import math
import numpy as np


class DataHandler(object):
    data_set = []
    prediction_labels = []
    data_set_file_name = ""
    prediction_label = ""
    cos_to_ignore = []

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
        print features
        print predictions
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

    def convert_docs_to_bow(self, documents):
        terms_in_documents = {}

        for document in documents:
            words = document.split()
            for word in words:
                if word not in terms_in_documents:
                    terms_in_documents[word] = 0

        tf_documents = []
        count = 0

        # get all terms into an array now, so that we put each term in the list in a fixed order
        all_terms = terms_in_documents.keys()

        # calculate all the term frequencies in all the documents
        for document in documents:
            tf_document = deepcopy(terms_in_documents)

            for term in all_terms:
                doc_words = document.split()
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

    def print_set_len(self):
        dslen = len(self.features)
        print dslen
        print self.k
