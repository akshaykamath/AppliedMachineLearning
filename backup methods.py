# def convert_docs_to_bow(self, documents):
#     terms_in_documents = {}
#
#     for document in documents:
#         words = self.get_clean_word_split(document)
#         for word in words:
#             if word not in terms_in_documents:
#                 terms_in_documents[word] = 0
#
#     tf_documents = []
#     count = 0
#
#     # get all terms into an array now, so that we put each term in the list in a fixed order
#     all_terms = terms_in_documents.keys()
#
#     # calculate all the term frequencies in all the documents
#     for document in documents:
#         tf_document = deepcopy(terms_in_documents)
#
#         for term in all_terms:
#             doc_words = self.get_clean_word_split(document)
#             number_of_words_in_current_document = len(doc_words)
#             tf_document[term] = doc_words.count(term) / number_of_words_in_current_document
#
#         count+=1
#
#         print "Tf calculated for document: ", count
#         tf_documents.append(tf_document)
#
#     idf_terms = deepcopy(terms_in_documents)
#     number_of_documents = len(tf_documents)
#
#     # compute the idf per term.
#     # idf = log(N/dft),
#     # where N = Total number of documents
#     # dft = Total number of document where the term appears
#     for term in all_terms:
#         docs_that_contain_term = [tf_doc for tf_doc in tf_documents if tf_doc[term] > 0]
#         number_of_documents_that_contain_word = len(docs_that_contain_term)
#
#         idf_terms[term] = math.log(number_of_documents / number_of_documents_that_contain_word)
#
#     # final list containing the tf*idfs per feature i.e. word
#     bow_tfidfs = []
#
#     # iterate through tf of each document,
#     # compute the tf*idf for each term of the document,
#     # put them in an array of tuples
#     # and return the list
#
#     for tf_doc in tf_documents:
#         feature_vec = []
#         for term in all_terms:
#             tf_doc[term] =  tf_doc[term] * idf_terms[term]
#             feature_vec.append(tf_doc[term])
#
#         bow_tfidfs.append(tuple(feature_vec))
#
#     return all_terms, bow_tfidfs
#test: 19938
#19938