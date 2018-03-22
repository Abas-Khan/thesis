#!/usr/bin/python
# -*- coding: utf-8 -*-
# :LICENSE: MIT

__author__ = "Saurav Ghosh"
__email__ = "sauravcsvt@vt.edu"

import os

import cPickle as pickle
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords

from gensim.models.doc2vec import TaggedLineDocument
import time
import logging
import argparse
import numpy as np
import multiprocessing
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.parsing.preprocessing import remove_stopwords

from gensim.models import Phrases
from gensim.models.phrases import Phraser


import sys
import codecs


logging.basicConfig()

class Dis2Vec(object):

    def __init__(self, **kwargs):
        self.params = kwargs

    def run_Dis2Vec(self):

        start_time = time.time()
        cores = multiprocessing.cpu_count()

       
        all_docs = self.params["tag_doc"]
        model = Doc2Vec(self.params["sent"], vector_size=self.params["dim"], window=self.params["win"], 
                            min_count=self.params["min_cnt"], workers=1,hs=0,negative=5,
                            dm=0,dbow_words=1,epochs=self.params["iter"], smoothing=0.5,
                            sampling_param=0.3, objective_param=0.3, vocab_file=self.params["vocab"])
                            
        # if you pass the document to the model , no need to build the vocabulary
        #model.build_vocab(self.params["sent"])

        # No need to train either if you provide the documents to the model it will train on it automatically
        #model.train(self.params["sent"], total_examples=model.corpus_count, epochs=model.iter)
        #for word, vocab_obj in model.wv.vocab.items():
        #    print word
        #query_doc = "".split()
        #inferred_docvec = model.infer_vector(query_doc,steps=500)
        #inferred_docvec = model.infer_vector(all_docs[0].words)
        #print model.docvecs.most_similar([inferred_docvec], topn=10)
        #print model.similar_by_word("Bangladesh", topn=10)

        query_doc = "tajiskistan is a small landlocked country in asia with an estimated population of 8.7 million people.".split()
        inferred_docvec = model.infer_vector(query_doc,steps=100)
        #inferred_docvec = model.infer_vector(all_docs[0].words)
        print model.docvecs.most_similar([inferred_docvec], topn=10)


        query_doc = "Football also known as soccer is a team sports that involves kicking a ball to score a goal , word football , association football , football codes".split()
        inferred = model.infer_vector(query_doc,steps=500)
        #inferred_docvec = model.infer_vector(all_docs[0].words)
        print model.docvecs.most_similar([inferred], topn=10)
        #
        
        print " \n  results \n"
        '''
        for doc in all_docs:   
            print doc[0][0:2]
            inferred_docvec = model.infer_vector(doc.words)
            print model.docvecs.most_similar([inferred_docvec], topn=10)
        '''
       
        #print model.wv.get_vector('sport')
        model.wv.setvector('das-land','country')
        print model.similar_by_word("country", topn=10)
   
        #print "....................."
        #print model.most_similar_cosmul("sport",topn=10)
        end_time = time.time()
        print ("Total time taken is: " + str((end_time - start_time) / 3600.) + " hours")

        	
        X = model[model.wv.vocab]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        pyplot.scatter(result[:, 0], result[:, 1])
        words = list(model.wv.vocab)
        for i, word in enumerate(words):
	        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()   
        out_folder = './output/'
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        if model.sample == 0:
            model.save(out_folder + 'model_Dis2Vec.word2vec')
        else:
            model.save(out_folder + 'model_Dis2Vec_w_sample.word2vec')

def parse_args():

    ap = argparse.ArgumentParser()
    ap.add_argument("-ic", "--inputcorpus", type = str, required = True, help = "Input corpus which should be a list of sentences as input where each sentence is a list of tokens. file should be in .pkl format")
    ap.add_argument("-v", "--domainvocab", type = str, required = True, help = "Domain-specific vocabulary. file should be in .pkl format")
    ap.add_argument("-d", "--dim", type = str, required = True, help = "Dimension of word embeddings (300, 600)")
    ap.add_argument("-w", "--window", type = str, required = True, help = "Word window (5, 10, 15)")
    ap.add_argument("-n", "--negative", type = str, required = True, help = "Number of negative samples (1, 5, 15)")
    ap.add_argument("-spm", "--samplingparameter", type = str, required = True, help = "Sampling parameter (0.3, 0.5, 0.7)")
    ap.add_argument("-opm", "--objectiveparameter", type = str, required = True, help = "Objective selection parameter (0.3, 0.5, 0.7)")
    ap.add_argument("-sm", "--smoothing", type = str, required = True, help = "smoothing parameter (0.75, 1.0)")
    return ap.parse_args()


class TaggedTester(object):
    def __init__(self, data):
        self.data = data
        #self.wiki.metadata = True
    def __iter__(self):
        for line_no,doc in enumerate(self.data):
            doc = list(filter(lambda word: word not in stopwords.words('english'), doc))
            label = " ".join(doc[:5])
            doc = [element.lower() for element in doc]
        
            yield TaggedDocument(doc, [label])

def main():

    
   # sentences_corpus = pickle.load(open(_arg.inputcorpus, "r")) # Input corpus (list of sentences as input where each sentence is a list of tokens. file should be in .pkl format).
    sentences =  LineSentence('countries_filter.txt')
    contents = TaggedTester(sentences)
    #contents = TaggedLineDocument("countries_filter.txt")


    
    #phrases = Phrases(contents, min_count=1)
    #bigram = Phraser(phrases)
    # filtered_words = list(filter(lambda word: word not in stopwords.words('english'), line))
           
    '''
    tagged_docs = []
    for item_no, line in enumerate(contents):
            filtered_words = list(filter(lambda word: word not in stopwords.words('english'), line))
            tagged_docs.append(TaggedDocument(bigram[filtered_words], [item_no]))
    '''
    #sentences = TaggedDocument(bigram[contents])



    domain_vocab_file = "sports sport players teams team score scores scored game games ball pass win wins play opponent net court opponent's"
    vocab_list = domain_vocab_file.split()


    dim = 300
    win = 8
    neg = 5

    #print tagged_docs[1]
    #sys.exit(0)
   
    kwargs = {"sent": contents, "vocab": vocab_list, 
              "dim": dim, "win": win, "min_cnt": 1, "neg": neg, "iter":30 , "tag_doc" :contents
              }
    Dis2Vec(**kwargs).run_Dis2Vec()
    

if __name__ == "__main__":
    main()
