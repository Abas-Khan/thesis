#!/usr/bin/python
# -*- coding: utf-8 -*-
# :LICENSE: MIT

__author__ = "Saurav Ghosh"
__email__ = "sauravcsvt@vt.edu"

import os

import re
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
                            dm=0,dbow_words=1,epochs=self.params["iter"], smoothing=0.75,
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
        #model.wv.setvector('das-land','country')
        #model.wv.setvector('football','soccer')
        query_doc = "Radioactive labeling and location of specific thiol groups in myosin from fast, slow and cardiac muscles.1. Based incorporation radioactively labeled N-ethylmaleimide, readily reactive thiol groups isolated myosin (EC 3.6.1.3) fast, slow cardiac muscles could classified 3 types. All 3 myosins contain 2 thiol-1, 2 thiol-2 variable number thiol-3 groups per molecule. Both thiol-1 thiol-2 groups essential functioning K+-stimulated ATPase, located heavy chains 3 myosin types. 2. The variation incorporation pattern N-ethylmaleimide 3 thiol group classes steady-state conditions Mg(2+) - ATP hydrolysis allowed different conformations reaction intermediates characterized. In 3 types myosin hydrolytic cycle Mg(2+) - ATP found controlled step 25 degrees C. In three cases, rate-limiting step changed way lowereing temperature. 3. Using chemically determined molecular weights myosin light chains, stoichiometry found basis sodium dodecyl sulfate electrophoresis 1.2 : 2.1 : 0.8 light chain-1: light chain-2:light chain-3 per molecule fast myosin, 2.0 : 1.9 light chain-1:light chain-2 per molecule slow myosin 1.9 : 1.9 light chain-1:light chain-2 per molecule cardiac myosin. This qualitative difference light subunit composition fast two types slow myosin reflected small variations characteristics exhibited isolated myosins, rather seems connected respective myofibrillar ATPase activities.".split()
        inferred_docvec = model.infer_vector(query_doc,steps=100)
        #inferred_docvec = model.infer_vector(all_docs[0].words)
        print model.docvecs.most_similar([inferred_docvec], topn=10)


        #query_doc = "Football also known as soccer is a team sports that involves kicking a ball to score a goal , word football , association football , football codes".split()
        #inferred = model.infer_vector(query_doc,steps=500)
        #inferred_docvec = model.infer_vector(all_docs[0].words)
        #print model.docvecs.most_similar([inferred], topn=10)
        #
        
        print " \n  results \n"
        '''
        for doc in all_docs:   
            print doc[0][0:2]
            inferred_docvec = model.infer_vector(doc.words)
            print model.docvecs.most_similar([inferred_docvec], topn=10)
        '''
       
        #print model.wv.get_vector('sport')
        
        #print model.similar_by_word("country", topn=10)
   
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

class TaggedTester(object):
    def __init__(self, data):
        self.data = data
        #self.wiki.metadata = True
    def __iter__(self):
        for line_no,doc in enumerate(self.data):
            #str_doc = " ".join(doc)
            #str_doc = re.sub(r'(?<!\d)[.,;:](?!\d)', '', str_doc)

            
            #str_doc = re.sub(r"(\S)\(",r"\1 ( ",str_doc)
            #print str_doc
            #sys.exit()
            doc = list(filter(lambda word: word not in stopwords.words('english'), doc))
            label = " ".join(doc[:5])
            doc = [element.lower() for element in doc]
        
            yield TaggedDocument(doc, [label])
            #break

def main():

    
   # sentences_corpus = pickle.load(open(_arg.inputcorpus, "r")) # Input corpus (list of sentences as input where each sentence is a list of tokens. file should be in .pkl format).
    sentences =  LineSentence('refined.txt')
    contents = TaggedTester(sentences)

    '''
    for item in contents:
        print item
    '''

    domain_vocab_file = "study investigations effects studies analysis evidence liver cancer"
    vocab_list = domain_vocab_file.split()


    dim = 200
    win = 8
    neg = 5
   
    kwargs = {"sent": contents, "vocab": vocab_list, 
              "dim": dim, "win": win, "min_cnt": 2, "neg": neg, "iter":20 , "tag_doc" :contents
              }
    Dis2Vec(**kwargs).run_Dis2Vec()
    

if __name__ == "__main__":
    main()


'''
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

'''