#!/usr/bin/python
# -*- coding: utf-8 -*-
# :LICENSE: MIT

__author__ = "Abbas Khan"

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
from gensim.corpora.wikicorpus import WikiCorpus
from gensim import utils, matutils

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
                            sampling_param=0.5, objective_param=0.3, vocab_file=self.params["vocab"])
                            
        end_time = time.time()
        print ("Total time taken is: " + str((end_time - start_time) / 3600.) + " hours")

        model.save("dis2wiki")        	
     


class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for idx,(content, (page_id, title)) in enumerate(self.wiki.get_texts()):
            print content[0:5]
            yield TaggedDocument([c for c in content], [title])
            if idx > 500:
                break
            
           

def main():

   
    wiki = WikiCorpus("enwiki-latest-pages-articles.xml.bz2")
    documents = TaggedWikiDocument(wiki)
 


    domain_vocab_file = "poems poets poem poet symfony symfonies ghazal ghazals song lyrics"
    vocab_list = domain_vocab_file.split()


    dim = 200
    win = 8
    neg = 5
   
    kwargs = {"sent": documents, "vocab": vocab_list, 
              "dim": dim, "win": win, "min_cnt": 19, "neg": neg, "iter":20 , "tag_doc" :documents
              }
    Dis2Vec(**kwargs).run_Dis2Vec()
    

if __name__ == "__main__":
    main()

