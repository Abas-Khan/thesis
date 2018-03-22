#!/usr/bin/python
# -*- coding: utf-8 -*-
# :LICENSE: MIT

__author__ = "Saurav Ghosh"
__email__ = "sauravcsvt@vt.edu"

import os

import cPickle as pickle
from Dis2Vec.gensim.models import Word2Vec
from Dis2Vec.gensim.models import Doc2Vec
from  Dis2Vec.gensim.models.word2vec import LineSentence

from Dis2Vec.gensim.models.doc2vec import TaggedLineDocument
import time
import logging
import argparse
import numpy as np
logging.basicConfig()

class Dis2Vec(object):

    def __init__(self, **kwargs):
        self.params = kwargs

    def run_Dis2Vec(self):

        start_time = time.time()
        model_HM = Doc2Vec(self.params["sent"], size=self.params["dim"], window=self.params["win"], 
                            min_count=self.params["min_cnt"], sample=self.params["sample"], workers=100,
                            dm=0,dbow_words=1, hs=0, negative=self.params["neg"], iter=self.params["iter"], 
                            sampling_param=self.params["spm"], objective_param=self.params["opm"], 
                            smoothing=self.params["smoothing"], vocab_file=self.params["vocab"])
        end_time = time.time()
        print ("Total time taken is: " + str((end_time - start_time) / 3600.) + " hours")
        out_folder = './output/'
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        if model_HM.sample == 0:
            model_HM.save(out_folder + 'model_Dis2Vec.word2vec')
        else:
            model_HM.save(out_folder + 'model_Dis2Vec_w_sample.word2vec')

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

def main():

    
   # sentences_corpus = pickle.load(open(_arg.inputcorpus, "r")) # Input corpus (list of sentences as input where each sentence is a list of tokens. file should be in .pkl format).
    contents = TaggedLineDocument("countries_filter.txt")




    domain_vocab_file = "Sports Sport sport"
    vocab_list = domain_vocab_file.split()'


    dim = 300
    win = 5
    neg = 5
    spm = 0.3
    opm = 0.5
    smoothing = 0.75
    kwargs = {"sent": contents, "vocab": vocab_list, 
              "dim": dim, "win": win, "min_cnt": 5, "neg": neg, "iter": 1, 
              "spm": spm, "opm": opm, "smoothing": smoothing, "sample": 1e-05}
    Dis2Vec(**kwargs).run_Dis2Vec()

if __name__ == "__main__":
    main()
