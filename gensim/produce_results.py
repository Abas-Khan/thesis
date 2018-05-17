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

print "loading model"
model = Doc2Vec.load("./output/model_Dis2Vec_w_sample.word2vec")
print "model loaded"
query_doc = "An elderly 70 year old patient. Primary tumor in liver. metastasis in lung and liver".lower().split()
inferred_docvec = model.infer_vector(query_doc,steps=5000)
        #inferred_docvec = model.infer_vector(all_docs[0].words)
result = model.docvecs.most_similar([inferred_docvec], topn=20)
content = [h[0] for h in result]

for item in content:
    print item