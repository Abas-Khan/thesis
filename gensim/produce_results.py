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
from gensim.parsing.preprocessing import stem_text


import sys
import codecs

print "loading model"
#NOTE fixed_stemmoutput was trained wthout removing stopwords
model = Doc2Vec.load("./Fixed_Labels_Multi-tag/mod")
print "model loaded"
query_doc = "Influence of a new virostatic compound on the induction of enzymes in rat liver. The virostatic compound N,N-diethyl-4-[2-(2-oxo-3-tetradecyl-1-imidazolidinyl)-ethyl]-1-piperazinecarboxamide-hydrochloride (5531) analyzed effect induction tryptophan-pyrrolase tyrosineaminotransferase rat liver. 1. The basic activity enzymes influenced substance either normal adrenalectomized animals. 2. The induction enzymes cortisone increased presence compound whereas substrate induction remained unchanged. 3. The induction tyrosine-aminotransferase dexamethasonephosphate tissue culture inhibited dose compound 5531 higher 5 mug/ml."

new_q = "70 years old elderly patient gender male.Primary tumor in lung,  lung metastasis chemotherapy null radiotherapy "
#query_doc = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:"," ",query_doc)



new_q = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:"," ",new_q)
                
#print "before stem",new_q
new_q = remove_stopwords(new_q)
new_q = stem_text(new_q).split()


#query_doc = remove_stopwords(query_doc)
#query_doc = stem_text(query_doc).split()

bigram = Phraser.load('./gensim_stopwords_stemmed_big_phrases')
trigram = Phraser.load('./gensim_stopwords_stemmed_trigram_phrases')


#print query_doc

print trigram[bigram[new_q]]
inferred_docvec = model.infer_vector(trigram[bigram[new_q]],steps=5000)
        #inferred_docvec = model.infer_vector(all_docs[0].words)
result = model.docvecs.most_similar([inferred_docvec], topn=20)

print result
#content = [h[0] for h in result]

#for item in content:
#    print item
