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
import codecs
from gensim.parsing.preprocessing import remove_stopwords

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.parsing.preprocessing import stem_text


import sys
import codecs


print "loading model"
#NOTE fixed_stemmoutput was trained wthout removing stopwords
model = Doc2Vec.load("./preprocessed_negative_sampling_ten/mod")
print "model loaded"
query_doc = "Influence of a new virostatic compound on the induction of enzymes in rat liver. The virostatic compound N,N-diethyl-4-[2-(2-oxo-3-tetradecyl-1-imidazolidinyl)-ethyl]-1-piperazinecarboxamide-hydrochloride (5531) analyzed effect induction tryptophan-pyrrolase tyrosineaminotransferase rat liver. 1. The basic activity enzymes influenced substance either normal adrenalectomized animals. 2. The induction enzymes cortisone increased presence compound whereas substrate induction remained unchanged. 3. The induction tyrosine-aminotransferase dexamethasonephosphate tissue culture inhibited dose compound 5531 higher 5 mug/ml."

new_q = "70 year-old man. elderly patient gender male. colon cancer. colonic carcinoma lung metastases pulmonary metastases"
#query_doc = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:"," ",query_doc)



new_q = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:|\?|\;"," ",new_q)
                
#print "before stem",new_q
new_q = remove_stopwords(new_q)
new_q = stem_text(new_q).split()


#query_doc = remove_stopwords(query_doc)
#query_doc = stem_text(query_doc).split()


bigram = Phraser.load('./preprocessed_big_phrases')
trigram = Phraser.load('./preprocessed_trigram_phrases')

#age_test = stem_text("16 years-old").split()

#print trigram[bigram[age_test]]



#print query_doc

print trigram[bigram[new_q]]
 
inferred_docvec = model.infer_vector(trigram[bigram[new_q]],steps=5000)
        #inferred_docvec = model.infer_vector(all_docs[0].words)
#result = model.docvecs.most_similar([inferred_docvec] + model[trigram[bigram[stem_text("small cell carcinoma").split()]]], topn=20)
result = model.docvecs.most_similar([inferred_docvec]+ model[trigram[bigram[stem_text("lung metastases").split()]]], topn=20)

print result

fin = open("age_fix.txt","r").read()
data = fin.split("\n")

content = [h[0] for h in result]

colon_cancer_vocab = ['colon','colonic','sigmoid']
lung_vocab = ['pulmonary','lung','lungs','bronchus','bronchogenic','bronchi']
liver_vocab = ['hepatic','liver','hepatocellular']
judge = colon_cancer_vocab + liver_vocab

colorectal_vocab = ['esophagus','stomach','liver','hepatic','gallbladder','intestine','colon','colonic','cecum','rectum','rectal','pancreas','sigmoid','anus','anal','renal','pulmonary','lung','bronchus','bronchogenic','bronchi','adenocarcinoma','adenocarcinomas']

print "\n Full text"
result = []
result_preprocessed = []
for item in content:
        temp = filter(lambda x: item in x, data)
        #temp = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:|\?|\;"," ",temp)
        result.extend(temp)

    
for record in result:
        print record
        temp = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:|\?|\;"," ",record)
        result_preprocessed.append(temp)        


print "processed results ",result_preprocessed    
relevant_results = []    

relevant_results.extend([s.split() for s in result_preprocessed if any(xs in s.lower().split() for xs in colorectal_vocab)]) 
print len(relevant_results)   
