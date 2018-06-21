# -*- coding: utf-8 -*-
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
from gensim.similarities.index import AnnoyIndexer
from gensim.summarization import keywords
from gensim.corpora import Dictionary


import sys
import codecs


print "loading model"
#NOTE fixed_stemmoutput was trained wthout removing stopwords
model = Doc2Vec.load("./latest/mod")
print "model loaded"
#query_doc = "A histochemical method of differentiating lower gastrointestinal mucin from other null mucins in primary or null metastatic tumours.Epithelial null mucins normal terminal null ileum null caecum null colon null rectum man unique alone exhibit staining following null periodate null borohydride null technique. Application technique null enables one differentiate mucin producing metastases arising adenocarcinoma lower gastrointestinal tract arising elsewhere, may occasionally useful determining site null primary tumour null doubt. Furthermore, found especially useful distinguishing primary adenocarcinoma null lung metastases null lung adenocarcinoma lower gastrointestinal tract."

#query_doc = "Cancer markers in patients receiving null chemotherapy for null colorectal cancer null: a preliminary report.The combination CEA, hepatic function marker null enzymes, null four acute phase reactant proteins (haptoglobin, null alpha 1 null antitrypsin, null alpha 1 acid glycoprotein null, prealbumin) used null  monitor patients colorectal cancer receiving chemotherapy. In 18 patients advanced lesions survived least 3 months treatment markers predicted progression 92% 25 incidents progression; mean lead time 2.8 null months. A rising CEA present 28%, patients gave mean lead time 4 months. In group 14 patients minimal residual disease progression clinically detectable disease occurred 9 them. In cases markers predicted progression mean lead time 6 months; six patients markers indicated progression, yet disease detectable, mean lead time least 8.6 months. CEA liver enzyme markers sensitive indicators progression minimal residual disease group."
#new_q = "70 year-old man. elderly patient gender male. colon cancer. colon carcinoma. lung metastases pulmonary metastases"

#query_doc = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:|\?|\;"," ",query_doc)


#sample_query = "70 year-old man. elderly patient gender male. colon tumor. curative surgery. resection with primary anastomosis. palliative tumor resection. tumor resection. primary tumor size. size of primary tumor. regional lymph nodes. liver metastasis."

#new_q = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:|\?|\;"," ",new_q)
                
#print "before stem",new_q

#print keywords(query_doc,words=8)

#query_doc = remove_stopwords(query_doc)
#query_doc = stem_text(query_doc).split()


bigram = Phraser.load('./preprocessed_big_phrases')
trigram = Phraser.load('./preprocessed_trigram_phrases')

col_1="The patient was a 47 year-old man, whose chief complaint was melena. He visited a nearby hospital, and further evaluation showed rectal cancer invading the prostate, with multiple lung and liver metastases. The clinical diagnosis was cT4b(prostate), cN1, cM1b(H2, PUL2), cStage IV . We performed colostomy in the transverse colon prior to chemotherapy. He was administered 1 course of mFOLFOX6 plus bevacizumab and 7 courses of FOLFOXIRI plus bevacizumab. The primary tumor showed PR. The liver metastases were localized and shrunken, while the lung metastases disappeared. Approximately 6 months after the start of chemotherapy, a laparoscopic total pelvic exenteration and ileal conduit were performed following the diagnosis of ycT4b(prostate), ycN1, ycM1a(H2), ycStage IV . About 3 months later, a partial resection of the left liver lobes(S1 and S5/S8)was performed laparoscopically. He has been cancer-free for 8 months."
col_1 = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:|\?|\;"," ",col_1)

col_1 = remove_stopwords(col_1)
col_1 = stem_text(col_1).split()

col_1 = trigram[bigram[col_1]]

#print col_1
#inferred_docvec = model.infer_vector(trigram[bigram[new_q]],steps=5000)

col_2 = "Combined null modality therapy is sufficient to treat advanced rectal cancer with multiple metastases. Her, we report a case of long-term survival in a patient with multiple metastases from rectal cancer. A5 8-year-old man had previously undergone low anterior resection for advanced rectal cancer. Multiple liver and lung metastases were identified prior to operation; therefore, we initiated null chemotherapy (FOLFOX). Partial resection of metastatic lesions and radiofrequency ablation(RFA)were also administered, but newly developed liver, lung, and adrenal gland metastases were identified. We changed the chemotherapy null regimen and administered topical therapies(partial resection, RFA, hepatic arterial infusion null chemotherapy, null radiotherapy)for each chemotherapy refractory metastatic lesion. Although the patient is in a tumor bearing state, he is still alive 10 years after his first operation. This combined modality therapy is an option for patients with chemotherapy null refractory metastases from rectal cancer."
col_2 = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:|\?|\;"," ",col_2)

col_2 = remove_stopwords(col_2)
col_2 = stem_text(col_2).split()

col_2 = trigram[bigram[col_2]]

#print col_2
#inferred_docvec = model.infer_vector(trigram[bigram[col_2]],steps=5000)


non_col = "Objective: To investigate the clinicopathologic and molecular features of the rare cribriform morular variant of papillary thyroid carcinoma (CMV-PTC). Methods: The clinicopathologic data of 10 patients with CMV-PTC were retrospectively reviewed. Immunohistochemical (IHC) staining was done using LSAB method. DNA sequencing for APC were applied using Sanger method. BRAF V600E mutation was examined using ARMS method. The cytological, morphological, IHC and molecular features were analyzed. Results: All patients were female at an average age of 27 years old. The tumors were mostly located in the right lobe of thyroid. Fine needle aspiration cytology was performed in three patients; two were diagnosed as suspicious for PTC and one as PTC. Nine tumors presented as solitary nodule and two as multiple nodules in both lobes. Infiltration was demonstrated in three cases. The average size was 2.6 cm. The neoplastic cells were arranged in papillary, cribriform, solid and glandular patterns, with rare or without colloid inside the lumen. The number of morula varied, ranging from zero to many. The neoplastic cells were variably enlarged, showing round, oval or spindle shape. Nuclear irregularity was identified as irregular membrane, nuclear grooves or pseudoinclusion, but no typical ground glass feature. Peculiar nuclear clearing could be observed in the morular cells. IHC staining showed the neoplastic cells were negative for thyroglobulin and p63, but positive for TTF1, cytokeratin 19 and estrogen receptor. Diffuse staining with cytokeratin was seen in the neoplastic cells and the morula. Specific cytoplasmic and nuclear staining of β-catenin was seen in the neoplastic cells but not the morula. Ki-67 proliferation index was 1%-30%. No recurrence or metastasis was observed. One patient was demonstrated to harbor both somatic and germline mutations of the APC gene, who was found to have adenomatous polyposis and her mother died of colonic carcinoma. No BRAF V600E mutation was detected. Conclusions: CMV-PTC is rare and shows atypical cytological and clinicopathological features, and it is easily misdiagnosed.TG, TTF1, ER and β-catenin are specific IHC markers for CMV-PTC. The morula is negative for cytokeratin 19, in contrast to squamous metaplasia. Although CMV-PTC has indolent clinical behavior, a definite diagnosis is necessary to rule out the possibility of APC gene mutation and related extra-thyroidal neoplasm, such as FAP and Gardner syndrome."
non_col = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:|\?|\;"," ",non_col)

non_col = remove_stopwords(non_col)
non_col = stem_text(non_col).split()

non_col = trigram[bigram[non_col]]

#print non_col
#inferred_docvec = model.infer_vector(trigram[bigram[non_col]],steps=5000)



print model.docvecs.similarity_unseen_docs(model,col_1,col_2, alpha=0.1, min_alpha=0.0001, steps=5000)

print " VS "
print model.docvecs.similarity_unseen_docs(model,col_2 ,non_col, alpha=0.1, min_alpha=0.0001, steps=5000)

'''

#age_test = stem_text("16 years-old").split()

ngrams = trigram[bigram[query_doc]]


for item in ngrams:
        try:
                model[item]
        except: print item," does not exist"        



#print query_doc

#print trigram[bigram[new_q]]
 
inferred_docvec = model.infer_vector(trigram[bigram[query_doc]],steps=5000)
        #inferred_docvec = model.infer_vector(all_docs[0].words)
#result = model.docvecs.most_similar([inferred_docvec] + model[trigram[bigram[stem_text("small cell carcinoma").split()]]], topn=20)

#indexer = AnnoyIndexer(model, 200)
#indexer=indexer
result = model.docvecs.most_similar([inferred_docvec] , topn=20)

print result

fin = open("age_fix.txt","r").read()
data = fin.split("\n")

content = [h[0] for h in result]

colon_cancer_vocab = ['colon','colonic','sigmoid']
lung_vocab = ['pulmonary','lung','lungs','bronchus','bronchogenic','bronchi']
liver_vocab = ['hepatic','liver','hepatocellular']
judge = colon_cancer_vocab + lung_vocab
judge = ["colorectal cancer"]


#colorectal_vocab = ['esophagus','stomach','liver','hepatic','gallbladder','intestine','colon','colonic','cecum','rectum','rectal','pancreas','sigmoid','anus','anal','renal','pulmonary','lung','bronchus','bronchogenic','bronchi','adenocarcinoma','adenocarcinomas']
colorectal_vocab = ['colorectal','hepatic','liver','hepatocellular','colon','colonic','cecum','rectum','rectal','sigmoid','anus','anal','pulmonary','lung','bronchus','bronchogenic','bronchi','adenocarcinoma','adenocarcinomas']

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
'''