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
                            min_count=self.params["min_cnt"], workers=cores,hs=0,negative=5,
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
            model.save(out_folder + 'new_vocab_model')
        else:
            model.save(out_folder + 'new_vocab_model')

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
            label = " ".join(doc[:10])
            doc = list(filter(lambda word: word not in stopwords.words('english'), doc))
            
            doc = [element.lower() for element in doc]
        
            yield TaggedDocument(doc, [label])
            #break

def main():

    
   # sentences_corpus = pickle.load(open(_arg.inputcorpus, "r")) # Input corpus (list of sentences as input where each sentence is a list of tokens. file should be in .pkl format).
    sentences =  LineSentence('big_home_test.txt')
    contents = TaggedTester(sentences)

    '''
    for item in contents:
        print item
    '''

    vocab_list = ['rs10795668', 'miR-135a', 'Lynch syndrome I', 'biopsy', 'disease', 'C18.8', 'FOLFIRI-CETUXIMAB', 'rs4939827', 'IIIB', 'colon carcinoma', 'outcome', 'transverse colon cancer', 'therapy resistance', 'CTNNB1', 'IIIA', 'rs1035209', 'family history', 'relapse free survival', 'p14', 'anastomosis', 'Cowden syndrome', 'oxaliplatin', 'MSI-H', 'bleeding', 'DNA Image Cytometry', 'CAPOX', 'weight loss', 'ICD', 'Endorectal MRI', 'aflibercept', 'argon', 'EGF', 'immunotherapy', 'physical activity', 'rs4925386', 'C18.0', 'side effects', 'disease subtypes', 'angiogenesis inhibitors', 'cloacogenic carcinoma', 'colonic neoplasms', 'CD29', 'dysplasia in inflammatory bowel disease', 'serrated polyposis', 'EpCAM', 'intestinal polyposis', 'rs1800469', 'CD44', 'miR-135b', 'G1n1317', 'rs34612342', 'symptoms', 'rectal cancer', 'ramucirumab', 'interstitial brachytherapy', 'VEGFA', 'tetraploid', 'MSI', 'RX', 'FAP', 'Array-CGH', 'miR-92', 'irinotecan', 'T4a-N2a-M0', 'adenomatous polyposis syndromes', 'colon cancer', 'radiofrequency ablation', 'hereditary nonpolyposis type 5', 'R2', 'microRNA markers', 'mucositis', 'RAS-MAPK', 'gardner syndrome', 'genes', 'neoadjuvant chemo', 'IIC', 'adjuvant chemo', 'double contrast barium enema', 'MGMT', 'smoking', 'euploid', 'tingling', 'cyramza', 'monoclonal antibodies', 'vomiting', 'appetite loss', 'nausea', 'C18.4', 'MLH1', 'miR-155', 'C18.6', 'IHC MSI markers', 'barium enema', 'hamartomatous polyposis syndromes', 'MSH6', 'response', 'biomarkers', 'D17S250', 'rs12603526', 'hereditary nonpolyposis', 'alcohol', 'PI3K', 'RTK', 'nausea', 'blood disorders', 'lack of physical exercise', 'follow-up', 'immune checkpoint inhibitors', 'pembrolizumab', 'transanal endoscopic microsurgery', 'weakness', 'colorectal cancer', 'rs10911251', 'polymerase proofreading-associated polyposis', 'IIB','DNA MSI test results', 'molecular features', 'descending  colon cancer', 'C18.5', 'T4b-N0-M0', 'hepatic artery infusion', 'molecular marker testing', 'rs1799977', 'predictive', 'p16', '18q AI expression', 'stereotactic', 'anus neoplasms', 'CD133', 'fever', 'IVB', 'good', 'colon Kaposi sarcoma', 'WNT', 'E1317Q', 'rs3802842', 'weak muscle', 'Tis-N0-M0', 'splenic flexure cancer', 'chemotherapy', 'targeted therapy', 'C18.7', 'Turcot syndrome', 'miR-21', 'rs4779584', 'adenosquamous colon carcinoma', 'pathways', 'upsetstomach', 'gender male', 'rs11169552', 'survival', 'rs459552', 'rs3217810', 'internal', 'overall survival', 'rectal bleeding', 'BRAF mutation', 'T1-N0-M0', 'external beam', 'PMS2 loss', 'blood based', 'Gardner syndrome', 'attenuated adenomatous polyposis coli', 'PTGS2', 'T2-N0-M0', 'ploidy status', 'genomic instability', 'bloody stools', 'progressive disease', 'hereditary nonpolyposis type 8', 'nervous system effects', 'headaches', 'stomach pain', 'five-year survival', 'local excision', 'types', 'hereditary nonpolyposis type 6', 'III', 'T1\xe2\x80\x93T2-N1/N1c-M0', 'therapy', 'hair loss', 'CEA', 'chemotherapy drugs', 'rs3824999', 'colon lymphoma', 'recurrence', 'ulcerative colitis', 'disease etiology', 'G2', 'apoptotic', 'IIIC', 'Any T -Any N-M1b', '0', 'high red meat diet', 'Juvenile polyposis syndrome', 'rs1800734', 'microscopy', 'dMMR', 'fitness', 'R0', 'MRI', 'skin irritation', 'leukopenia', 'NGS', 'systemic', 'desmoid disease', 'POLE', 'CTC', 'miR-211', 'IIA', 'rs12241008', 'malignancy', 'G13D', 'rs961253', 'age', 'hereditary mixed polyposis syndrome 2', 'DPYD', 'Epigenetic gene silencing', 'F594L', 'constipation', 'cologuard', 'hereditary colon cancer', 'T4b-N1\xe2\x80\x93N2-M0', 'poor', 'obesity', 'partial', 'regional', 'R1', 'thrombocytopenia', 'dMMR test', 'colon sarcoma', 'rs174550', 'peeling', 'rectum cancer', 'T1\xe2\x80\x93T2-N2b-M0', 'D2S123', 'rs4444235', 'laparoscopy', 'CIN markers', 'loss of balance', 'laser therapy', 'KRAS mutational testing', 'SNPs', 'liver metastasis', 'prognosis', 'rs1321311', 'CT', 'aneuploid', 'G12V', 'KRAS', 'rs36053993', 'MSI test', 'hereditary nonpolyposis type 4', 'APC', 'TIMP-1', 'G4', 'p53 expression', 'FDA approveddrugs', 'G12S', 'single specimen guaiac FOBT', 'combinations', 'neuropathy', 'MLH1 loss', 'endocavitary', 'fungal infection', 'hereditary nonpolyposis type 1', 'BRAF mutation test', 'anemia', 'CEA assay', 'colorectal neoplasms', 'polyploidy test', 'regorafenib', 'G1', 'DNA MSI markers', 'Peutz-Jeghers syndrome', 'adenomatous polyposis coli', 'rs10411210', 'EPCAM', 'colectomy', 'prognostic', 'autosomal recessive colorectal adenomatous polyposis', 'hereditary nonpolyposis type 3', 'rs158634', 'colonic L-cell glucagon-like peptide producing tumor', 'C20', 'metastatic colorectal cancer', 'XELIRI', 'burning', 'Hyperplastic Polyposis Syndrome', 'bevacizumab', 'rectosigmoid juction cancer', 'european', 'T2\xe2\x80\x93T3-N2a-M0', 'carbon dioxide', 'CD24', 'tumor MSI-H expression', 'colorectal adenocarcinoma', 'Any T- Any N-M1a', 'virtual colonoscopy', 'Crohn&apos;s disease', 'tenderness', 'diploid', 'T3\xe2\x80\x93T4a-N1/N1c-M0', 'PMS2', 'muscle pain', 'FOLFIRI-BEVACIZUMAB', 'rectal neoplasms', 'predictive biomarker', 'BRAF', 'NRASmutation', 'BAT25', 'PET', 'rs1042522', 'complete', 'CIN', 'sigmoid colon cancer', 'ascending colon cancer', 'radiation therapy','KRT20', 'mouth and throat sores', 'BAT26', 'APC mutations', 'DRE', 'colon leiomysarcoma', 'fatigue', 'RAS mutation test', 'C19','diagnosis', 'shaking', 'Lynch syndrome', 'C18.9', 'tyrosine kinase inhibitors', 'risk factors', 'CA 19-9', 'hMLH1', 'MSH2 loss','rs4813802', 'colostomy', 'screening', 'V600E', 'colon singlet ring adenocarcinoma', 'altered bowel habits', 'XELOX', 'IVA', 'II', 'stable disease', 'rs12309274', 'I', 'hereditary nonpolyposis type 7', 'lung metastasis', 'anal canal carcinoma', 'FU-LV', 'prognostic biomarker', 'colon small cell carcinoma', 'resectability', 'rs647161', 'Li-Fraumeni syndrome', 'Q61K', 'rs10936599', 'sexual issues', 'rs7758229', 'hepatic flexure cancer', 'proctectomy', 'clinical features', 'MSH2', 'DNA mismatch-repair', 'C18.2', 'MRT', 'cryosurgery', 'PIK3CA', 'hereditary mixed polyposis syndrome 1', 'oligodontia-colorectal cancer syndrome', 'SEPT9 methylation', 'FIT', 'lonsurf', 'exercise', 'pain', 'east asian', 'colonoscopy', 'adenomas', 'TGF-\xce\xb2', 'G12D', 'rs704017', 'surgery', 'Faecal M2-PK', 'polyploidy test results', 'MSH6 loss', 'inherited genetic disorders', 'Lgr5', 'KRAS mutation', 'submucosal invasivecolon adenocarcinoma', 'BMI', 'R classification', 'rs9929218', 'sigmoidoscopy', 'stem cell', 'MUTYH-associated polyposis', '5-FU', 'VEGF', 'T3\xe2\x80\x93T4a-N2b-M0', 'nonpolyposis syndrome', 'T1-N2a-M0', 'hyperthermia', 'high fat intake', 'type of care', 'G3', 'population based SNP', 'ALK', 'miR-92a', 'CD166', 'anal gland neoplasms', 'T4a-N0-M0', 'metastasis', 'D5S346', 'rs10849432', 'blistering', 'rs61764370', 'rs1801155', 'PLOD1', 'C18.3', 'optical colonoscopy', 'miR-31', 'rs16892766', 'IV', 'rectosigmoid cancer', 'panitumumab', 'T3-N0-M0', 'miR-17', 'GX', 'FISH', 'cognitive dysfunction', 'EGFR', 'rs1801166', 'prognostic factors', 'bladder irritation', 'acute myelocytic leukemia', 'TYMS', 'UICC staging', 'FOLFOX', 'lipomatous hemangiopericytoma', 'rs6691170', 'ALDH1', 'tumor budding', 'MUTYH', 'MSS', 'grade', 'attenuated familial adenomatous polyposis', 'colon adenocarcinoma', 'high sensitivity faecal occult blood test', 'Samson Gardner syndrome', 'colon mucinous adenocarcinoma', 'pMMR', 'TP53', 'G463V', 'capsule colonoscopy', 'colon squamous cell carcinoma', 'rectal irritation', 'C18.1', 'HRAS', 'CEACAM5', 'neodymium:yttrium-aluminum-garnet', 'cetuximab', 'FOLFIRI', 'rs6983267', 'MSI-L', 'C18']


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