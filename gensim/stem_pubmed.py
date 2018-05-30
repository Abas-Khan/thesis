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
from gensim import utils, matutils
from gensim.parsing.preprocessing import stem_text
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import any2unicode
logging.basicConfig()

class Dis2Vec(object):

    def __init__(self, **kwargs):
        self.params = kwargs

    def run_Dis2Vec(self):

        start_time = time.time()
        cores = multiprocessing.cpu_count()

       
        all_docs = self.params["tag_doc"]

    

        model = Doc2Vec(self.params["sent"], vector_size=self.params["dim"], window=self.params["win"], 
                            min_count=self.params["min_cnt"], workers=cores,hs=0,negative=10,
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
        out_folder = './Fixed_Labels_Multi-tag/'
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        if model.sample == 0:
            model.save(out_folder + 'mod')
        else:
            model.save(out_folder + 'mod')


class TaggedPubMed(object):

    def __init__(self, source):
    
        self.source = source
        self.bigram = Phraser.load('./gensim_stopwords_stemmed_big_phrases')
        self.trigram = Phraser.load('./gensim_stopwords_stemmed_trigram_phrases')

    def __iter__(self):
        with utils.smart_open(self.source) as fin,utils.smart_open("tag_info.txt","r") as exta_tag:
            for idx,(line,tag_info) in enumerate(zip(fin,exta_tag)):

                #line = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|'","",line)

                #Split words if they get stuffed together
                #line = re.sub( r"([A-Z][A-Z]*)", r" \1", line)


                #line = re.sub(r"\.(?=[^\d])|(?<=\w|\d)\(|(?<=\w|\d)\)"," ",line)
                #line = re.sub(r"(?<=\w[^\d])\.|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|'","",line)
                tag_info = re.sub(r"(?<=^) ","",tag_info)
                tags=  tag_info.replace("\n","").split(",")
                labels = []
                
                label = " ".join(line.split(" ")[:10])

                #NOTE split() will take care of extra spaces
                line = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:"," ",line)
                line = remove_stopwords(line)

                
                labels.append(label)
                #print label  
                if tags!=['']:
                    #   print tags
                    labels.extend(tags)
                
                line = stem_text(line).split()
                

                #Dropping the stop word calculation, Since phrases were calculated without it
                doc = line
                doc = self.trigram[self.bigram[doc]]
                

                #doc = [element.lower() for element in doc]
                yield TaggedDocument(doc, labels)   
                        

def main():

    
    #NOTE I used tr -d to remove ' from the file
    #contents = TaggedPubMed('big_home_test.txt')
    contents = TaggedPubMed("million_records.txt") 
    


    vocab_list = ['rs10795668', 'mir-135a', 'lynch_syndrom_i', 'biopsi', 'diseas', 'c18.8', 'folfiri-cetuximab', 'rs4939827', 'iiib', 'colon_carcinoma', 'outcom', 'transvers_colon_cancer', 'therapi_resist', 'ctnnb1', 'iiia', 'rs1035209', 'famili_histori', 'relaps_free_surviv', 'p14', 'anastomosi', 'cowden_syndrom', 'oxaliplatin', 'msi-h', 'bleed', 'dna_imag_cytometri', 'capox', 'weight_loss', 'icd', 'endorect_mri', 'aflibercept', 'argon', 'egf', 'immunotherapi', 'physic_activ', 'rs4925386', 'c18.0', 'side_effect', 'diseas_subtyp', 'angiogenesi_inhibitor', 'cloacogen_carcinoma', 'colon_neoplasm', 'cd29', 'dysplasia_in_inflammatori_bowel_diseas', 'serrat_polyposi', 'epcam', 'intestin_polyposi', 'rs1800469', 'cd44', 'mir-135b', 'g1n1317', 'rs34612342', 'symptom', 'rectal_cancer', 'ramucirumab', 'interstiti_brachytherapi', 'vegfa', 'tetraploid', 'msi', 'rx', 'fap', 'array-cgh', 'mir-92', 'irinotecan', 't4a-n2a-m0', 'adenomat_polyposi_syndrom', 'colon_cancer', 'radiofrequ_ablat', 'hereditari_nonpolyposi_type_5', 'r2', 'microrna_marker', 'mucos', 'ras-mapk', 'gardner_syndrom', 'gene', 'neoadjuv_chemo', 'iic', 'adjuv_chemo', 'doubl_contrast_barium_enema', 'mgmt', 'smoke', 'euploid', 'tingl', 'cyramza', 'monoclon_antibodi', 'vomit', 'appetit_loss', 'nausea', 'c18.4', 'mlh1', 'mir-155', 'c18.6', 'ihc_msi_marker', 'barium_enema', 'hamartomat_polyposi_syndrom', 'msh6', 'respons', 'biomark', 'd17s250', 'rs12603526', 'hereditari_nonpolyposi', 'alcohol', 'pi3k', 'rtk', 'nausea', 'blood_disord', 'lack_of_physic_exercis', 'follow-up', 'immun_checkpoint_inhibitor', 'pembrolizumab', 'transan_endoscop_microsurgeri', 'weak', 'colorect_cancer', 'rs10911251', 'polymeras_proofreading-associ_polyposi', 'iib', 'dna_msi_test_result', 'molecular_featur', 'descend_colon_cancer', 'c18.5', 't4b-n0-m0', 'hepat_arteri_infus', 'molecular_marker_test', 'rs1799977', 'predict', 'p16', '18q_ai_express', 'stereotact', 'anu_neoplasm', 'cd133', 'fever', 'ivb', 'good', 'colon_kaposi_sarcoma', 'wnt', 'e1317q', 'rs3802842', 'weak_muscl', 'tis-n0-m0', 'splenic_flexur_cancer', 'chemotherapi', 'target_therapi', 'c18.7', 'turcot_syndrom', 'mir-21', 'rs4779584', 'adenosquam_colon_carcinoma', 'pathwai', 'upsetstomach', 'gender_male', 'rs11169552', 'surviv', 'rs459552', 'rs3217810', 'intern', 'overal_surviv', 'rectal_bleed', 'braf_mutat', 't1-n0-m0', 'extern_beam', 'pms2_loss', 'blood_base', 'gardner_syndrom', 'attenu_adenomat_polyposi_coli', 'ptgs2', 't2-n0-m0', 'ploidi_statu', 'genom_instabl', 'bloodi_stool', 'progress_diseas', 'hereditari_nonpolyposi_type_8', 'nervou_system_effect', 'headach', 'stomach_pain', 'five-year_surviv', 'local_excis', 'type', 'hereditari_nonpolyposi_type_6', 'iii', 't1\xe2\x80\x93t2-n1/n1c-m0', 'therapi', 'hair_loss', 'cea', 'chemotherapi_drug', 'rs3824999', 'colon_lymphoma', 'recurr', 'ulcer_coliti', 'diseas_etiolog', 'g2', 'apoptot', 'iiic', 'ani_t_-ani_n-m1b', '0', 'high_red_meat_diet', 'juvenil_polyposi_syndrom', 'rs1800734', 'microscopi', 'dmmr', 'fit', 'r0', 'mri', 'skin_irrit', 'leukopenia', 'ng', 'system', 'desmoid_diseas', 'pole', 'ctc', 'mir-211', 'iia', 'rs12241008', 'malign', 'g13d', 'rs961253', 'ag', 'hereditari_mix_polyposi_syndrom_2', 'dpyd', 'epigenet_gene_silenc', 'f594l', 'constip', 'cologuard', 'hereditari_colon_cancer', 't4b-n1\xe2\x80\x93n2-m0', 'poor', 'obes', 'partial', 'region', 'r1', 'thrombocytopenia', 'dmmr_test', 'colon_sarcoma', 'rs174550', 'peel', 'rectum_cancer', 't1\xe2\x80\x93t2-n2b-m0', 'd2s123', 'rs4444235', 'laparoscopi', 'cin_marker', 'loss_of_balanc', 'laser_therapi', 'kra_mutat_test', 'snp', 'liver_metastasi', 'prognosi', 'rs1321311', 'ct', 'aneuploid', 'g12v', 'kra', 'rs36053993', 'msi_test', 'hereditari_nonpolyposi_type_4', 'apc', 'timp-1', 'g4', 'p53_express', 'fda_approveddrug', 'g12', 'singl_specimen_guaiac_fobt', 'combin', 'neuropathi', 'mlh1_loss', 'endocavitari', 'fungal_infect', 'hereditari_nonpolyposi_type_1', 'braf_mutat_test', 'anemia', 'cea_assai', 'colorect_neoplasm', 'polyploidi_test', 'regorafenib', 'g1', 'dna_msi_marker', 'peutz-jegh_syndrom', 'adenomat_polyposi_coli', 'rs10411210', 'epcam', 'colectomi', 'prognost', 'autosom_recess_colorect_adenomat_polyposi', 'hereditari_nonpolyposi_type_3', 'rs158634', 'colon_l-cell_glucagon-lik_peptid_produc_tumor', 'c20', 'metastat_colorect_cancer', 'xeliri', 'burn', 'hyperplast_polyposi_syndrom', 'bevacizumab', 'rectosigmoid_juction_cancer', 'european', 't2\xe2\x80\x93t3-n2a-m0', 'carbon_dioxid', 'cd24', 'tumor_msi-h_express', 'colorect_adenocarcinoma', 'ani_t-_ani_n-m1a', 'virtual_colonoscopi', 'crohn&apos;_diseas', 'tender', 'diploid', 't3\xe2\x80\x93t4a-n1/n1c-m0', 'pms2', 'muscl_pain', 'folfiri-bevacizumab', 'rectal_neoplasm', 'predict_biomark', 'braf', 'nrasmut', 'bat25', 'pet', 'rs1042522', 'complet', 'cin', 'sigmoid_colon_cancer', 'ascend_colon_cancer', 'radiat_therapi', 'krt20', 'mouth_and_throat_sore', 'bat26', 'apc_mutat', 'dre', 'colon_leiomysarcoma', 'fatigu', 'ra_mutat_test', 'c19', 'diagnosi', 'shake', 'lynch_syndrom', 'c18.9', 'tyrosin_kinas_inhibitor', 'risk_factor', 'ca_19-9', 'hmlh1', 'msh2_loss', 'rs4813802', 'colostomi', 'screen', 'v600e', 'colon_singlet_ring_adenocarcinoma', 'alter_bowel_habit', 'xelox', 'iva', 'ii', 'stabl_diseas', 'rs12309274', 'i', 'hereditari_nonpolyposi_type_7', 'lung_metastasi', 'anal_canal_carcinoma', 'fu-lv', 'prognost_biomark', 'colon_small_cell_carcinoma', 'resect', 'rs647161', 'li-fraumeni_syndrom', 'q61k', 'rs10936599', 'sexual_issu', 'rs7758229', 'hepat_flexur_cancer', 'proctectomi', 'clinic_featur', 'msh2', 'dna_mismatch-repair', 'c18.2', 'mrt', 'cryosurgeri', 'pik3ca', 'hereditari_mix_polyposi_syndrom_1', 'oligodontia-colorect_cancer_syndrom', 'sept9_methyl', 'fit', 'lonsurf', 'exercis', 'pain', 'east_asian', 'colonoscopi', 'adenoma', 'tgf-\xce\xb2', 'g12d', 'rs704017', 'surgeri', 'faecal_m2-pk', 'polyploidi_test_result', 'msh6_loss', 'inherit_genet_disord', 'lgr5', 'kra_mutat', 'submucos_invasivecolon_adenocarcinoma', 'bmi', 'r_classif', 'rs9929218', 'sigmoidoscopi', 'stem_cell', 'mutyh-associ_polyposi', '5-fu', 'vegf', 't3\xe2\x80\x93t4a-n2b-m0', 'nonpolyposi_syndrom', 't1-n2a-m0', 'hyperthermia', 'high_fat_intak', 'type_of_care', 'g3', 'popul_base_snp', 'alk', 'mir-92a', 'cd166', 'anal_gland_neoplasm', 't4a-n0-m0', 'metastasi', 'd5s346', 'rs10849432', 'blister', 'rs61764370', 'rs1801155', 'plod1', 'c18.3', 'optic_colonoscopi', 'mir-31', 'rs16892766', 'iv', 'rectosigmoid_cancer', 'panitumumab', 't3-n0-m0', 'mir-17', 'gx', 'fish', 'cognit_dysfunct', 'egfr', 'rs1801166', 'prognost_factor', 'bladder_irrit', 'acut_myelocyt_leukemia', 'tym', 'uicc_stage', 'folfox', 'lipomat_hemangiopericytoma', 'rs6691170', 'aldh1', 'tumor_bud', 'mutyh', 'mss', 'grade', 'attenu_famili_adenomat_polyposi', 'colon_adenocarcinoma', 'high_sensit_faecal_occult_blood_test', 'samson_gardner_syndrom', 'colon_mucin_adenocarcinoma', 'pmmr', 'tp53', 'g463v', 'capsul_colonoscopi', 'colon_squamou_cell_carcinoma', 'rectal_irrit', 'c18.1', 'hra', 'ceacam5', 'neodymium:yttrium-aluminum-garnet', 'cetuximab', 'folfiri', 'rs6983267', 'msi-l', 'c18']
    
    #NOTE only Fixed_Multi-Tag model has this update!
    vocab_list = [any2unicode(element) for element in vocab_list]
    dim = 200
    win = 8
    neg = 10
   
    kwargs = {"sent": contents, "vocab": vocab_list, 
              "dim": dim, "win": win, "min_cnt": 2, "neg": neg, "iter":20 , "tag_doc" :contents
              }
    Dis2Vec(**kwargs).run_Dis2Vec()
    

if __name__ == "__main__":
    main()


