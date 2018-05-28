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
        out_folder = './newoutput/'
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        if model.sample == 0:
            model.save(out_folder + 'mod')
        else:
            model.save(out_folder + 'mod')


class TaggedPubMed(object):

    def __init__(self, source):
    
        self.source = source
        self.bigram = Phraser.load('./big_phrases')
        self.trigram = Phraser.load('./trigram_phrases')

    def __iter__(self):
        with utils.smart_open(self.source) as fin:
            for item_no, line in enumerate(fin):

                #Remove full-stops ,brackets and grammatic commas
                line = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )","",line)

                label = " ".join(line.split(" ")[:10])
                

                #Calculate trigram before lowering the case
                doc = filter(lambda word: word not in stopwords.words('english'),line.split(" ") )
                doc = self.trigram[self.bigram[doc]]

                doc = [element.lower() for element in doc]
                
                yield TaggedDocument(doc, [label])            

def main():

    

    contents = TaggedPubMed('big_home_test.txt')


    vocab_list = ['rs10795668', 'mir-135a', 'lynch_syndrome_i', 'biopsy', 'disease', 'c18.8', 'folfiri-cetuximab', 'rs4939827', 'iiib', 'colon_carcinoma', 'outcome', 'transverse_colon_cancer', 'therapy_resistance', 'ctnnb1', 'iiia', 'rs1035209', 'family_history', 'relapse_free_survival', 'p14', 'anastomosis', 'cowden_syndrome', 'oxaliplatin', 'msi-h', 'bleeding', 'dna_image_cytometry', 'capox', 'weight_loss', 'icd', 'endorectal_mri', 'aflibercept', 'argon', 'egf', 'immunotherapy', 'physical_activity', 'rs4925386', 'c18.0', 'side_effects', 'disease_subtypes', 'angiogenesis_inhibitors', 'cloacogenic_carcinoma', 'colonic_neoplasms', 'cd29', 'dysplasia_in_inflammatory_bowel_disease', 'serrated_polyposis', 'epcam', 'intestinal_polyposis', 'rs1800469', 'cd44', 'mir-135b', 'g1n1317', 'rs34612342', 'symptoms', 'rectal_cancer', 'ramucirumab', 'interstitial_brachytherapy', 'vegfa', 'tetraploid', 'msi', 'rx', 'fap', 'array-cgh', 'mir-92', 'irinotecan', 't4a-n2a-m0', 'adenomatous_polyposis_syndromes', 'colon_cancer', 'radiofrequency_ablation', 'hereditary_nonpolyposis_type_5', 'r2', 'microrna_markers', 'mucositis', 'ras-mapk', 'gardner_syndrome', 'genes', 'neoadjuvant_chemo', 'iic', 'adjuvant_chemo', 'double_contrast_barium_enema', 'mgmt', 'smoking', 'euploid', 'tingling', 'cyramza', 'monoclonal_antibodies', 'vomiting', 'appetite_loss', 'nausea', 'c18.4', 'mlh1', 'mir-155', 'c18.6', 'ihc_msi_markers', 'barium_enema', 'hamartomatous_polyposis_syndromes', 'msh6', 'response', 'biomarkers', 'd17s250', 'rs12603526', 'hereditary_nonpolyposis', 'alcohol', 'pi3k', 'rtk', 'nausea', 'blood_disorders', 'lack_of_physical_exercise', 'follow-up', 'immune_checkpoint_inhibitors', 'pembrolizumab', 'transanal_endoscopic_microsurgery', 'weakness', 'colorectal_cancer', 'rs10911251', 'polymerase_proofreading-associated_polyposis', 'iib', 'dna_msi_test_results', 'molecular_features', 'descending__colon_cancer', 'c18.5', 't4b-n0-m0', 'hepatic_artery_infusion', 'molecular_marker_testing', 'rs1799977', 'predictive', 'p16', '18q_ai_expression', 'stereotactic', 'anus_neoplasms', 'cd133', 'fever', 'ivb', 'good', 'colon_kaposi_sarcoma', 'wnt', 'e1317q', 'rs3802842', 'weak_muscle', 'tis-n0-m0', 'splenic_flexure_cancer', 'chemotherapy', 'targeted_therapy', 'c18.7', 'turcot_syndrome', 'mir-21', 'rs4779584', 'adenosquamous_colon_carcinoma', 'pathways', 'upsetstomach', 'gender_male', 'rs11169552', 'survival', 'rs459552', 'rs3217810', 'internal', 'overall_survival', 'rectal_bleeding', 'braf_mutation', 't1-n0-m0', 'external_beam', 'pms2_loss', 'blood_based', 'gardner_syndrome', 'attenuated_adenomatous_polyposis_coli', 'ptgs2', 't2-n0-m0', 'ploidy_status', 'genomic_instability', 'bloody_stools', 'progressive_disease', 'hereditary_nonpolyposis_type_8', 'nervous_system_effects', 'headaches', 'stomach_pain', 'five-year_survival', 'local_excision', 'types', 'hereditary_nonpolyposis_type_6', 'iii', 't1\xe2\x80\x93t2-n1/n1c-m0', 'therapy', 'hair_loss', 'cea', 'chemotherapy_drugs', 'rs3824999', 'colon_lymphoma', 'recurrence', 'ulcerative_colitis', 'disease_etiology', 'g2', 'apoptotic', 'iiic', 'any_t_-any_n-m1b', '0', 'high_red_meat_diet', 'juvenile_polyposis_syndrome', 'rs1800734', 'microscopy', 'dmmr', 'fitness', 'r0', 'mri', 'skin_irritation', 'leukopenia', 'ngs', 'systemic', 'desmoid_disease', 'pole', 'ctc', 'mir-211', 'iia', 'rs12241008', 'malignancy', 'g13d', 'rs961253', 'age', 'hereditary_mixed_polyposis_syndrome_2', 'dpyd', 'epigenetic_gene_silencing', 'f594l', 'constipation', 'cologuard', 'hereditary_colon_cancer', 't4b-n1\xe2\x80\x93n2-m0', 'poor', 'obesity', 'partial', 'regional', 'r1', 'thrombocytopenia', 'dmmr_test', 'colon_sarcoma', 'rs174550', 'peeling', 'rectum_cancer', 't1\xe2\x80\x93t2-n2b-m0', 'd2s123', 'rs4444235', 'laparoscopy', 'cin_markers', 'loss_of_balance', 'laser_therapy', 'kras_mutational_testing', 'snps', 'liver_metastasis', 'prognosis', 'rs1321311', 'ct', 'aneuploid', 'g12v', 'kras', 'rs36053993', 'msi_test', 'hereditary_nonpolyposis_type_4', 'apc', 'timp-1', 'g4', 'p53_expression', 'fda_approveddrugs', 'g12s', 'single_specimen_guaiac_fobt', 'combinations', 'neuropathy', 'mlh1_loss', 'endocavitary', 'fungal_infection', 'hereditary_nonpolyposis_type_1', 'braf_mutation_test', 'anemia', 'cea_assay', 'colorectal_neoplasms', 'polyploidy_test', 'regorafenib', 'g1', 'dna_msi_markers', 'peutz-jeghers_syndrome', 'adenomatous_polyposis_coli', 'rs10411210', 'epcam', 'colectomy', 'prognostic', 'autosomal_recessive_colorectal_adenomatous_polyposis', 'hereditary_nonpolyposis_type_3', 'rs158634', 'colonic_l-cell_glucagon-like_peptide_producing_tumor', 'c20', 'metastatic_colorectal_cancer', 'xeliri', 'burning', 'hyperplastic_polyposis_syndrome', 'bevacizumab', 'rectosigmoid_juction_cancer', 'european', 't2\xe2\x80\x93t3-n2a-m0', 'carbon_dioxide', 'cd24', 'tumor_msi-h_expression', 'colorectal_adenocarcinoma', 'any_t-_any_n-m1a', 'virtual_colonoscopy', 'crohn&apos;s_disease', 'tenderness', 'diploid', 't3\xe2\x80\x93t4a-n1/n1c-m0', 'pms2', 'muscle_pain', 'folfiri-bevacizumab', 'rectal_neoplasms', 'predictive_biomarker', 'braf', 'nrasmutation', 'bat25', 'pet', 'rs1042522', 'complete', 'cin', 'sigmoid_colon_cancer', 'ascending_colon_cancer', 'radiation_therapy', 'krt20', 'mouth_and_throat_sores', 'bat26', 'apc_mutations', 'dre', 'colon_leiomysarcoma', 'fatigue', 'ras_mutation_test', 'c19', 'diagnosis', 'shaking', 'lynch_syndrome', 'c18.9', 'tyrosine_kinase_inhibitors', 'risk_factors', 'ca_19-9', 'hmlh1', 'msh2_loss', 'rs4813802', 'colostomy', 'screening', 'v600e', 'colon_singlet_ring_adenocarcinoma', 'altered_bowel_habits', 'xelox', 'iva', 'ii', 'stable_disease', 'rs12309274', 'i', 'hereditary_nonpolyposis_type_7', 'lung_metastasis', 'anal_canal_carcinoma', 'fu-lv', 'prognostic_biomarker', 'colon_small_cell_carcinoma', 'resectability', 'rs647161', 'li-fraumeni_syndrome', 'q61k', 'rs10936599', 'sexual_issues', 'rs7758229', 'hepatic_flexure_cancer', 'proctectomy', 'clinical_features', 'msh2', 'dna_mismatch-repair', 'c18.2', 'mrt', 'cryosurgery', 'pik3ca', 'hereditary_mixed_polyposis_syndrome_1', 'oligodontia-colorectal_cancer_syndrome', 'sept9_methylation', 'fit', 'lonsurf', 'exercise', 'pain', 'east_asian', 'colonoscopy', 'adenomas', 'tgf-\xce\xb2', 'g12d', 'rs704017', 'surgery', 'faecal_m2-pk', 'polyploidy_test_results', 'msh6_loss', 'inherited_genetic_disorders', 'lgr5', 'kras_mutation', 'submucosal_invasivecolon_adenocarcinoma', 'bmi', 'r_classification', 'rs9929218', 'sigmoidoscopy', 'stem_cell', 'mutyh-associated_polyposis', '5-fu', 'vegf', 't3\xe2\x80\x93t4a-n2b-m0', 'nonpolyposis_syndrome', 't1-n2a-m0', 'hyperthermia', 'high_fat_intake', 'type_of_care', 'g3', 'population_based_snp', 'alk', 'mir-92a', 'cd166', 'anal_gland_neoplasms', 't4a-n0-m0', 'metastasis', 'd5s346', 'rs10849432', 'blistering', 'rs61764370', 'rs1801155', 'plod1', 'c18.3', 'optical_colonoscopy', 'mir-31', 'rs16892766', 'iv', 'rectosigmoid_cancer', 'panitumumab', 't3-n0-m0', 'mir-17', 'gx', 'fish', 'cognitive_dysfunction', 'egfr', 'rs1801166', 'prognostic_factors', 'bladder_irritation', 'acute_myelocytic_leukemia', 'tyms', 'uicc_staging', 'folfox', 'lipomatous_hemangiopericytoma', 'rs6691170', 'aldh1', 'tumor_budding', 'mutyh', 'mss', 'grade', 'attenuated_familial_adenomatous_polyposis', 'colon_adenocarcinoma', 'high_sensitivity_faecal_occult_blood_test', 'samson_gardner_syndrome', 'colon_mucinous_adenocarcinoma', 'pmmr', 'tp53', 'g463v', 'capsule_colonoscopy', 'colon_squamous_cell_carcinoma', 'rectal_irritation', 'c18.1', 'hras', 'ceacam5', 'neodymium:yttrium-aluminum-garnet', 'cetuximab', 'folfiri', 'rs6983267', 'msi-l', 'c18']


    dim = 200
    win = 8
    neg = 5
   
    kwargs = {"sent": contents, "vocab": vocab_list, 
              "dim": dim, "win": win, "min_cnt": 2, "neg": neg, "iter":20 , "tag_doc" :contents
              }
    Dis2Vec(**kwargs).run_Dis2Vec()
    

if __name__ == "__main__":
    main()


