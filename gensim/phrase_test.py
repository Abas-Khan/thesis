from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models.word2vec import LineSentence


import sys
import codecs

import unicodedata
from gensim.utils import any2utf8

#TODO i changed to million_records after having already trained on big_test file, maybe i should train again
contents = LineSentence("million_records.txt")


common_terms = ["of", "with", "without", "and", "or", "the", "a"]
my_bigrams = [['colon', 'carcinoma'], ['therapy', 'resistance'], ['family', 'history'], ['Cowden', 'syndrome'], ['weight', 'loss'], ['Endorectal', 'MRI'], ['physical', 'activity'], ['side', 'effects'], ['disease', 'subtypes'], ['angiogenesis', 'inhibitors'], ['cloacogenic', 'carcinoma'], ['colonic', 'neoplasms'], ['serrated', 'polyposis'], ['intestinal', 'polyposis'], ['rectal', 'cancer'], ['interstitial', 'brachytherapy'], ['colon', 'cancer'], ['radiofrequency', 'ablation'], ['microRNA', 'markers'], ['gardner', 'syndrome'], ['neoadjuvant', 'chemo'], ['adjuvant', 'chemo'], ['monoclonal', 'antibodies'], ['appetite', 'loss'], ['barium', 'enema'], ['hereditary', 'nonpolyposis'], ['blood', 'disorders'], ['colorectal', 'cancer'], ['molecular', 'features'], ['anus', 'neoplasms'], ['weak', 'muscle'], ['targeted', 'therapy'], ['Turcot', 'syndrome'], ['gender', 'male'], ['overall', 'survival'], ['rectal', 'bleeding'], ['BRAF', 'mutation'], ['external', 'beam'], ['PMS2', 'loss'], ['blood', 'based'], ['Gardner', 'syndrome'], ['ploidy', 'status'], ['genomic', 'instability'], ['bloody', 'stools'], ['progressive', 'disease'], ['stomach', 'pain'], ['five-year', 'survival'], ['local', 'excision'], ['hair', 'loss'], ['chemotherapy', 'drugs'], ['colon', 'lymphoma'], ['ulcerative', 'colitis'], ['disease', 'etiology'], ['skin', 'irritation'], ['desmoid', 'disease'], ['dMMR', 'test'], ['colon', 'sarcoma'], ['rectum', 'cancer'], ['CIN', 'markers'], ['laser', 'therapy'], ['liver', 'metastasis'], ['MSI', 'test'], ['p53', 'expression'], ['FDA', 'approveddrugs'], ['MLH1', 'loss'], ['fungal', 'infection'], ['CEA', 'assay'], ['colorectal', 'neoplasms'], ['polyploidy', 'test'], ['Peutz-Jeghers', 'syndrome'], ['carbon', 'dioxide'], ['colorectal', 'adenocarcinoma'], ['virtual', 'colonoscopy'], ['Crohn&apos;s', 'disease'], ['muscle', 'pain'], ['rectal', 'neoplasms'], ['predictive', 'biomarker'], ['radiation', 'therapy'], ['APC', 'mutations'], ['colon', 'leiomysarcoma'], ['Lynch', 'syndrome'], ['risk', 'factors'], ['CA', '19-9'], ['MSH2', 'loss'], ['stable', 'disease'], ['lung', 'metastasis'], ['prognostic', 'biomarker'], ['Li-Fraumeni', 'syndrome'], ['sexual', 'issues'], ['clinical', 'features'], ['DNA', 'mismatch-repair'], ['SEPT9', 'methylation'], ['east', 'asian'], ['Faecal', 'M2-PK'], ['MSH6', 'loss'], ['KRAS', 'mutation'], ['R', 'classification'], ['stem', 'cell'], ['MUTYH-associated', 'polyposis'], ['nonpolyposis', 'syndrome'], ['optical', 'colonoscopy'], ['rectosigmoid', 'cancer'], ['cognitive', 'dysfunction'], ['prognostic', 'factors'], ['bladder', 'irritation'], ['UICC', 'staging'], ['lipomatous', 'hemangiopericytoma'], ['tumor', 'budding'], ['colon', 'adenocarcinoma'], ['capsule', 'colonoscopy'], ['rectal', 'irritation']]
my_trigrams = [['Lynch_syndrome', 'I'], ['transverse_colon', 'cancer'], ['relapse_free', 'survival'], ['DNA_Image', 'Cytometry'], ['adenomatous_polyposis', 'syndromes'], ['IHC_MSI', 'markers'], ['hamartomatous_polyposis', 'syndromes'], ['immune_checkpoint', 'inhibitors'], ['transanal_endoscopic', 'microsurgery'], ['polymerase_proofreading-associated', 'polyposis'], ['hepatic_artery', 'infusion'], ['molecular_marker', 'testing'], ['18q_AI', 'expression'], ['colon_Kaposi', 'sarcoma'], ['splenic_flexure', 'cancer'], ['adenosquamous_colon', 'carcinoma'], ['nervous_system', 'effects'], ['Juvenile_polyposis', 'syndrome'], ['Epigenetic_gene', 'silencing'], ['hereditary_colon', 'cancer'], ['loss_of', 'balance'], ['KRAS_mutational', 'testing'], ['BRAF_mutation', 'test'], ['DNA_MSI', 'markers'], ['adenomatous_polyposis', 'coli'], ['metastatic_colorectal', 'cancer'], ['Hyperplastic_Polyposis', 'Syndrome'], ['rectosigmoid_juction', 'cancer'], ['tumor_MSI-H', 'expression'], ['sigmoid_colon', 'cancer'], ['ascending_colon', 'cancer'], ['RAS_mutation', 'test'], ['tyrosine_kinase', 'inhibitors'], ['altered_bowel', 'habits'], ['anal_canal', 'carcinoma'], ['hepatic_flexure', 'cancer'], ['oligodontia-colorectal_cancer', 'syndrome'], ['polyploidy_test', 'results'], ['inherited_genetic', 'disorders'], ['submucosal_invasivecolon', 'adenocarcinoma'], ['high_fat', 'intake'], ['type_of', 'care'], ['population_based', 'SNP'], ['anal_gland', 'neoplasms'], ['acute_myelocytic', 'leukemia'], ['Samson_Gardner', 'syndrome'], ['colon_mucinous', 'adenocarcinoma']]

phrases = Phrases(contents,threshold=0.25,scoring="npmi",custom_bigrams=my_bigrams)
bigram = Phraser(phrases)
tri_phase = Phrases(bigram[contents],custom_bigrams=my_trigrams,threshold=0.25,scoring="npmi")
trigram = Phraser(tri_phase)


sent = [u'red', u'shift', u'square', u'pants',u'bit', u'parts',u'transverse',u'colon',u'cancer',u'trans',u'atlantic',u'ocean']
print(bigram[sent])

print trigram[bigram[sent]]

#print item
