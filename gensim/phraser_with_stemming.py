from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
from gensim.utils import keep_vocab_item, call_on_class_only
from gensim import utils, matutils
from gensim.parsing.preprocessing import stem_text
import itertools
from nltk.corpus import stopwords
import re
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.parsing.preprocessing import remove_stopwords
from word2number import w2n


def fix_age(matchobj):
    #print matchobj.group(0)
    return str(w2n.word_to_num(matchobj.group(0)))+" "  

class LineIterator(object):
    """Simple format: one sentence = one line; words already preprocessed and separated by whitespace.
    """

    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):

        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):

            with utils.smart_open(self.source) as fin:
                for idx,line in enumerate(itertools.islice(fin, self.limit)):
                    print idx,
                    print "\r",
                    
                    #line = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|'","",line)

                    #Split words if they get stuffed together
                    #line = re.sub( r"([A-Z][A-Z]*)", r" \1", line)

                    #line = re.sub(r"\.(?=[^\d])|(?<=\w|\d)\(|(?<=\w|\d)\)"," ",line)
                    #line = re.sub(r"(?<=\w[^\d])\.|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|'","",line)
                    
                    #NOTE split() will take care of extra spaces
                    #line = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:|\?|;|>|<|=|%|\+|-"," ",line)
                    
                    
                    
                    #NOTE convert ages to numeric
                    '''
                    try:
                        line = re.sub(r"([a-z]+ (?=year-old))",fix_age,line)
                    except:
                        pass
                    
                    '''  
                    #document = line.split("<sep>")[1]
                    line = line.split("<sep>")[1]
                    line = remove_stopwords(line)
                    #doc = filter(lambda word: word not in stopwords.words('english'),line.split(" ") )
                    line = stem_text(line)
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i: i + self.max_sentence_length]
                        i += self.max_sentence_length



#'dysplasia_inflammatori_bowel_diseas'


# Stemmed n-grams for stemmed corpus
my_bigrams = [['colon', 'carcinoma'], ['therapi', 'resist'], ['famili', 'histori'], ['cowden', 'syndrom'], ['weight', 'loss'], ['endorect', 'mri'], ['physic', 'activ'], ['side', 'effect'], ['diseas', 'subtyp'], ['angiogenesi', 'inhibitor'], ['cloacogen', 'carcinoma'], ['colon', 'neoplasm'],['serrat', 'polyposi'], ['intestin', 'polyposi'], ['rectal', 'cancer'], ['interstiti', 'brachytherapi'],['colon', 'cancer'], ['radiofrequ', 'ablat'], ['microrna', 'marker'], ['gardner', 'syndrom'], ['neoadjuv', 'chemo'], ['adjuv', 'chemo'], ['monoclon', 'antibodi'], ['appetit', 'loss'], ['barium', 'enema'], ['hereditari', 'nonpolyposi'], ['blood', 'disord'], ['colorect', 'cancer'], ['molecular', 'featur'], ['anu', 'neoplasm'], ['weak', 'muscl'], ['target', 'therapi'], ['turcot', 'syndrom'], ['gender', 'male'], ['overal', 'surviv'], ['rectal', 'bleed'],['braf', 'mutat'], ['extern', 'beam'], ['pms2', 'loss'], ['blood', 'base'], ['gardner', 'syndrom'], ['ploidi', 'statu'], ['genom', 'instabl'], ['bloodi', 'stool'], ['progress', 'diseas'], ['stomach', 'pain'],['five-year', 'surviv'], ['local', 'excis'], ['hair', 'loss'], ['chemotherapi', 'drug'], ['colon', 'lymphoma'],['ulcer', 'coliti'], ['diseas', 'etiolog'], ['skin', 'irrit'], ['desmoid', 'diseas'], ['dmmr', 'test'],['colon', 'sarcoma'], ['rectum', 'cancer'], ['cin', 'marker'], ['laser', 'therapi'], ['liver', 'metastasi'],['msi', 'test'], ['p53', 'express'], ['fda', 'approveddrug'], ['mlh1', 'loss'], ['fungal', 'infect'], ['cea', 'assai'], ['colorect', 'neoplasm'], ['polyploidi', 'test'], ['peutz-jegh', 'syndrom'], ['carbon', 'dioxid'], ['colorect', 'adenocarcinoma'], ['virtual', 'colonoscopi'], ['crohn&apos;', 'diseas'], ['muscl', 'pain'], ['rectal', 'neoplasm'], ['predict', 'biomark'], ['radiat', 'therapi'], ['apc', 'mutat'],['colon', 'leiomysarcoma'], ['lynch', 'syndrom'], ['risk', 'factor'], ['ca', '19-9'], ['msh2', 'loss'],['stabl', 'diseas'], ['lung', 'metastasi'], ['prognost', 'biomark'], ['li-fraumeni', 'syndrom'], ['sexual', 'issu'], ['clinic', 'featur'], ['dna', 'mismatch-repair'], ['sept9', 'methyl'], ['east', 'asian'], ['faecal', 'm2-pk'], ['msh6', 'loss'], ['kra', 'mutat'], ['r', 'classif'], ['stem', 'cell'], ['mutyh-associ', 'polyposi'], ['nonpolyposi', 'syndrom'], ['optic', 'colonoscopi'], ['rectosigmoid', 'cancer'], ['cognit', 'dysfunct'], ['prognost', 'factor'], ['bladder', 'irrit'], ['uicc', 'stage'], ['lipomat', 'hemangiopericytoma'], ['tumor', 'bud'], ['colon', 'adenocarcinoma'], ['capsul', 'colonoscopi'], ['rectal', 'irrit'],['relaps','free'],['dna','imag'],['adenoma','polyposi'],['ihc','msi'],['hamartomat','polyposi'],['immun-','checkpoint'],['transan','endoscop'],['polymeras','proofreading-associ'],['hepat','arteri'],['molecular','marker'],['18q','ai'],['colon','kaposi'],['splenic','flexur'],['adenosquam','colon'],['nervou','system'],['juvenil','polyposi'],['epigenet','gene'],['loss','of'],['kra','mutat'],['braf','mutat'],['dna','msi'],['adenomat','polyposi'],['hyperplast','polyposi'],['rectosigmoid','juction'],['tumor','msi-h'],['ra','mutat'],['tyrosin','kinas'],['alter','bowel'],['anal','canal'],['hepat','flexur'],['oligodontia-colorect','cancer'],['polyploidi','test'],['inherit','genet'],['submucos','invasivecolon'],['high','fat'],['type','of'],['popul','base'],['anal','gland'],['acut','myelocyt'],['samson','gardner'],['colon','mucin']]
my_trigrams = [['lynch_syndrom', 'i'], ['transvers', 'colon_cancer'], ['relaps_free', 'surviv'], ['dna_imag', 'cytometri'],['adenomat_polyposi', 'syndrom'], ['ihc_msi', 'marker'], ['hamartomat_polyposi', 'syndrom'], ['immun_checkpoint', 'inhibitor'], ['transan_endoscop', 'microsurgeri'],['polymeras_proofreading_associ', 'polyposi'], ['descend', 'colon_cancer'],['hepat_arteri', 'infus'], ['molecular_marker', 'test'], ['18q_ai', 'express'],['colon_kaposi', 'sarcoma'], ['splenic_flexur', 'cancer'], ['adenosquam', 'colon_carcinoma'], ['nervou_system', 'effect'], ['juvenil_polyposi', 'syndrom'], ['epigenet_gene', 'silenc'], ['hereditari', 'colon_cancer'], ['loss_of', 'balanc'], ['kra_mutat', 'test'], ['braf_mutat', 'test'],['dna_msi', 'marker'], ['adenomat_polyposi', 'coli'], ['metastat', 'colorect_cancer'],['hyperplast_polyposi', 'syndrom'], ['rectosigmoid_juction', 'cancer'], ['tumor_msi_h', 'express'],['sigmoid', 'colon_cancer'], ['ascend', 'colon_cancer'], ['ra_mutat', 'test'],['tyrosin_kinas', 'inhibitor'], ['alter_bowel', 'habit'], ['anal_canal', 'carcinoma'], ['hepat_flexur', 'cancer'], ['oligodontia_colorect_cancer', 'syndrom'], ['polyploidi_test', 'result'], ['inherit_genet', 'disord'], ['submucos_invasivecolon', 'adenocarcinoma'], ['high_fat', 'intak'],['type_of', 'care'], ['popul_base', 'snp'], ['anal_gland', 'neoplasm'],['acut_myelocyt', 'leukemia'], ['samson_gardner', 'syndrom'], ['colon_mucin', 'adenocarcinoma']]

#contents = LineIterator("big_home_test.txt")



uni_gram_ignore_list = ['rs10795668', 'mir-135a', 'biopsi', 'diseas', 'c18.8', 'folfiri-cetuximab', 'rs4939827', 'iiib', 'outcom', 'ctnnb1', 'iiia', 'rs1035209', 'p14', 'anastomosi', 'oxaliplatin', 'msi-h', 'bleed', 'capox', 'icd', 'aflibercept', 'argon', 'egf', 'immunotherapi', 'rs4925386', 'c18.0', 'cd29', 'epcam', 'rs1800469', 'cd44', 'mir-135b', 'g1n1317', 'rs34612342', 'symptom', 'ramucirumab', 'vegfa', 'tetraploid', 'msi', 'rx', 'fap', 'array-cgh', 'mir-92', 'irinotecan', 't4a-n2a-m0', 'r2', 'mucos', 'ras-mapk', 'gene', 'iic', 'mgmt', 'smoke', 'euploid', 'tingl', 'cyramza', 'vomit', 'nausea', 'c18.4', 'mlh1', 'mir-155', 'c18.6', 'msh6', 'respons', 'biomark', 'd17s250', 'rs12603526', 'alcohol', 'pi3k', 'rtk', 'nausea', 'follow-up', 'pembrolizumab', 'weak', 'rs10911251', 'iib', 'c18.5', 't4b-n0-m0', 'rs1799977', 'predict', 'p16', 'stereotact', 'cd133', 'fever', 'ivb', 'good', 'wnt', 'e1317q', 'rs3802842', 'tis-n0-m0', 'chemotherapi', 'c18.7', 'mir-21', 'rs4779584', 'pathwai', 'rs11169552', 'surviv', 'rs459552', 'rs3217810', 'intern', 't1-n0-m0', 'ptgs2', 't2-n0-m0', 'headach', 'type', 'iii', 't1\xe2\x80\x93t2-n1/n1c-m0', 'therapi', 'cea', 'rs3824999', 'recurr', 'g2', 'apoptot', 'iiic', '0', 'rs1800734', 'microscopi', 'dmmr', 'fit', 'r0', 'mri', 'leukopenia', 'ng', 'system', 'pole', 'ctc', 'mir-211', 'iia', 'rs12241008', 'malign', 'g13d', 'rs961253', 'ag','dpyd', 'f594l', 'constip', 'cologuard', 't4b-n1\xe2\x80\x93n2-m0', 'poor', 'obes', 'partial', 'region', 'r1', 'thrombocytopenia', 'rs174550', 'peel','t1\xe2\x80\x93t2-n2b-m0', 'd2s123', 'rs4444235', 'laparoscopi', 'snp', 'prognosi', 'rs1321311', 'ct', 'aneuploid', 'g12v', 'kra', 'rs36053993', 'apc', 'timp-1', 'g4', 'g12', 'combin', 'neuropathi', 'endocavitari', 'anemia', 'regorafenib', 'g1', 'rs10411210', 'epcam', 'colectomi', 'prognost', 'rs158634', 'c20', 'xeliri', 'burn', 'bevacizumab', 'european', 't2\xe2\x80\x93t3-n2a-m0', 'cd24', 'tender', 'diploid', 't3\xe2\x80\x93t4a-n1/n1c-m0', 'pms2','folfiri-bevacizumab', 'braf', 'bat25', 'pet', 'rs1042522', 'complet', 'cin', 'krt20', 'bat26', 'dre', 'fatigu', 'c19', 'diagnosi', 'shake', 'c18.9', 'hmlh1', 'rs4813802', 'colostomi', 'screen', 'v600e', 'xelox', 'iva', 'ii', 'rs12309274', 'i', 'fu-lv', 'resect', 'rs647161', 'q61k', 'rs10936599', 'rs7758229', 'proctectomi', 'msh2', 'c18.2', 'mrt', 'cryosurgeri', 'pik3ca', 'fit', 'lonsurf', 'exercis', 'pain', 'colonoscopi', 'adenoma', 'tgf-\xce\xb2', 'g12d', 'rs704017', 'surgeri', 'lgr5', 'bmi', 'rs9929218', 'sigmoidoscopi', '5-fu', 'vegf', 't3\xe2\x80\x93t4a-n2b-m0', 't1-n2a-m0', 'hyperthermia', 'g3', 'alk', 'mir-92a', 'cd166', 't4a-n0-m0', 'metastasi', 'd5s346', 'rs10849432', 'blister', 'rs61764370', 'rs1801155', 'plod1', 'c18.3', 'mir-31', 'rs16892766', 'iv', 'panitumumab', 't3-n0-m0', 'mir-17', 'gx', 'fish', 'egfr', 'rs1801166', 'tym', 'folfox', 'rs6691170', 'aldh1', 'mutyh', 'mss', 'grade', 'pmmr', 'tp53', 'g463v', 'c18.1', 'hra', 'ceacam5', 'neodymium:yttrium-aluminum-garnet', 'cetuximab', 'folfiri', 'rs6983267', 'msi-l', 'c18']
real_bigrams = []

for item in my_bigrams:
    real_bigrams.append("_".join(item))


#NOTE I used tr -d to remove ' from the file
contents = LineIterator("crc_bash_processed.txt")

phrases = Phrases(contents,threshold=0.25, scoring='npmi',custom_bigrams=my_bigrams,ignore_list=uni_gram_ignore_list)
bigram = Phraser(phrases)
print "bigrams calculated"
tri_phase = Phrases(bigram[contents],threshold=0.25,scoring='npmi',custom_bigrams=my_trigrams,ignore_list=real_bigrams)
trigram = Phraser(tri_phase)
print "trirams calculated"

#NOTE fixed min_count + trigram bugs
bigram.save('./crc_fixed_bigram')
print "ngrams saved"
trigram.save('./crc_fixed_trigram')
print "ngrams saved"
