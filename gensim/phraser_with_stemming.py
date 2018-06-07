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
                    line = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:|\?|\;"," ",line)
                    
                    
                    
                    #NOTE convert ages to numeric
                    
                    try:
                        line = re.sub(r"([a-z]+ (?=year-old))",fix_age,line)
                    except:
                        pass
                      

                    line = remove_stopwords(line)
                    #doc = filter(lambda word: word not in stopwords.words('english'),line.split(" ") )
                    line = stem_text(line)
                    line = utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i: i + self.max_sentence_length]
                        i += self.max_sentence_length


# Stemmed n-grams for stemmed corpus
my_bigrams = [['colon', 'carcinoma'], ['therapi', 'resist'], ['famili', 'histori'], ['cowden', 'syndrom'], ['weight', 'loss'], ['endorect', 'mri'], ['physic', 'activ'], ['side', 'effect'], ['diseas', 'subtyp'], ['angiogenesi', 'inhibitor'], ['cloacogen', 'carcinoma'], ['colon', 'neoplasm'], ['serrat', 'polyposi'], ['intestin', 'polyposi'], ['rectal', 'cancer'], ['interstiti', 'brachytherapi'], ['colon', 'cancer'], ['radiofrequ', 'ablat'], ['microrna', 'marker'], ['gardner', 'syndrom'], ['neoadjuv', 'chemo'], ['adjuv', 'chemo'], ['monoclon', 'antibodi'], ['appetit', 'loss'], ['barium', 'enema'], ['hereditari', 'nonpolyposi'], ['blood', 'disord'], ['colorect', 'cancer'], ['molecular', 'featur'], ['anu', 'neoplasm'], ['weak', 'muscl'], ['target', 'therapi'], ['turcot', 'syndrom'], ['gender', 'male'], ['overal', 'surviv'], ['rectal', 'bleed'], ['braf', 'mutat'], ['extern', 'beam'], ['pms2', 'loss'], ['blood', 'base'], ['gardner', 'syndrom'], ['ploidi', 'statu'], ['genom', 'instabl'], ['bloodi', 'stool'], ['progress', 'diseas'], ['stomach', 'pain'], ['five-year', 'surviv'], ['local', 'excis'], ['hair', 'loss'], ['chemotherapi', 'drug'], ['colon', 'lymphoma'], ['ulcer', 'coliti'], ['diseas', 'etiolog'], ['skin', 'irrit'], ['desmoid', 'diseas'], ['dmmr', 'test'], ['colon', 'sarcoma'], ['rectum', 'cancer'], ['cin', 'marker'], ['laser', 'therapi'], ['liver', 'metastasi'], ['msi', 'test'], ['p53', 'express'], ['fda', 'approveddrug'], ['mlh1', 'loss'], ['fungal', 'infect'], ['cea', 'assai'], ['colorect', 'neoplasm'], ['polyploidi', 'test'], ['peutz-jegh', 'syndrom'], ['carbon', 'dioxid'], ['colorect', 'adenocarcinoma'], ['virtual', 'colonoscopi'], ['crohn&apos;', 'diseas'], ['muscl', 'pain'], ['rectal', 'neoplasm'], ['predict', 'biomark'], ['radiat', 'therapi'], ['apc', 'mutat'], ['colon', 'leiomysarcoma'], ['lynch', 'syndrom'], ['risk', 'factor'], ['ca', '19-9'], ['msh2', 'loss'], ['stabl', 'diseas'], ['lung', 'metastasi'], ['prognost', 'biomark'], ['li-fraumeni', 'syndrom'], ['sexual', 'issu'], ['clinic', 'featur'], ['dna', 'mismatch-repair'], ['sept9', 'methyl'], ['east', 'asian'], ['faecal', 'm2-pk'], ['msh6', 'loss'], ['kra', 'mutat'], ['r', 'classif'], ['stem', 'cell'], ['mutyh-associ', 'polyposi'], ['nonpolyposi', 'syndrom'], ['optic', 'colonoscopi'], ['rectosigmoid', 'cancer'], ['cognit', 'dysfunct'], ['prognost', 'factor'], ['bladder', 'irrit'], ['uicc', 'stage'], ['lipomat', 'hemangiopericytoma'], ['tumor', 'bud'], ['colon', 'adenocarcinoma'], ['capsul', 'colonoscopi'], ['rectal', 'irrit']]
my_trigrams = [['lynch_syndrom', 'i'], ['transvers_colon', 'cancer'], ['relaps_free', 'surviv'], ['dna_imag', 'cytometri'], ['adenomat_polyposi', 'syndrom'], ['ihc_msi', 'marker'], ['hamartomat_polyposi', 'syndrom'], ['immun_checkpoint', 'inhibitor'], ['transan_endoscop', 'microsurgeri'], ['polymeras_proofreading-associ', 'polyposi'], ['descend_colon', 'cancer'], ['hepat_arteri', 'infus'], ['molecular_marker', 'test'], ['18q_ai', 'express'], ['colon_kaposi', 'sarcoma'], ['splenic_flexur', 'cancer'], ['adenosquam_colon', 'carcinoma'], ['nervou_system', 'effect'], ['juvenil_polyposi', 'syndrom'], ['epigenet_gene', 'silenc'], ['hereditari_colon', 'cancer'], ['loss_of', 'balanc'], ['kra_mutat', 'test'], ['braf_mutat', 'test'], ['dna_msi', 'marker'], ['adenomat_polyposi', 'coli'], ['metastat_colorect', 'cancer'], ['hyperplast_polyposi', 'syndrom'], ['rectosigmoid_juction', 'cancer'], ['tumor_msi-h', 'express'], ['sigmoid_colon', 'cancer'], ['ascend_colon', 'cancer'], ['ra_mutat', 'test'], ['tyrosin_kinas', 'inhibitor'], ['alter_bowel', 'habit'], ['anal_canal', 'carcinoma'], ['hepat_flexur', 'cancer'], ['oligodontia-colorect_cancer', 'syndrom'], ['polyploidi_test', 'result'], ['inherit_genet', 'disord'], ['submucos_invasivecolon', 'adenocarcinoma'], ['high_fat', 'intak'], ['type_of', 'care'], ['popul_base', 'snp'], ['anal_gland', 'neoplasm'], ['acut_myelocyt', 'leukemia'], ['samson_gardner', 'syndrom'], ['colon_mucin', 'adenocarcinoma']]

#contents = LineIterator("big_home_test.txt")


#NOTE I used tr -d to remove ' from the file
contents = LineIterator("age_fix.txt")

phrases = Phrases(contents,threshold=0.25,scoring="npmi",custom_bigrams=my_bigrams)
bigram = Phraser(phrases)
print "bigrams calculated"
tri_phase = Phrases(bigram[contents],custom_bigrams=my_trigrams,threshold=0.25,scoring="npmi")
trigram = Phraser(tri_phase)
print "trirams calculated"

bigram.save('./preprocessed_big_phrases')
print "ngrams saved"
trigram.save('./preprocessed_trigram_phrases')
print "ngrams saved"
