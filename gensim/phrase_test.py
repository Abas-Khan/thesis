from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models.word2vec import LineSentence


import sys
import codecs

import unicodedata
from gensim.utils import any2utf8

contents = LineSentence("test.txt")


common_terms = ["of", "with", "without", "and", "or", "the", "a"]
my_bigrams = [['bit','parts']]
my_trigrams = [[['transverse colon', 'cancer']]]
phrases = Phrases(contents,threshold=0.50,scoring="npmi",custom_bigrams=my_bigrams,common_terms=common_terms)
bigram = Phraser(phrases)
tri_phase = Phrases(bigram[contents],custom_bigrams=my_trigrams,threshold=0.25,scoring="npmi")
trigram = Phraser(tri_phase)

bigram.save('./phrases')
sent = [u'red', u'shift', u'square', u'pants',u'bit', u'parts',u'transverse',u'colon',u'cancer']
#print(bigram[sent])

print trigram[bigram[sent]]

#print item
