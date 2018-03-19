from gensim.models import Phrases
from gensim.models.phrases import Phraser


import sys
import codecs

import unicodedata
from gensim.utils import any2utf8

contents = []
with codecs.open("sports.txt",encoding='utf-8') as infile:
    for line in infile:
        #print unicodedata.normalize('NFKD', line).encode('ascii','ignore')
        contents.append(line)


phrases = Phrases(contents)
bigram = Phraser(phrases)
sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
print(bigram[sent])
