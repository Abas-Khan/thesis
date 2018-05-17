from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import numpy as np

content = LineSentence("testing_nos.txt")
for item in content:
    print item
common_terms = ["no"]
phrases = Phrases(content, common_terms=common_terms, min_count=1)
bigram = Phraser(phrases)


sent = [u'no', u'oranges', u'tomatos', u'oranges', u'fruites', u'no', u'fruites']
result = bigram[sent]
li = [i for i,val in enumerate(result) if val=='no']
print li,".........."

for idx in li:
    result[idx+1] = "no " + result[idx+1]
final_result = filter(lambda a: a != "no", result)    
print final_result

