from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.word2vec import LineSentence
import multiprocessing
"""
sentences = LineSentence('countries_filter.txt')
cores = multiprocessing.cpu_count()

model = Doc2Vec(documents = sentences,dm=0, size=200, window=8, min_count=1, iter=10, workers=cores)
"""
cores = multiprocessing.cpu_count()
sentences = TaggedLineDocument('countries_filter.txt')
#sentences = LineSentence('countries_filter.txt')
# train word2vec on the two sentences
#model = Word2Vec(sentences, min_count=1,sg=1)
model = Doc2Vec(documents = sentences,dm=1, size=200, window=8, min_count=1, iter=10, workers=cores)