from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from multiprocessing import cpu_count
from gensim.models import Doc2Vec
from gensim.models.word2vec import LineSentence
from gensim.similarities import MatrixSimilarity, WmdSimilarity, SoftCosineSimilarity
import re
from gensim import utils, matutils



def softcossim(query, documents):
    # Compute Soft Cosine Measure between the query and the documents.
    query = tfidf[dictionary.doc2bow(query)]
    index = SoftCosineSimilarity(
        tfidf[[dictionary.doc2bow(document) for document in documents]],
        similarity_matrix)
    similarities = index[query]
    return similarities





fin = open("age_fix.txt","r").read()
documents = fin.split("\n")
corpus = LineSentence('age_fix.txt')


dictionary = Dictionary(corpus)
tfidf = TfidfModel(dictionary=dictionary)
print "loading model"
#NOTE fixed_stemmoutput was trained wthout removing stopwords
model = Doc2Vec.load("./preprocessed_with_everything/mod")
print "model loaded"

similarity_matrix = model.wv.similarity_matrix(dictionary, tfidf, nonzero_limit=100)
print("Number of unique words: %d" % len(dictionary))

new_q = "70 year-old man. elderly patient gender male. colon cancer. lung metastasis lung metastases"
#query_doc = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:"," ",query_doc)

new_q = re.sub(r"(?<=\w[^\d])\.|\.(?=[^\d])|\(|\)|\[|\]|,(?= )|((?<=[^\w])-|-(?=[^\w]))|:|\?|\;"," ",new_q)
new_q = utils.to_unicode(new_q).split()
results = softcossim(new_q,corpus)




