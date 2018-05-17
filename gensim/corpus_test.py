from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
import multiprocessing
from gensim.models.word2vec import LineSentence
from nltk.corpus import stopwords
from gensim.utils import call_on_class_only
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
import unicodedata

'''
wiki = WikiCorpus("enwiki-latest-pages-articles.xml.bz2")
#wiki = WikiCorpus("enwiki-YYYYMMDD-pages-articles.xml.bz2")
the_file = open('wikipedia_data.txt', 'a')


import sys
class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def write_to_file(self):
        for idx,(content, (page_id, title)) in enumerate(self.wiki.get_texts()):
            print idx,
            print "\r",
            if idx>500:
                break
              
            content = [unicodedata.normalize('NFKD', unicode(c)).encode('ascii', 'ignore') for c in content]
            the_file.write(utils.any2utf8(title)+":"+ utils.any2utf8(" ".join(content)))
            the_file.write("\n")    
        the_file.close()     
        
documents = TaggedWikiDocument(wiki)
documents.write_to_file()    
   


'''
class TaggedWiki(object):

    def __init__(self, source):
        
        self.source = source

    def __iter__(self):
        """Iterate through the lines in the source."""

        with utils.smart_open(self.source) as fin:
            for item_no, line in enumerate(fin):
                #content = line
                content = line.split(":")
                text = content[1].split()
                title = content[0]


                
                yield TaggedDocument([unicodedata.normalize('NFKD', unicode(c)).encode('ascii', 'ignore') for c in text], [title])

contents = TaggedWiki("wikipedia_data.txt")
cores = multiprocessing.cpu_count()

domain_vocab_file = "poet poets symfony symfonies ghazal ghazals poem poems"
vocab_list = domain_vocab_file.split()
model = Doc2Vec(documents=contents, vector_size=200, window=8, 
                            min_count=19, workers=1,hs=0,negative=5,
                            dm=0,dbow_words=1,epochs=20, smoothing=0.75,
                            sampling_param=0.5, objective_param=0.3, vocab_file=vocab_list)                                                           