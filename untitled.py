import os 
from numpy import exp, dot, zeros, outer, random, dtype, get_include, float32
import pyximport
models_dir = os.path.dirname(__file__) or os.getcwd()

print(models_dir)
pyximport.install(setup_args={"include_dirs": [models_dir, get_include()]})
from gensim.models.word2vec_inner import train_batch_sg, train_batch_cbow, FAST_VERSION

