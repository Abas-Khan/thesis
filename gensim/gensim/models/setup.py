from Cython.Build import cythonize
from distutils.core import setup, Extension
import os 
import numpy
from numpy import exp, dot, zeros, outer, random, dtype, get_include

models_dir = os.path.dirname(__file__) or os.getcwd()
setup(
    ext_modules=[
        Extension("doc2vec_inner", ["doc2vec_inner.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)
setup(
    ext_modules=cythonize("doc2vec_inner.pyx"),
    include_dirs=[numpy.get_include()]
)    
