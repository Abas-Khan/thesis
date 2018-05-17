import re 
PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
from nltk.tokenize import word_tokenize
from gensim import utils, matutils 

def simple_tokenize_org(text):
    """Tokenize input test using :const:`gensim.utils.PAT_ALPHABETIC`.

    Parameters
    ----------
    text : str
        Input text.

    Yields
    ------
    str
        Tokens from `text`.

    """
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()

def simple_tokenize(text):
    """Tokenize input test using :const:`gensim.utils.PAT_ALPHABETIC`.

    Parameters
    ----------
    text : str
        Input text.

    Yields
    ------
    str
        Tokens from `text`.

    """
    text = re.sub(r"\.[^\w]"," ",text)
    return word_tokenize(text)
    



string ="This is ok. But we need 2.5."

with utils.smart_open("test.txt") as fin:
    for item_no, line in enumerate(fin):

        result = simple_tokenize(line)
        print result