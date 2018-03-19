from Naked.toolshed.shell import muterun_js
import sys
from gensim.utils import any2utf8
import json
from pprint import pprint
import collections
import unicodedata
from collections import OrderedDict

f = open('output.txt', 'a+')

wiki_list = ["pakistan","france"]

for value in wiki_list:
    response = muterun_js('parser.js',value)


    if response.exitcode == 0: 
        result = response.stdout
        result = result.replace("{","").replace('"','').replace("}","").replace("\n","")
        print result
        #result = unicodedata.normalize('NFKD', unicode(result, "utf-8")).encode('ascii','ignore')
        #test = json.loads(response.stdout, object_pairs_hook= collections.OrderedDict)
        f.write(any2utf8(result))
        f.write("\n")
        #for key,value in test.iteritems():
        #    print key," : ",value
    
f.close()    
    

