import cPickle as pickle
some_stuf="cancer , madness,sick ill"

output = open('data.pkl', 'wb')
pickle.dump(some_stuf,output)
output.close()


pkl_file = open('data.pkl', 'rb')

data1 = pickle.load(pkl_file)

print(data1[0])