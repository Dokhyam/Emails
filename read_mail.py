import numpy as np
from itertools import tee, izip, chain
import os
from gensim.models import word2vec
from sklearn import tree
import re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets, svm
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import string
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier 

def mysplit(s):
  return filter(None, re.split(r'(\d+)', s))


def clean_data(data):
	punctuations = list(string.punctuation)
	data = data.replace("\n"," ").replace(":", " ").replace(",","").replace(".","").replace("'s","").replace("?","")
	stemmer = PorterStemmer()
	stemmer2 = SnowballStemmer('english')
	tokenizer = RegexpTokenizer(r'\w+')
	tokenizer.tokenize(data)
	ndata1 = list(mysplit(data))
	ndata1 = [[stemmer.stem(xi) for xi in y.split(" ")] for y in ndata1] 
	ndata1 = [[stemmer2.stem(xi) for xi in y] for y in ndata1]
	ndata = [x for x in ndata1 if not x == ":"]
	ndata = [filter(None, x) for x in ndata]
	ndata = [x for x in ndata if x != []]
	return ndata

def create_clusters(word_vectors, num_clusters):
	kmeans_clustering = KMeans( n_clusters = num_clusters )
	idx = kmeans_clustering.fit_predict( word_vectors )
	return idx

#read data and clean
folders = os.listdir("blair_mail")
m = []
i = 0
labels = []
os.chdir ('blair_mail')
for f_path in folders:
	mails = os.listdir(f_path)
	i +=i
	for mail in mails:
		temp = open(os.path.join(f_path,mail), 'r')
		data = temp.read()
		data1 = clean_data(data)
		labels.append(i)
		m.append(data1)

#convert data to vectors
word_vec =  word2vec.Word2Vec(list(chain(*m)))
word_vec2 = word_vec.syn0
datal = word_vec2.shape[0]
#create cluster vectors
num_clusters = datal/10
clusters = create_clusters(word_vec2, num_clusters)
word_centroid_map = dict(zip( word_vec.index2word, clusters ))

#create training/test sets
perm1 = np.random.permutation(datal)
train_size = round(0.8* float(datal))
train_idx = perm1[0:train_size]
test_idx = perm1[train_size+1:]
d_train = word_vec2[train_idx,:]
d_test = word_vec2[test_idx,:]

print d_train
print d_test


