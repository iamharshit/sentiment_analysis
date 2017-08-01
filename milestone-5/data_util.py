import pandas as pd
import numpy as np

vocab_size = 0	
max_sentence_len = 20

def load(filename):
    df = pd.read_csv(filename,delimiter='\t')
    ans = zip( list(df['review']), list(df['sentiment']))
    return ans

def split_dataset_(data, ratio = [0.7, 0.15, 0.15]):
	x = np.array(data)[:,0]
	y = np.array(data)[:,1]
	data_len = len(x)
	lens = [ int(data_len*item) for item in ratio ]

	trainX, trainY = x[:lens[0]], y[:lens[0]]
	testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
	validX, validY = x[-lens[-1]:], y[-lens[-1]:]

	return (trainX,trainY), (testX,testY), (validX,validY)

def sentence2indexs(data):
	global max_sentence_len
	import tensorflow as tf
	preprocessor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sentence_len)
	review, sentiment = np.array(data)[:,0], np.array(data)[:,1]

	review = np.array(list(preprocessor.fit_transform(review)))	
	global vocab_size
	vocab_size = len(preprocessor.vocabulary_)

	return zip(review, sentiment.astype(np.float))
