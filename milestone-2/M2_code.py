
# coding: utf-8

# # MILESTONE 2
# 
# IMDB dataset + Siraj's Network

# In[1]:

import numpy as np
import tensorflow as tf


# ## Preprocessing Dataset
# 
# 1. Removing punctuations
# 2. Generating word_to_int map
# 3. Coverting each review in ints
# 4. Padding each review with 0's and generating input of length 200

# In[2]:

import re
from collections import Counter
from nltk.corpus import stopwords

def preprocess(text):
    
    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <PERIOD> ')
    text = text.replace('"', ' <PERIOD> ')
    text = text.replace(';', ' <PERIOD> ')
    text = text.replace('!', ' <PERIOD> ')
    text = text.replace('?', ' <PERIOD> ')
    text = text.replace('(', ' <PERIOD> ')
    text = text.replace(')', ' <PERIOD> ')
    text = text.replace('--', ' <PERIOD> ')
    text = text.replace('?', ' <PERIOD> ')
    text = text.replace('<br />', ' <PERIOD> ')
    text = text.replace('\\', ' <PERIOD> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <PERIOD> ')
    text = text.replace(' <PERIOD> ', ' ')
    words = text.split()
    
    return words

def removing_noise(words):
    word_count = Counter(words)
    #stops = set(stopwords.words("english"))
    words_new = [word for word in words if (word_count[word]>5) #and (not word in stops)
                ]
    return words_new
    


# In[3]:

import csv

filename = 'data/labeledTrainData.tsv'
review_ids = []
reviews = []
labels = []
#importing dataset into lists
with open(filename, 'r') as f:
    next(f)
    reader = csv.reader(f, delimiter='\t')
    
    for row in reader:
        review_ids.append(row[0])
        labels.append([int(row[1])] )
        reviews.append(row[2])


# In[4]:

reviews_pp = []
words = []

for review in reviews:
    review_pp = preprocess(review)
    reviews_pp.append(review_pp)
    words.extend(review_pp)
    
words = removing_noise(words)


# In[5]:

#Converting word to integers and making the vocabulary
vocab = set(words)
vocab_size = len(vocab)
words_count = Counter(words)
sorted_vocab = sorted(words_count, key = words_count.get, reverse = True)
word_to_int = {word:i for i,word in enumerate(sorted_vocab,1)}

#Converting each review in the form of integers
reviews_pp_ints = []
for review in reviews_pp:
    this_review_int = []
    for word in review:
        if word in vocab:
            this_review_int.append(word_to_int[word])
    reviews_pp_ints.append(this_review_int)


# In[6]:

len(reviews_pp_ints[0])


# In[7]:

len(reviews_pp[0])


# In[8]:

len(reviews_pp_ints)


# In[9]:

max_seq_len = 200
features = np.zeros((len(reviews_pp_ints), max_seq_len), dtype=int)
for i, row in enumerate(reviews_pp_ints):
    features[i, :len(row)] = np.array(row[:max_seq_len] )


# In[10]:

# 'features' is a 2d array storing all sequences


# ## Train Test Validation split

# In[11]:

split_frac = 0.8
split_idx = int(len(features)*0.8)
train_x, val_x = features[:split_idx], features[split_idx:]
train_y, val_y = labels[:split_idx], labels[split_idx:]

test_idx = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_idx], val_x[test_idx:]
val_y, test_y = val_y[:test_idx], val_y[test_idx:]


# ## Building Network

# In[12]:

input_size = 200
embed_size = 500
lstm_size = 128
lstm_layers = 1
batch_size = 500

X = tf.placeholder(tf.int32, [None, None], name = 'inputs')
Y = tf.placeholder(tf.float32, [None, 1], name = 'labels')

embedding = tf.Variable(tf.random_uniform((vocab_size+1, embed_size), -1, 1))
embed = tf.nn.embedding_lookup(embedding, X)

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
cell = tf.contrib.rnn.MultiRNNCell([lstm]*lstm_layers)

#getting an initial state of zeros\n",
initial_state = cell.zero_state(batch_size, tf.float32)

outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state = initial_state)

predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
#predictions,Y\n",

loss = tf.reduce_mean(tf.square(Y - predictions))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)


# In[13]:

#Accuracy:
correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.float32), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# ## Training

# In[14]:

n_epochs = 10


# In[17]:

def get_batches(x, y, batch_size=100):
    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# In[ ]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(n_epochs):
        state = sess.run(initial_state)
        iteration = 1
        
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {X: x, Y: y, initial_state: state}
            
            state, loss_,  _ = sess.run([final_state, loss, optimizer], feed_dict=feed)
            
            if iteration%5==0:
                print("Epoch: {}/{}".format(e, n_epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss_))
            
            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {X: x,
                            Y: y,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            
            iteration +=1
            
print('Training Completed')


# ## Testing

# In[ ]:

test_acc = []

test_state = sess.run(cell.zero_state(batch_size, tf.float32))
for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
    feed = {X: x,Y: y,initial_state: test_state}
    
    batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
    test_acc.append(batch_acc)
    
print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


# In[ ]:



