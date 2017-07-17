# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
#tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # Send request
  #with open(FLAGS.image, 'rb') as f:
    # See prediction_service.proto for gRPC request/response details.
  
  import pickle
  import numpy as np
  pkl_file = open('test_x.pkl', 'rb')
  test_x = np.array( pickle.load(pkl_file)[:1] )
  #pkl_file = open('test_y.pkl', 'rb')
  #test_y = np.array( pickle.load(pkl_file)[:1] ) 
  
  import numpy as np
  import tensorflow as tf
  import re
  from collections import Counter
  import json
  from pprint import pprint
  from tensorflow.contrib import learn
  import re
  import csv

  # Glove Load:
  filename = 'tensorflow_serving/example/glove.6B.300d.txt'
  def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if (len(row[1:]) == 300):
            vocab.append(row[0])
            embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd

  vocab,embd = loadGloVe(filename)
  vocab_size = len(vocab)
  embedding_dim = len(embd[0])
  embedding = np.asarray(embd)
  vocab = set(vocab)
  word_to_int = {word:i for i,word in enumerate(vocab,1)}
  
  def tokenize_(s):
    pattern = r'''\d+|[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]|[A-Z][A-Z]+|http[s]?://[\w\./]+|[\w]+@[\w]+\.[\w]+|[a-z][a-z]+|[A-Za-z]\.[\w][\w\.]+|[\w]+|[-'a-z]+|[\S]+'''
    l = re.findall(pattern, s)
    return l

  def sentences_to_int(sentences):
    sentence2int = []
    for each in sentences:
        each = tokenize_(each)
        this_sentence_int = []
        for word in each:
            if word in vocab:
                this_sentence_int.append(word_to_int[word])
        sentence2int.append(this_sentence_int)

    max_seq_len = 5
    sentence2int_ = np.zeros((len(sentence2int), max_seq_len), dtype=int)
    for i, row in enumerate(sentence2int):
        sentence2int_[i, :len(row)] = np.array(row[:max_seq_len] )

    return sentence2int_

	    

  #Variables:
  max_seq_len = 5

  #Build Look up table
  W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")
  embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
  embedding_init = W.assign(embedding_placeholder)

  #Build Graph
  X = tf.placeholder(tf.int32, [None, None], name = 'inputs')
  embed = tf.nn.embedding_lookup(W, X)

  #Start session & feed ints
  sentences_string = ['Thank you']
  sentences_int = sentences_to_int(sentences_string)
  sess = tf.Session() 
  sess.run(tf.global_variables_initializer())

  _,test_x2 = sess.run([embedding_init, embed], feed_dict={embedding_placeholder:embedding, X:sentences_int})
  print(np.shape(test_x), np.shape(test_x2) ) 
  
  data= test_x2.astype(np.float32) #,test_y.astype(np.float32) 
  print(np.shape(data)) #, np.shape(label))
 
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'sentiment'
  request.model_spec.signature_name = 'predict_sentiment'
  request.inputs['sentence'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=np.shape(data)) )
 
  result = stub.Predict(request, 100.0)  # 10 secs timeout
  print(result)


if __name__ == '__main__':
  tf.app.run()
