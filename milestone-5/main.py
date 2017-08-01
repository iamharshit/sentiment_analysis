import data_util
import numpy as np
import CNN_wrapper

raw_data = data_util.load('dataset/labeledTrainData.tsv')[:10000]
data = data_util.sentence2indexs(raw_data)

train, test, valid = data_util.split_dataset_(data)

model = CNN_wrapper.CNN(seq_len=data_util.max_sentence_len, 
							n_classes=1,
							vocab_size=data_util.vocab_size)

model.train(train, valid)
