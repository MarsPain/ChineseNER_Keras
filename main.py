import os
import re
import sklearn.model_selection
from data_utils import load_sentences, prepare_data, create_emb_index, create_emb_matrix
from model import create_model

batch_size = 32

#path for data
train_file = os.path.join("data", "example.train")
test_file = os.path.join("data", "example.test")
dev_file = os.path.join("data", "example.dev")
#path for data_medicine
# train_file = os.path.join("data", "example_medicine.train")
# test_file = os.path.join("data", "example_medicine.test")
# dev_file = os.path.join("data", "example_medicine.dev")
emb_file = os.path.join("data", "wiki_100.utf8")    #path for pre_trained embedding

#load data and get sentences
train_sentences = load_sentences(train_file)
dev_sentences = load_sentences(dev_file)
test_sentences = load_sentences(test_file)
# print(train_sentences[5], '\n', dev_sentences[5], '\n', dev_sentences[5])

#prepare data，对sentences进行处理得到sentence的序列化表示，以及word到ID的映射序列
train_data = prepare_data(train_sentences)
# print("train_data:", "\n", train_data[0][2], "\n", train_data[1], "\n", train_data[2][2])
dev_data = prepare_data(dev_sentences)
test_data = prepare_data(test_sentences)

word_index, tag_index = train_data[1], train_data[3]
# print(len(word_index))
word_sequence_train, labels_train = train_data[0], train_data[2]
# print(labels_train, labels_train.shape)
# print(word_sequence_train[50], '\n', tags_sequence_train[50])
embedding_index = create_emb_index(emb_file)
# print(embedding_index)
embedding_matrix = create_emb_matrix(word_index, embedding_index)
# print(embedding_matrix)
word_sequence_dev, tags_sequence_dev = dev_data[0], dev_data[2]

model = create_model(embedding_matrix, tag_index)
model.fit(word_sequence_train, labels_train, batch_size=batch_size, epochs=3)
score, acc = model.evaluate(word_sequence_dev, tags_sequence_dev, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)