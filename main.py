import os
import re
import sklearn.model_selection
from data_utils import load_sentences, prepare_data, create_emb_index, create_emb_matrix
from model import create_model, seg_dim

batch_size = 30
epochs = 50

#path for data
# train_file = os.path.join("data", "example.train")
# dev_file = os.path.join("data", "example.dev")
#path for data_medicine_three
# train_file = os.path.join("data", "example_medicine_three.train")
# dev_file = os.path.join("data", "example_medicine_three.dev")
#path for data_medicine_all
train_file = os.path.join("data", "example_medicine_all.train")
dev_file = os.path.join("data", "example_medicine_all.dev")
emb_file = os.path.join("data", "wiki_100.utf8")    #path for pre_trained embedding

#load data and get sentences
train_sentences = load_sentences(train_file)
dev_sentences = load_sentences(dev_file)
# print(train_sentences[5], '\n', dev_sentences[5], '\n', dev_sentences[5])

#prepare data，对sentences进行处理得到sentence的序列化表示，以及word到ID的映射序列
train_data = prepare_data(train_sentences, seg_dim)
# print("train_data:", "\n", train_data[0][2], "\n", train_data[1], "\n", train_data[2][2])
dev_data = prepare_data(dev_sentences, seg_dim)

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
if seg_dim:
    seg_sequence_train = train_data[4]
    seg_sequence_dev = dev_data[4]
    model.fit([word_sequence_train, seg_sequence_train], labels_train, batch_size=batch_size, epochs=epochs)
    score, acc = model.evaluate([word_sequence_dev, seg_sequence_dev], tags_sequence_dev, batch_size=batch_size)
else:
    model.fit(word_sequence_train, labels_train, batch_size=batch_size, epochs=epochs)
    score, acc = model.evaluate(word_sequence_dev, tags_sequence_dev, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)