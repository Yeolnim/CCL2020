# coding=UTF-8

import copy
import pandas as pd
import jieba
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, Embedding, Bidirectional
from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from bert_serving.client import BertClient

model_path = 'en_bert_han.h5'
train = pd.read_csv('../data/en_train.csv')

print(train)
train['sen_cut'] = train['Sentence'].astype(str).apply(jieba.lcut)

X_train = train['sen_cut'].apply(lambda x: ' '.join(x)).tolist()
y_train = pd.get_dummies((np.asarray(train["Label"])))
text = np.array(X_train)

vocab_size = 30000
maxlen = 40

print("开始统计语料的词频信息...")
t = Tokenizer(vocab_size)
t.fit_on_texts(text)
word_index = t.word_index
print('完整的字典大小：', len(word_index))
print("开始序列化句子...")
X_train = t.texts_to_sequences(X_train)
print("开始对齐句子序列...")
X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')
print("完成！")

small_word_index = copy.deepcopy(word_index)  # 防止原来的字典被改变
x = list(t.word_counts.items())
s = sorted(x, key=lambda p: p[1], reverse=True)
print("移除word_index字典中的低频词...")
for item in s[10000:]:
    small_word_index.pop(item[0])  # 对字典pop
print("完成！")

bc = BertClient()
# 定义随机矩阵

embedding_matrix = np.random.uniform(size=(vocab_size + 1, 768))
print("构建embedding_matrix...")
for word, index in small_word_index.items():
    try:
        word_vector = bc.encode([word])
        embedding_matrix[index] = word_vector
        # print("Word: [", index, "]")
    except:
        print("Word: [", word, "] not in wvmodel! Use random embedding instead.")
print("完成！")
print("Embedding matrix shape:\n", embedding_matrix.shape)


# Hierarchical Attention Networks
class Attention(Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


wv_dim = 768
n_timesteps = maxlen
inputs = Input(shape=(maxlen,))
embedding_sequences = Embedding(vocab_size + 1, wv_dim, input_length=maxlen, weights=[embedding_matrix])(inputs)
lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding_sequences)
l = Attention(attention_size=128)(lstm)
l = Dense(128, activation="tanh")(l)
l = Dropout(0.5)(l)
l = Dense(2, activation="softmax")(l)
m = Model(inputs, l)
m.summary()
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

m.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
m.save(model_path)
