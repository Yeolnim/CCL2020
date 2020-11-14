# coding=UTF-8

import pandas as pd
import jieba
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer

class Attention(Layer):
    def __init__(self, attention_size=128, **kwargs):
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

a=Attention(attention_size=128)

model = load_model('./model/cn_roberta_han.h5',custom_objects={'Attention': a})

train = pd.read_csv('./data/cn_train.csv')
test = pd.read_csv('./data/cn_dev.csv')

train['Sentence'] = train['Sentence'].astype(str)
train['sen_cut'] = train['Sentence'].apply(jieba.lcut)
test['Sentence'] = test['Sentence'].astype(str)
test['sen_cut'] = test['Sentence'].apply(jieba.lcut)
X_train = train['sen_cut'].apply(lambda x: ' '.join(x)).tolist()
X_test = test['sen_cut'].apply(lambda x: ' '.join(x)).tolist()

text = np.array(X_train)

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

vocab_size = 30000
maxlen = 100

print("开始统计语料的词频信息...")
t = Tokenizer(vocab_size)
t.fit_on_texts(text)
word_index = t.word_index
print('完整的字典大小：', len(word_index))

print("开始序列化句子...")
X_test = t.texts_to_sequences(X_test)
print("开始对齐句子序列...")
X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')
print("完成！")

predicted = np.array(model.predict(X_test))
print(predicted)
test_predicted=np.argmax(predicted,axis=1)

test['Label']=test_predicted
# test['id']=test['数据编号']

# test['label'].replace({0:'angry', 1:'fear',2:'happy',3:'neural',4:'sad',5:'surprise'}, inplace=True)


order=['ID','Label']
result = test[order]
result.to_csv('./result/cn_DUFL.csv', encoding='utf-8',index=False)
