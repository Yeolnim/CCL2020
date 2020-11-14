# coding=UTF-8

import pandas as pd
from collections import Counter
import collections

train = pd.read_csv('../data/cn_train.csv')
lst = train['Label']    # lst存放所谓的100万个元素
d = collections.Counter(lst)
# 瞬间出结果
print(d)
print(Counter(lst))
