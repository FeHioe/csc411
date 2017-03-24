from naivebayes import log_sum
import string
import numpy as np
from numpy import random
from collections import Counter
from collections import OrderedDict
from glob import iglob
import re
import os
from shutil import copy2
import math
import heapq

neg_keywords = {}
pos_keywords = {}
pos_count = 0
neg_count = 0

for filepath in iglob(os.path.join('txt_sentoken/pos/train', '*.txt')):
    with open(filepath) as file:
        words = file.read().lower().translate(None, string.punctuation).split()
        words = list(OrderedDict.fromkeys(words))
        length = len(words)        
        for word in words:
            if word not in pos_keywords:
                pos_keywords[word] = 1.0 / float(length)
            else:
                pos_keywords[word] += 1.0 / float(length)
        pos_count += 1.0 / float(length)

for filepath in iglob(os.path.join('txt_sentoken/neg/train', '*.txt')):
    with open(filepath) as file:
        words = file.read().lower().translate(None, string.punctuation).split()
        words = list(OrderedDict.fromkeys(words))
        length = len(words)
        for word in words:
            if word not in neg_keywords:
                neg_keywords[word] = 1.0 / float(length)
            else:
                neg_keywords[word] += 1.0 / float(length)
        neg_count += 1.0 / float(length)

pos_prob = float(pos_count) / float(pos_count + neg_count) 
neg_prob = float(neg_count) / float(pos_count + neg_count)

training_set = {i: pos_keywords.get(i, 0) + neg_keywords.get(i, 0) for i in set(pos_keywords) | set(neg_keywords)}
 
nb_thetas = []
for word in training_set:
    words = [word]
    diff = log_sum(words, pos_keywords, pos_count, pos_prob) - log_sum(words, neg_keywords, neg_count, neg_prob)
    heapq.heappush(nb_thetas, (abs(diff), word))

for i in heapq.nlargest(100, nb_thetas):
    print(i)

# Top 100 linear regression thetas can be found in part6_data.txt after running logistics.py