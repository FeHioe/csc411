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

'''------------------------------------------------------------------------------------------------------'''
'''                                       Part 1                                                         '''
'''------------------------------------------------------------------------------------------------------'''

def word_freqency():
    neg_keywords = Counter()
    pos_keywords = Counter()

    for filepath in iglob(os.path.join('txt_sentoken/neg/', '*.txt')):
        with open(filepath) as file:
            neg_keywords.update(file.read().lower().translate(None, string.punctuation).split())
    
    neg = open("top_neg.txt", "wb")
    neg.write( "Negative Top 100\r\n")
    for word, count in neg_keywords.most_common(500):
        neg.write('{}: {}\r\n'.format(count, word))
    neg.close()
    
    for filepath in iglob(os.path.join('txt_sentoken/pos/', '*.txt')):
        with open(filepath) as file:
            pos_keywords.update(file.read().lower().translate(None, string.punctuation).split())
    
    pos = open("top_pos.txt", "wb")
    pos.write( "Positive Top 100\r\n")
    for word, count in pos_keywords.most_common(500):
        pos.write('{}: {}\r\n'.format(count, word))
    pos.close()

'''------------------------------------------------------------------------------------------------------'''
'''                                       Part 2                                                         '''
'''------------------------------------------------------------------------------------------------------'''
def get_sets():
    if not(os.path.exists("txt_sentoken/neg/train")):
        os.makedirs("txt_sentoken/neg/train")
    if not(os.path.exists("txt_sentoken/neg/test")):
        os.makedirs("txt_sentoken/neg/test")
    if not(os.path.exists("txt_sentoken/neg/valid")):
        os.makedirs("txt_sentoken/neg/valid")
        
    if not(os.path.exists("txt_sentoken/pos/train")):
        os.makedirs("txt_sentoken/pos/train")
    if not(os.path.exists("txt_sentoken/pos/test")):
        os.makedirs("txt_sentoken/pos/test")
    if not(os.path.exists("txt_sentoken/pos/valid")):
        os.makedirs("txt_sentoken/pos/valid")
    
    np.random.seed(0)
    rand_pos = np.array(random.permutation(1000))
    rand_neg = np.array(random.permutation(1000))
    
    for i in range(800):
        copy2('txt_sentoken/pos/' + os.listdir('txt_sentoken/pos/')[rand_pos[i]], 'txt_sentoken/pos/train')
        copy2('txt_sentoken/neg/' + os.listdir('txt_sentoken/neg/')[rand_neg[i]], 'txt_sentoken/neg/train')

    for i in range(800, 900):
        copy2('txt_sentoken/pos/' + os.listdir('txt_sentoken/pos/')[rand_pos[i]], 'txt_sentoken/pos/test')
        copy2('txt_sentoken/neg/' + os.listdir('txt_sentoken/neg/')[rand_neg[i]], 'txt_sentoken/neg/test')
        
    for i in range(900, 1000):
        copy2('txt_sentoken/pos/' + os.listdir('txt_sentoken/pos/')[rand_pos[i]], 'txt_sentoken/pos/valid')
        copy2('txt_sentoken/neg/' + os.listdir('txt_sentoken/neg/')[rand_neg[i]], 'txt_sentoken/neg/valid')

def performance(pos_path, neg_path, set_name):
    pos_accuracy = 0
    neg_accuracy = 0
    
    if (set_name == 'train'):
        num_files = 800
    else:
        num_files = 100
    
    for filepath in iglob(os.path.join(pos_path, '*.txt')):
        with open(filepath) as file:
            words = file.read().lower().translate(None, string.punctuation).split()
            words = list(OrderedDict.fromkeys(words))
            if (log_sum(words, pos_keywords, pos_count, pos_prob) >= log_sum(words, neg_keywords, neg_count, neg_prob)):
                pos_accuracy += 1
    print "\n" + set_name + " positive performance: " + str(float(pos_accuracy * 100) / float(num_files)) + "%"
    
    for filepath in iglob(os.path.join(neg_path, '*.txt')):
        with open(filepath) as file:
            words = file.read().lower().translate(None, string.punctuation).split()
            words = list(OrderedDict.fromkeys(words))
            if not (log_sum(words, pos_keywords, pos_count, pos_prob) >= log_sum(words, neg_keywords, neg_count, neg_prob)):
                neg_accuracy += 1
    print set_name + " negative performance: " + str(float(neg_accuracy * 100) / float(num_files)) + "%"
    
    print set_name + " performance: " + str(float((pos_accuracy + neg_accuracy) * 100) / float(num_files * 2)) + "%"
    
def log_sum(words, keywords, np_count, prob):
    log_sum = 0
    
    m = 0.21
    k = 300
    
    for word in words:
        if word in keywords:
            count = keywords[word]
        else:
            count = 0
        log_sum += math.log((count + m * k) / (np_count + k))

    return log_sum + math.log(prob)

def part2():    
    performance('txt_sentoken/pos/train', 'txt_sentoken/neg/train', 'train')
    performance('txt_sentoken/pos/test', 'txt_sentoken/neg/test', 'test')
    performance('txt_sentoken/pos/valid', 'txt_sentoken/neg/valid', 'validation')

'''------------------------------------------------------------------------------------------------------'''
'''                                       Part 3                                                         '''
'''------------------------------------------------------------------------------------------------------'''
def part3(): 
    pos = []
    neg = []
    
    for word in pos_keywords:
        prob_word = log_sum([word], pos_keywords, pos_count, pos_prob) - (math.log(training_set[word] / (pos_count + neg_count)))
        heapq.heappush(pos, (prob_word, word))

    print "\nTop 10 Positive:"
    print [i[1] for i in heapq.nlargest(10, pos)]

    for word in neg_keywords:
        prob_word = log_sum([word], neg_keywords, neg_count, neg_prob) - (math.log(training_set[word] / (pos_count + neg_count)))
        heapq.heappush(neg, (prob_word, word))

    print "\nTop 10 Negative:"
    print [i[1] for i in heapq.nlargest(10, neg)]

'''------------------------------------------------------------------------------------------------------'''
'''                                       Executiuon                                                     '''
'''------------------------------------------------------------------------------------------------------'''
if __name__ == "__main__":
    print 'Executing Setup------------------------------------'
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
    
    print 'Executing Part 1------------------------------------'
    part1()
    print 'Executing Part 2------------------------------------'
    part2()
    print 'Executing Part 3------------------------------------'
    part3()
    

