import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import string
from collections import Counter
from collections import OrderedDict
from glob import iglob
import re
import os
from shutil import copy2
import math
import heapq
from scipy.spatial import distance

embed = np.load("embeddings.npz")["emb"] # size: 41524 by 128
indices = np.load("embeddings.npz")["word2ind"].flatten()[0]
word_index = {i: j for j, i in indices.items()}


'''------------------------------------------------------------------------------------------------------'''
'''                                       Part 8                                                         '''
'''------------------------------------------------------------------------------------------------------'''
def part8():
    words = ['story', 'good', 'student', 'teacher']
    
    for word in words:
        source = embed[word_index[word]]
        distances = []
        
        for i in range(len(embed)):
            if i != word_index[word]:
                
                dist = np.linalg.norm(source - embed[i])
                heapq.heappush(distances, (dist, indices[i]))
        
        print "\nTop ten closest to " + word
        for i in heapq.nsmallest(10, distances):
            print i[1]

print 'Executing Part 8------------------------------------'
part8()

'''------------------------------------------------------------------------------------------------------'''
'''                                       Setup                                                          '''
'''------------------------------------------------------------------------------------------------------'''
def get_data(set_name, limit):
    x = np.zeros((0,256))
    y = np.zeros((0,2))
    
    keywords = []
    pairs = []
    count = 0
    
    for filepath in iglob(os.path.join('txt_sentoken/pos/' + set_name, '*.txt')):
        with open(filepath) as file:
            if limit < count:
                break 
            count += 1
            
            words = file.read().lower().translate(None, string.punctuation).split()
            keywords += words
            
            for k in range(len(words) - 1): 
                if (words[k] in word_index) and (words[k + 1] in word_index):
                    if (words[k] < words[k + 1]):
                        x_ = np.concatenate((embed[word_index[words[k]]], embed[word_index[words[k + 1]]]))
                        x = np.vstack((x, x_))
                        y = np.vstack((y, np.array([1, 0])))
                        pairs.append((words[k], words[k + 1]))
                    else:
                        x_ = np.concatenate((embed[word_index[words[k + 1]]], embed[word_index[words[k]]]))
                        x = np.vstack((x, x_))
                        y = np.vstack((y, np.array([1, 0])))
                        pairs.append((words[k + 1], words[k]))
                           
    keywords = list(OrderedDict.fromkeys(keywords))
    
    print 'Fetched keywords'
    print ('For ' + str(x.shape[0]) + ' iterations')
    
    for i in range(x.shape[0]):
        word1 = keywords[random.randint(0, len(keywords) - 1)]
        word2 = keywords[random.randint(0, len(keywords) - 1)]
        if (word1 > word2):
            temp = word1
            word1 = word2
            word2 = temp
 
        while (word1 not in word_index) or (word2 not in word_index) or ((word1, word2) in pairs):
            word1 = keywords[random.randint(0, len(keywords) - 1)]
            word2 = keywords[random.randint(0, len(keywords) - 1)]
            if (word1 > word2):
                temp = word1
                word1 = word2
                word2 = temp    
                        
        x_ = np.concatenate((embed[word_index[word1]], embed[word_index[word2]]))
        x = np.vstack((x, x_))
        y = np.vstack((y, np.array([0, 1])))
    
    return x, y
    
def get_train_batch(x_train, y_train, N): 
    batch_xs = np.zeros((0, x_train.shape[1]))
    batch_y_s = np.zeros((0, y_train.shape[1]))
        
    for k in range(N):
        idx = random.sample(range(x_train.shape[0]), N)
        batch_xs = np.vstack((batch_xs, x_train[idx[k]]))
        batch_y_s = np.vstack((batch_y_s, y_train[idx[k]]))
    return batch_xs, batch_y_s

'''------------------------------------------------------------------------------------------------------'''
'''                                       Part 7                                                         '''
'''------------------------------------------------------------------------------------------------------'''
print 'Executing Setup------------------------------------'
training_performance = []
test_performance = []
validation_performance = []

x_test, y_test = get_data('test', 20)
print 'Fetched test data'

x_validation, y_validation = get_data('valid', 5)
print 'Fetched validation data'

x_training, y_training = get_data('train', 5)
print 'Fetched training data'

print 'Executing Part 7------------------------------------'
x = tf.placeholder(tf.float32, [None, x_training.shape[1]])

W0 = tf.Variable(tf.random_normal([256, 2], stddev=0.01)) 
b0 = tf.Variable(tf.random_normal([2], stddev=0.01))

layer1 = tf.nn.sigmoid(tf.matmul(x, W0)+b0)

y = tf.nn.softmax(layer1)
y_ = tf.placeholder(tf.float32, [None, y_training.shape[1]])

lam = 0.00001
decay_penalty =lam*tf.reduce_sum(tf.square(W0))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.AdamOptimizer(0.00005).minimize(reg_NLL)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

x_axis = []
for i in range(2000):
    batch_xs, batch_ys = get_train_batch(x_training, y_training, 50) 
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    if i % 200 == 0:
        print "i=",i
        x_axis.append(i)
        
        training_accuracy = sess.run(accuracy, feed_dict={x: x_training, y_: y_training})
        training_performance.append(training_accuracy * 100)
        print "Training:", training_accuracy
        
        test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
        test_performance.append(test_accuracy * 100)
        print "Test:", test_accuracy
    
        validation_accuracy = sess.run(accuracy,feed_dict={x: x_validation, y_: y_validation})
        validation_performance.append(validation_accuracy * 100)
        print "Validation:", validation_accuracy

x_axis = np.array(x_axis)

plt.figure()
plt.title("Performances")
plt.xlabel("Iterations")
plt.ylabel("Performance Accuracy")
plt.plot(x_axis, test_performance, label="test")
plt.plot(x_axis, training_performance, label="training")
plt.plot(x_axis, validation_performance, label="validation")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.show()
plt.savefig("part7.png")
