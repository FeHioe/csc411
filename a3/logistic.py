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

'''------------------------------------------------------------------------------------------------------'''
'''                                       Setup                                                          '''
'''------------------------------------------------------------------------------------------------------'''
def get_keywords():
    keywords = []
    
    for filepath in iglob(os.path.join('txt_sentoken/pos/train', '*.txt')):
        with open(filepath) as file:
            words = file.read().lower().translate(None, string.punctuation).split()
            keywords += words
    
    for filepath in iglob(os.path.join('txt_sentoken/neg/train', '*.txt')):
        with open(filepath) as file:
            words = file.read().lower().translate(None, string.punctuation).split()
            keywords += words
    
    return list(OrderedDict.fromkeys(keywords))
    
def get_data(set_name):        
    x = np.zeros((0, len(keywords)))
    y = np.zeros((0, 2)) # pos and neg

    for filepath in iglob(os.path.join('txt_sentoken/pos/' + set_name, '*.txt')):
        with open(filepath) as file:
            
            words = file.read().lower().translate(None, string.punctuation).split()
            x_ = np.zeros((1, len(keywords)))
            
            for word in words:
                if word in keywords:
                    x_[0][keywords.index(word)] = 1
                    
            x = np.vstack((x, x_))
            encode = np.zeros(2)
            encode[0] = 1
            y = np.vstack((y, encode))
            
    for filepath in iglob(os.path.join('txt_sentoken/neg/' + set_name, '*.txt')):
        with open(filepath) as file:
            
            words = file.read().lower().translate(None, string.punctuation).split()
            x_ = np.zeros((1, len(keywords)))
            
            for word in words:
                if word in keywords:
                    x_[0][keywords.index(word)] = 1
                    
            x = np.vstack((x, x_))
            encode = np.zeros(2)
            encode[1] = 1
            y = np.vstack((y, encode))

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
'''                                       Part 4                                                         '''
'''------------------------------------------------------------------------------------------------------'''
print 'Executing Setup------------------------------------'
training_performance = []
test_performance = []
validation_performance = []

keywords = get_keywords()
# print len(keywords)
print 'Fetched keywords'

x_test, y_test = get_data('test')
print 'Fetched test data'

x_validation, y_validation = get_data('valid')
print 'Fetched validation data'

x_training, y_training = get_data('train')
print 'Fetched training data'

print 'Executing Part 4------------------------------------'
x = tf.placeholder(tf.float32, [None, x_training.shape[1]])

W0 = tf.Variable(tf.random_normal([len(keywords), 2], stddev=0.01)) 
b0 = tf.Variable(tf.random_normal([2], stddev=0.01))

layer1 = tf.nn.sigmoid(tf.matmul(x, W0)+b0)

y = tf.nn.softmax(layer1)
y_ = tf.placeholder(tf.float32, [None, y_training.shape[1]])

lam = 0.0000
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


'''------------------------------------------------------------------------------------------------------'''
'''                                       Part 6                                                         '''
'''------------------------------------------------------------------------------------------------------'''
print 'Executing Part 6------------------------------------'
theta = sess.run(W0)
print theta
if not os.path.exists("part6_data.txt"):
    theta_list = []
    for i in range(0, theta.shape[0]):
        diff = theta[i, 0] - theta[i, 1]
        print diff
        heapq.heappush(theta_list, (abs(diff), keywords[i]))
        
    f = open("part6_data.txt", "w")
    for i in heapq.nlargest(100, theta_list):
        print i
        f.write(str(i) + '\n')
    f.close()

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
plt.savefig("part4.png")
