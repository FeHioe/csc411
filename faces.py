'''------------------------------------------------------------------------------------------------------'''
'''                                       Python 3.6                                                     '''
'''------------------------------------------------------------------------------------------------------'''
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import hashlib
import fnmatch
import tensorflow as tf

'''------------------------------------------------------------------------------------------------------'''
'''                                           Part 7                                                     '''
'''------------------------------------------------------------------------------------------------------'''   

'''------------------------------------------------------------------------------------------------------'''
'''                                         Get Images                                                   '''
'''------------------------------------------------------------------------------------------------------'''   
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def get_data(actors):
    testfile = urllib.URLopener()
    
    for a in actors:
        name = a.split()[1].lower()
        i = 0
        for line in open("facescrub_actors.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                timeout(testfile.retrieve, (line.split()[4], "uncropped/" + filename), {}, 15)
                i += 1
    
    for a in actors:
        name = a.split()[1].lower()
        i = 0
        for line in open("facescrub_actresses.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 15)
                i += 1
                
'''------------------------------------------------------------------------------------------------------'''
'''                                         Filter Images                                                   '''
'''------------------------------------------------------------------------------------------------------'''      
def preprocess():
    all_images = os.listdir('uncropped');
    
    for image in all_images:
        
        name = image.split('.')[0]
        name = ''.join(i for i in name if not i.isdigit())
        name = name.title()
        
        try:
            filename = open("uncropped/" + image, "rb").read()
        except:
            print("Error reading file")
            
        h = hashlib.sha256()
        h.update(filename)
        hash = h.hexdigest()
        
        for line in open("facescrub_actors.txt"):
            if name in line:
                try:
                    test = line.split()[6]
                    if(test == hash):
                        position = line.split()[5].split(',')
                        y1 = int(position[1])
                        y2 = int(position[3])
                        x1 = int(position[0])
                        x2 = int(position[2])
                        
                        process = imread("uncropped/" + image)
                        bound = process[y1:y2, x1:x2, :]
                        gray = rgb2gray(bound)
                        cropped = imresize(gray, (32, 32))
                        
                        imsave("cropped/" + image, cropped)
                        break
                except:
                    continue
                    
        for line in open("facescrub_actresses.txt"):
            if name in line:
                try:
                    test = line.split()[6]
                    if(test == hash):
                        position = line.split()[5].split(',')
                        y1 = int(position[1])
                        y2 = int(position[3])
                        x1 = int(position[0])
                        x2 = int(position[2])
                        
                        process = imread("uncropped/"+image)
                        bound = process[y1:y2, x1:x2, :]
                        gray = rgb2gray(bound)
                        cropped = imresize(gray, (32, 32))
                        
                        imsave("cropped/" + image, cropped)
                        break
                except:
                    continue

def make_sets(act):
    if not os.path.exists("part7"):
        os.mkdir("part7")
        os.mkdir("part7/training")
        os.mkdir("part7/test")
        os.mkdir("part7/validation")
        
        for i in act:
            actor = i.split()[1].lower()
    
                
            all_images = fnmatch.filter(os.listdir('cropped'), actor.lower()+'*')
            
            random.seed(0)
            shuffle = random.sample(range(0, len(all_images)), 120)
            
            test_set = shuffle[:30]
            training_set = shuffle[30:100]
            validation_set = shuffle[100:120]

            for i in training_set:
                image = imread("cropped/" + all_images[i])
                imsave("part7/training/" + all_images[i], image)
                
            for i in validation_set:
                image = imread("cropped/" + all_images[i])
                imsave("part7/validation/" + all_images[i], image)
            
            for i in test_set:
                image = imread("cropped/"+ all_images[i])
                imsave("part7/test/" + all_images[i], image)


def getArray (str):
    im = imread(str)
    return(np.array([im.flatten()]))
    
def get_whole_set(act, file):
    x = zeros((0,1024))
    y = zeros((0,len(act)))
    for k in range(len(act)):
        counter = 0
        name = act[k].split()[1].lower()
        for fn in os.listdir('./' + file):
            if (name in fn):
                x = vstack((x, getArray(file + "/" + fn)))
                counter += 1
        one_hot = zeros(len(act))
        one_hot[k] = 1
        y = vstack((y, tile(one_hot, (counter,1))))
    return x, y

def part7(actors):
    #parameters used for part 7:
    nhid = 100             # number of hidden units
    alpha = 0.00001
    max_iter = 2000         #plot from 0 to 2000, every 200
    mini_batch_size = 50
    lam = 0.0000
    
    x_test, y_test = get_whole_set(actors, "part7/test/")
    x_val, y_val = get_whole_set(actors, "part7/validation/")
    x_train, y_train = get_whole_set(actors, "part7/training/")
    
    W0 = tf.Variable(np.random.normal(0.0, 0.1, (1024, nhid)).astype(float32) / math.sqrt(1024 * nhid))
    b0 = tf.Variable(np.random.normal(0.0, 0.1, (nhid)).astype(float32) / math.sqrt(nhid))
    
    W1 = tf.Variable(np.random.normal(0.0, 0.1, (nhid, y_train.shape[1])).astype(float32) / math.sqrt(y_train.shape[1] * nhid))
    b1 = tf.Variable(np.random.normal(0.0, 0.1, (y_train.shape[1])).astype(float32) / math.sqrt(y_train.shape[1]))
    
    grad_descent(x_test, y_test, x_val, y_val, x_train, y_train, nhid, alpha, max_iter, mini_batch_size, lam, W0, b0, W1, b1, 7)
    
    x_axis = np.arange(11) * 200
    
    plt.ylim(0,110)
    plt.plot(x_axis, test_performance, label="test")
    plt.plot(x_axis, train_performance, label="training")
    plt.plot(x_axis, val_performance, label="validation")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('Iteration')
    plt.ylabel('Correctness(%)')
    plt.savefig("part7.png")
    
def part9():    
    if not os.path.exists("part9_w0.txt"):
        #parameters used for part 9:
        nhid = 100             # number of hidden units
        alpha = 0.00001
        max_iter = 30000         #plot from 0 to 2000, every 200
        mini_batch_size = 50
        lam = 0.000001
        
        x_test, y_test = get_whole_set(part7_act, "part7/test/")
        x_val, y_val = get_whole_set(part7_act, "part7/validation/")
        x_train, y_train = get_whole_set(part7_act, "part7/training/")
        
        W0 = tf.Variable(np.random.normal(0.0, 0.1, (1024, nhid)).astype(float32)/math.sqrt(1024 * nhid))
        b0 = tf.Variable(np.random.normal(0.0, 0.1, (nhid)).astype(float32)/math.sqrt(nhid))
        
        W1 = tf.Variable(np.random.normal(0.0, 0.1, (nhid, y_train.shape[1])).astype(float32)/math.sqrt(y_train.shape[1] * nhid))
        b1 = tf.Variable(np.random.normal(0.0, 0.1, (y_train.shape[1])).astype(float32)/math.sqrt(y_train.shape[1]))
        
        final_W0, final_W1 = grad_descent(x_test, y_test, x_val, y_val, x_train, y_train, nhid, alpha, max_iter, mini_batch_size, lam, W0, b0, W1, b1, 8)
            
        # Save final_W0 for part 9.
        np.savetxt("part9_w0.txt", final_W0)
        np.savetxt("part9_w1.txt", final_W1)
    
    w0 = np.loadtxt("part9_w0.txt")
    w1 = np.loadtxt("part9_w1.txt")
    highest_units = np.argmax(w1, 0)
    
    for i in range(len(highest_units)):
        print("active unit for actor "+str(i)+" = "+str(highest_units[i]))
        imsave("part9_act"+str(i)+"_unit.jpg", reshape(w0[:, highest_units[i]], (32, 32)), cmap = cm.gray)
'''------------------------------------------------------------------------------------------------------'''
'''                                          Execution                                                   '''
'''------------------------------------------------------------------------------------------------------'''
if __name__ == "__main__":
    actors =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 
    'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    
    if (not os.path.exists("cropped")):
        os.mkdir("cropped");
        get_data(actors)
        preprocess();
        make_sets(actors);
    part7(actors)