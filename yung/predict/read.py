import caffe

import numpy as np

import cv2
import os
import sys
np.set_printoptions(threshold='nan')
caffe.set_mode_gpu()


MODEL_FILE = 'lenet.prototxt'

PRETRAIN_FILE = 'lenet_iter_10000.caffemodel'


params_txt = 'weight.txt'
# pf = open(params_txt, 'w')


net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)
# # img_gray = cv2.imread('0nb.jpg', cv2.IMREAD_GRAYSCALE)
pic = cv2.imread('0nb.jpg')
pic = cv2.resize(pic, (28, 28), interpolation=cv2.INTER_CUBIC)
img_gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
img_gray = img_gray/255.
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_raw_scale('data', 1)
net.blobs['data'].reshape(1, 1, 28, 28)
print img_gray.shape
net.blobs['data'].data[...] = transformer.preprocess('data', img_gray)
net.forward() 

with open('weight/'+'data'+'.txt', 'w') as pf:
    weight = net.blobs['data'].data
    pf.write('\n')
    pf.write('\n' + 'data' + '_weight: '+ str(weight.shape)+'\n\n')
        
    weight.shape = (-1, 1)
    for w in weight:
        pf.write('%f, ' % w)       
    # pf.write('\n\n' + 'data' + '_bias:\n\n')       
    # bias.shape = (-1, 1)
    # for b in bias:
    #     pf.write('%f, ' % b)
    # pf.write('\n\n')
for param_name in net.params.keys():
    
    weight = net.params[param_name][0].data
    
    bias = net.params[param_name][1].data
    with open('weight/'+param_name+'.txt', 'w') as pf:
       
        # pf.write( str(net.blobs[param_name].data))
        pf.write('\n')

        
        pf.write('\n' + param_name + '_weight:'+ str(weight.shape)+' \n\n')
        
        weight.shape = (-1, 1)

        for w in weight:
            pf.write('%f, ' % w)

        
        pf.write('\n\n' + param_name + '_bias:\n\n')
        
        bias.shape = (-1, 1)
        for b in bias:
            pf.write('%f, ' % b)

        pf.write('\n\n')

for param_name in net.blobs.keys():
    
    weight = net.blobs[param_name].data
    
    # bias = net.blobs[param_name][1].data
    with open('weight/cls_'+param_name+'.txt', 'w') as pf:
       
        # pf.write( str(net.blobs[param_name].data))
        pf.write('\n')

        
        pf.write('\n' + param_name + '_weight:'+ str(weight.shape)+' \n\n')
        
        weight.shape = (-1, 1)

        for w in weight:
            pf.write('%f, ' % w)

        
        # pf.write('\n\n' + param_name + '_bias:\n\n')
        
        # bias.shape = (-1, 1)
        # for b in bias:
        #     pf.write('%f, ' % b)

        # pf.write('\n\n')      
    


