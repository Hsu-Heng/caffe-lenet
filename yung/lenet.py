#!/usr/bin/python
# -*- coding: utf-8 -*-
import caffe
import numpy as np

import cv2
import matplotlib as plt
import os
import sys
np.set_printoptions(threshold='nan')


#img = caffe.io.load_image('C:/Users/liang/Desktop/caffe-windows/models/bvlc_alexnet/data/images/dog.jpg', color=False)

# deploy文件
MODEL_FILE = 'C:/Users/user/Desktop/Thesis/vgg1/params1/lenet.prototxt'
# 預先訓練好的caffe模型
PRETRAIN_FILE = 'C:/Users/user/Desktop/Thesis/vgg1/params1/lenet_iter_10000.caffemodel'
#輸入影像 240*240
#img = caffe.io.load_image('C:/Users/liang/Desktop/caffe-windows/models/bvlc_alexnet/data/images/dog.jpg')

# 保存参数的文件
filter1_txt = 'C:/Users/user/Desktop/Thesis/vgg1/params1/filter1.txt'
conv1_txt = 'C:/Users/user/Desktop/Thesis/vgg1/params1/conv1.txt'
data_txt = 'C:/Users/user/Desktop/Thesis/vgg1/params1/data.txt'
bias1_txt = 'C:/Users/user/Desktop/Thesis/vgg1/params1/bais1.txt'
df = open(data_txt, 'w')
pf = open(filter1_txt, 'w')
cf = open(conv1_txt, 'w')
bf = open(bias1_txt, 'w')
filter2_txt = 'C:/Users/user/Desktop/Thesis/vgg1/params1/filter2.txt'
conv2_txt = 'C:/Users/user/Desktop/Thesis/vgg1/params1/conv2.txt'
bias2_txt = 'C:/Users/user/Desktop/Thesis/vgg1/params1/bais2.txt'
pf1 = open(filter2_txt, 'w')
cf1 = open(conv2_txt, 'w')
bf1 = open(bias2_txt, 'w')


pool1_txt = 'C:/Users/user/Desktop/Thesis/vgg1/params1/pool1.txt'
pl = open(pool1_txt, 'w')
pool2_txt = 'C:/Users/user/Desktop/Thesis/vgg1/params1/pool2.txt'
pl1 = open(pool2_txt, 'w')


ip1_txt = 'C:/Users/user/Desktop/Thesis/vgg1/params1/ip1.txt'
fc1 = open(ip1_txt, 'w')
ip2_txt = 'C:/Users/user/Desktop/Thesis/vgg1/params1/ip2.txt'
fc2 = open(ip2_txt, 'w')



softmax_txt = 'C:/Users/user/Desktop/Thesis/vgg1/params1/softmax.txt'
s = open(softmax_txt, 'w')

# 让caffe以测试模式读取网络参数
net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)
img = cv2.imread('1.jpg')
type(img)
img_gray = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
#img = caffe.io.load_image('dog.jpg')
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 1)
net.blobs['data'].reshape(1, 1, 28, 28)
#img = caffe.io.load_image('1.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', img_gray)
net.forward()   #运行测试
print "data-blobs:",net.blobs['data'].data.shape
datablob = net.blobs['data'].data
df.write(str(datablob))
print "conv1_1-blobs:",net.blobs['conv1'].data.shape
print "conv1_2-blobs:",net.blobs['conv2'].data.shape
print "fc8-blobs:",net.blobs['ip2'].data.shape
conv = net.blobs['conv1'].data
conv1 = net.blobs['conv2'].data
pool =net.blobs['pool1'].data
pool1 =net.blobs['pool2'].data
ip = net.blobs['ip1'].data
ip1 = net.blobs['ip2'].data
#relu = net.blobs['relu1'].data
soft = net.blobs['prob'].data
cf.write(str(conv))
cf1.write(str(conv1))
#cf.write(str(conv2))
pl.write(str(pool))
pl1.write(str(pool1))

fc1.write(str(ip))
fc2.write(str(ip1))

s.write(str(soft))

w1=net.params['conv1'][0].data
bias=net.params['conv1'][1].data
pf.write(str(w1))
bf.write(str(bias))

w2=net.params['conv2'][0].data
bias2=net.params['conv2'][1].data
pf1.write(str(w2))
bf1.write(str(bias2))

#查看各层参数规模
for k,v in net.params.items():
	print(k)
	print(str(v[0].data.shape))

#weight=net.params['Conv1'][0].data  #提取参数w
#bias=net.params['Conv1'][1].data  #提取参数b

