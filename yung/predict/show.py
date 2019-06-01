# coding=UTF-8
import cv2
import matplotlib.pyplot as plt
import caffe
import numpy as np
MODEL_FILE = 'lenet.prototxt' 
PRETRAINED = 'lenet_iter_10000.caffemodel'
path = ""
context = "context.txt"
data = []
label = []
predict = []
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED) 
with open(path+context) as file:
    for line in file:
        text = line.split()
        print(text[0])
        pic = cv2.imread(path+text[0])
        pic = cv2.resize(pic, (28, 28), interpolation=cv2.INTER_CUBIC)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(pic,150,255,cv2.THRESH_BINARY_INV)
        data.append(thresh)
        label.append(int(text[1]))
        # image1=cv2.cvtColor(thresh,cv2.COLOR_BGR2RGB)
        # input_image=thresh/255.
        # input_image = np.transpose(thresh,(2,0,1))
        # print input_image
        input_image = np.reshape(thresh, (28, 28, -1)) 
        input_image = input_image/255.
        prediction = net.predict([input_image], oversample = False)
        print 'predicted class:', prediction[0].argmax()
        predict.append(prediction[0].argmax())


for i in range(len(predict)):
    plt.subplot(3,3,i+1),plt.imshow(data[i],'gray')
    plt.title(str(predict[i]))
    plt.xticks([]),plt.yticks([])
plt.show()
