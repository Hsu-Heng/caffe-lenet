import caffe

import numpy as np


np.set_printoptions(threshold='nan')


MODEL_FILE = 'mylenet_train_test.prototxt'

PRETRAIN_FILE = 'lenet_iter_10000.caffemodel'


params_txt = 'weight.txt'
# pf = open(params_txt, 'w')


net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)


for param_name in net.params.keys():
    
    weight = net.params[param_name][0].data
    
    bias = net.params[param_name][1].data
    with open('weight/'+param_name+'.txt', 'w') as pf:
        pf.write(param_name)
        pf.write('\n')

        
        pf.write('\n' + param_name + '_weight:\n\n')
        
        weight.shape = (-1, 1)

        for w in weight:
            pf.write('%ff, ' % w)

        
        pf.write('\n\n' + param_name + '_bias:\n\n')
        
        bias.shape = (-1, 1)
        for b in bias:
            pf.write('%ff, ' % b)

        pf.write('\n\n')
        
    


