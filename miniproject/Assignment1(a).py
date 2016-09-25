
# coding: utf-8

# In[16]:

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# In[17]:

caffe.set_device(0)
caffe.set_mode_gpu()



# In[18]:

net = caffe.Net('../models/bvlc_alexnet/1_deploy.prototxt',
                '../models/bvlc_alexnet/bvlc_alexnet.caffemodel',
                caffe.TEST)


# In[19]:

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)


# In[20]:

net.blobs['data'].reshape(1,3,227,227)


# In[21]:
#Train

accuracy = 0
n=180
features=[]
labels=[]
#class_set_car = set([817,656,609,656,436,511])
class_set_bicycle = set([870,671,444])
#class_set_airline = set([404])
for i in xrange(1,141):
    name = "../examples/images/Train_svm/"
    im = caffe.io.load_image(name+str(i)+'.jpg')
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net.forward()
    cls = out['fc7']
    #print len(cls[0])
    #labels = np.loadtxt("../data/ilsvrc12/synset_words.txt", str, delimiter='\t')
    #top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-4:-1]
    #print labels[top_k]

    #if cls in class_set_bicycle:
    #    accuracy+=1
#print float(accuracy)/n * 100
    #rint out['fc7'][0]
    lst_feature = list(out['fc7'][0])
    #print lst
    features.append(lst_feature)
    #print (out['fc7'][0].shape)
    #print features
    if i<=80:
        labels.append(1)
    else:
        labels.append(2)

#print len(features)
#print labels


# In[25]:

model=LogisticRegression()
model.fit(features,labels)


# In[26]:
#Test

test_features=[]
test_labels=[]
for i in xrange(1,26):
    name = "../examples/images/Test_svm/"
    im = caffe.io.load_image(name+str(i)+'.jpg')
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net.forward()
    cls = out['fc7']
    lst_feature = list(out['fc7'][0])
    test_features.append(lst_feature)
    if i<=10:
        test_labels.append(1)
    else:
        test_labels.append(2)

result = model.predict(test_features)


# In[27]:

accuracy=0
for i in zip(result,test_labels):
    accuracy+= (i[0]==i[1])
print float(accuracy)/len(result) * 100




