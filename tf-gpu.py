#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
#查看版本号
tf.__version__
#查看安装路径
tf.__path__


# In[12]:


#Creates a graph.
g = tf.Graph()
with g.as_default():
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    
#Creates a session with log_device_placement set to True.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config, graph=g)

#Runs the op.
print (sess.run(c))
sess.close()


# In[ ]:




