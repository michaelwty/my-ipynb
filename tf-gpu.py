import tensorflow as tf
import os
#查看版本号
tf.__version__
#查看安装路径
tf.__path__

#Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
#Creates a session with log_device_placement set to True.
gpu_no = '0' # or '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#Runs the op.
print (sess.run(c))