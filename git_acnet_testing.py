# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:29:19 2020

@author: 29071
"""
import tensorflow as tf
import numpy as np
import os # path join
 
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 128               

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar_python_directory = os.path.abspath('cifar-10-batches-py')

train_data_os = os.path.join(cifar_python_directory, 'data_batch_1')
train_data_set = unpickle(train_data_os)
train_data_ = train_data_set[b'data']
train_labels_ = train_data_set[b'labels']
for i in range (2,6):
    train_data_os = os.path.join(cifar_python_directory, 'data_batch_'+str(i))
    print(train_data_os)
    train_data_set = unpickle(train_data_os)
    train_data_ = np.vstack((train_data_,train_data_set[b'data']))
    train_labels_ = train_labels_ + train_data_set[b'labels']
    
test_data_os = os.path.join(cifar_python_directory, 'test_batch')
test_data_set = unpickle(test_data_os)

test_data_ = test_data_set[b'data']
test_labels_ = test_data_set[b'labels']


def _parse_function(pic, label):
    pic = tf.reshape(pic, (3, 32, 32))
    pic = tf.transpose(pic, [1,2,0])
    pic = tf.image.resize_image_with_crop_or_pad(pic,40,40)
    pic = tf.image.random_flip_left_right(pic)  
    pic = tf.random_crop(pic,[32,32,3])
    pic = tf.cast(pic,dtype=tf.float32)
    pic = tf.image.per_image_standardization(pic)
    label = tf.one_hot(label, depth=10, on_value=1.0, off_value=0.0)
    return pic, label

def _parse_function_for_test(pic, label):
    pic = tf.reshape(pic, (3, 32, 32))
    pic = tf.transpose(pic, [1,2,0])
    pic = tf.cast(pic,dtype=tf.float32)
    pic = tf.image.per_image_standardization(pic)
    label = tf.one_hot(label, depth=10, on_value=1.0, off_value=0.0)
    return pic, label

dataset = tf.data.Dataset.from_tensor_slices((train_data_, train_labels_))
dataset = dataset.map(_parse_function)
# dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.repeat()
dataset = dataset.batch(batch_size=BATCH_SIZE)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

dataset_test = tf.data.Dataset.from_tensor_slices((test_data_, test_labels_))
dataset_test = dataset_test.map(_parse_function_for_test)
# dataset = dataset.shuffle(buffer_size=10000)
dataset_test = dataset_test.repeat()
dataset_test = dataset_test.batch(batch_size=BATCH_SIZE)
iterator_test = dataset_test.make_initializable_iterator()
next_element_test = iterator_test.get_next()
##########################################
initializer = tf.contrib.layers.variance_scaling_initializer()

def bn_layer(x,training,name='BatchNorm',moving_decay=0.9,eps=1e-5):
    shape = x.shape
    assert len(shape) in [2,4]
    with tf.variable_scope(name):
        
        gamma = tf.Variable(1.,trainable=True, name='gamma')
        beta = tf.Variable(0.,trainable=True, name='beta')
        axes = list(range(len(shape)-1))
        batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')
        ema1 = tf.train.ExponentialMovingAverage(moving_decay)
        ema2 = tf.train.ExponentialMovingAverage(moving_decay)
        def mean_var_with_update():
            ema_apply_op1 = ema1.apply([batch_mean])
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_apply_op1)
            ema_apply_op2 = ema2.apply([batch_var])
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_apply_op2)
            
            with tf.control_dependencies([ema_apply_op1,ema_apply_op2]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.equal(training,True),mean_var_with_update,
                lambda:(ema1.average(batch_mean),ema2.average(batch_var)))

        inv = tf.math.rsqrt(var + eps)
        inv *= gamma
        bn = x * tf.cast(inv, x.dtype) + tf.cast(beta - mean * inv, x.dtype)
        return bn

def bn_layer_new(x,training,name='BatchNorm',moving_decay=0.9,eps=1e-5):
    shape = x.shape
    assert len(shape) in [2,4]

    with tf.variable_scope(name):
        
        gamma = tf.Variable(1.,trainable=True, name='gamma')
        beta = tf.Variable(0.,trainable=True, name='beta')

        axes = list(range(len(shape)-1))
        batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')
        ema1 = tf.train.ExponentialMovingAverage(moving_decay)
        ema2 = tf.train.ExponentialMovingAverage(moving_decay)
        def mean_var_with_update():
            ema_apply_op1 = ema1.apply([batch_mean])
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_apply_op1)
            ema_apply_op2 = ema2.apply([batch_var])
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_apply_op2)
            
            with tf.control_dependencies([ema_apply_op1,ema_apply_op2]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.equal(training,True),mean_var_with_update,
                lambda:(ema1.average(batch_mean),ema2.average(batch_var)))

        return mean,var,beta,gamma
def kernel_combine(kernel_33, kernel_13, kernel_31, mean_33, var_33, beta_33, gamma_33, mean_13, var_13, beta_13, gamma_13, mean_31, var_31, beta_31, gamma_31):
    var_33 = tf.math.rsqrt(var_33 + 1e-5)
    var_13 = tf.math.rsqrt(var_13 + 1e-5)
    var_31 = tf.math.rsqrt(var_31 + 1e-5)
    
    zero_matrix = tf.zeros(kernel_13.shape,dtype=tf.float32)
    oko_13 = tf.concat([zero_matrix,kernel_13,zero_matrix],0)
    zero_matrix = tf.zeros(kernel_31.shape,dtype=tf.float32)
    oko_31 = tf.concat([zero_matrix,kernel_31,zero_matrix],1)    
    
        
    k_33 = gamma_33*var_33 * kernel_33 
    k_13 = gamma_13*var_13 * oko_13 
    k_31 = gamma_31*var_31 * oko_31 
    
    b = -mean_33*gamma_33*var_33+beta_33 -mean_13*gamma_13*var_13+beta_13 -mean_31*gamma_31*var_31+beta_31
    sum_up = tf.add(tf.add(k_33,k_13),k_31)
    return sum_up,b

        
def block(x,chanel,training):
    resadd = x
    kernel_1 = tf.Variable(initializer([3, 3, chanel, chanel], dtype=tf.float32))
    kernel_2 = tf.Variable(initializer([1, 3, chanel, chanel], dtype=tf.float32))    
    kernel_3 = tf.Variable(initializer([3, 1, chanel, chanel], dtype=tf.float32))
    mean_33,var_33,beta_33,gamma_33 = bn_layer_new(x , training=training)              
    mean_13,var_13,beta_13,gamma_13 = bn_layer_new(x , training=training)
    mean_31,var_31,beta_31,gamma_31 = bn_layer_new(x , training=training)    
    new_kernel,bias = kernel_combine(kernel_1,kernel_2,kernel_3,mean_33,var_33,beta_33,gamma_33,mean_13,var_13,beta_13,gamma_13,mean_31,var_31,beta_31,gamma_31)                

    out = tf.nn.conv2d(x, new_kernel, [1, 1, 1, 1], padding='SAME') + bias
    out = tf.nn.relu(out)
   
    kernel_1 = tf.Variable(initializer([3, 3, chanel, chanel], dtype=tf.float32))
    kernel_2 = tf.Variable(initializer([1, 3, chanel, chanel], dtype=tf.float32))    
    kernel_3 = tf.Variable(initializer([3, 1, chanel, chanel], dtype=tf.float32))
    mean_33,var_33,beta_33,gamma_33 = bn_layer_new(out , training=training)              
    mean_13,var_13,beta_13,gamma_13 = bn_layer_new(out , training=training)
    mean_31,var_31,beta_31,gamma_31 = bn_layer_new(out , training=training)    
    new_kernel,bias = kernel_combine(kernel_1,kernel_2,kernel_3,mean_33,var_33,beta_33,gamma_33,mean_13,var_13,beta_13,gamma_13,mean_31,var_31,beta_31,gamma_31)                

    out = tf.nn.conv2d(out, new_kernel, [1, 1, 1, 1], padding='SAME') + bias 
    resadded = tf.add(out,resadd)

    return tf.nn.relu(resadded)    

def block_short(x,chanel,training):
    kernel = tf.Variable(initializer([1, 1, int(chanel/2), chanel], dtype=tf.float32))
    out_short_cut = tf.nn.conv2d(x , kernel, [1, 2, 2, 1], padding='SAME')
    resadd = bn_layer(out_short_cut , training=training)                        
    
    kernel_1 = tf.Variable(initializer([3, 3, int(chanel/2), chanel], dtype=tf.float32))
    out_33 = tf.nn.conv2d(x, kernel_1, [1, 2, 2, 1], padding='SAME')
    out_33 = bn_layer(out_33 , training=training)
    
    out = out_33  
    out = tf.nn.relu(out)
   
    kernel_1 = tf.Variable(initializer([3, 3, chanel, chanel], dtype=tf.float32))
    out_33 = tf.nn.conv2d(out, kernel_1, [1, 1, 1, 1], padding='SAME')
    out_33 = bn_layer(out_33 , training=training)

    out = out_33   
    resadded = tf.add(out,resadd)
                
    return tf.nn.relu(resadded)       
##########################################
class ResNet:
    def __init__(self, imgs, sess=None,training=True):
        self.imgs = imgs
        self.convlayers()
        self.probs = tf.nn.softmax(self.my_final,name='y_main_pred')
    def convlayers(self):

        with tf.name_scope('main') as scope:
            with tf.name_scope('conv1_1') as scope:
                kernel = tf.Variable(initializer([3, 3, 3, 16], dtype=tf.float32), name='weights')
                conv = tf.nn.conv2d(self.imgs , kernel, [1, 1, 1, 1], padding='SAME')
                biases = tf.Variable(initializer(shape=[16], dtype=tf.float32),trainable=True, name='biases')
                out = tf.nn.bias_add(conv, biases)
    
            with tf.name_scope('resconv1') as scope:
                for i in range(9):
                    out = block(out, 16, training)
                self.conv1 = out
                
            with tf.name_scope('resconv2') as scope:
                out = block_short(self.conv1, 32, training)
                for i in range(8):
                    out = block(out, 32, training)
                self.conv2 = out

            with tf.name_scope('resconv3') as scope:
                out = block_short(self.conv2, 64, training)
                for i in range(8):
                    out = block(out, 64, training)
                self.conv2 = out
            self.pool = tf.reduce_mean(self.conv2, axis=[1,2])
     
            with tf.name_scope('fc1') as scope:
                shape = int(np.prod(self.pool.get_shape()[1:]))
                
                fc1w = tf.Variable(initializer(shape=[shape, 10],dtype=tf.float32), name='weights')
                fc1b = tf.Variable(initializer(shape=[10], dtype=tf.float32),trainable=True, name='biases')
                pool5_flat = tf.reshape(self.pool, [-1, shape])
                
                self.my_final = tf.nn.bias_add(tf.matmul(pool5_flat , fc1w), fc1b)

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 32, 32, 3])
    training = tf.placeholder_with_default(False, shape=(), name='training')
    label_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 10], name='y_true')  

    global_step = tf.Variable(0,trainable= False)
    boundaries = [32000,48000]
    values = [0.1,0.01,0.001]
    learning_rate = tf.train.piecewise_constant(global_step,boundaries, values)
    resnet = ResNet(imgs, sess, training)
#####################################################################    
    with tf.name_scope("loss"):
        loss_main = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=resnet.my_final,labels= label_batch_placeholder ), name="loss_main")  
        reg_variables_main = tf.trainable_variables(scope = 'main')
        exceptbn_vars = []
        for i in range(len(reg_variables_main)):
            if 'BatchNorm' in reg_variables_main[i].name:
                continue
            else:
                exceptbn_vars.append(reg_variables_main[i])
        reg_term_main = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(scale=1e-4),exceptbn_vars)
        losses_main = loss_main + reg_term_main
        tf.summary.scalar("Loss_main", losses_main)

    with tf.name_scope("train"):
        optimizer_main = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
        output_vars_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main') 
        updata_ops_main = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(updata_ops_main):
            train_step_main = optimizer_main.minimize(losses_main,global_step,var_list = output_vars_main)        

##########################################################
    with tf.name_scope("accuracy"):
        correct_prediction_main = tf.equal(tf.argmax(resnet.probs, 1), tf.argmax(label_batch_placeholder , 1)) 
        accuracy_main = tf.reduce_mean(tf.cast(correct_prediction_main, tf.float32))
        tf.summary.scalar("accuracy_main", accuracy_main)  
    
    summ = tf.summary.merge_all()    
    
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'ExponentialMovingAverage' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
    
    print (5)
    tenboard_dir = './tensorboard/cifar-2020-01-23-01-delete/'
    writer = tf.summary.FileWriter(tenboard_dir)
    writer.add_graph(sess.graph)
    ################################################################
    model_path = './cifar-2020-02-07-01/cifar10.ckpt-2000'
    ######################################################
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())   
        sess.run([iterator.initializer,iterator_test.initializer])
        ######################
        saver.restore(sess,model_path)
        #####################
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)
        for i in range(10):
            image_out_eval, label_batch_one_hot_out_eval = sess.run(next_element_test)
            eval_accuracy1 = sess.run(accuracy_main, feed_dict={imgs: image_out_eval, label_batch_placeholder : label_batch_one_hot_out_eval,training: False})
            print ("eval_accuracy_main:",eval_accuracy1)

        coord.request_stop()
        coord.join(threads)
        sess.close()
    ####################
    


