# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:02:53 2020

@author: 29071
"""
import tensorflow as tf

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

    resadded = tf.add(out_33,resadd)
                
    return tf.nn.relu(resadded)       

################### use tf.control_dependencies to update moving mean and moving var ########### 
#updata_ops_main = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#with tf.control_dependencies(updata_ops_main):
#    train_step_main = optimizer.minimize(losses)        
##############################################################################################

################### When defining tf.train.Saver, have to add 'ExponentialMovingAverage' manually.############################
#var_list = tf.trainable_variables()
#g_list = tf.global_variables()
#bn_moving_vars = [g for g in g_list if 'ExponentialMovingAverage' in g.name]
#var_list += bn_moving_vars
#saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
##############################################################################################