# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:21:28 2020

@author: pjjj
"""
import tensorflow as tf

initializer = tf.contrib.layers.variance_scaling_initializer()

def bn_layer(x,training,name='BatchNorm',moving_decay=0.9,eps=1e-5):
    # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
    shape = x.shape
    assert len(shape) in [2,4]
#    param_shape = shape[-1]
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
    
def block(x,chanel,training):
    resadd = x
    kernel_1 = tf.Variable(initializer([3, 3, chanel, chanel], dtype=tf.float32))
    out_33 = tf.nn.conv2d(x, kernel_1, [1, 1, 1, 1], padding='SAME')
    out_33 = bn_layer(out_33, training=training)

    kernel_2 = tf.Variable(initializer([1, 3, chanel, chanel], dtype=tf.float32))
    out_13 = tf.nn.conv2d(x, kernel_2, [1, 1, 1, 1], padding='SAME')
    out_13 = bn_layer(out_13,training=training)

    kernel_3 = tf.Variable(initializer([3, 1, chanel, chanel], dtype=tf.float32))
    out_31 = tf.nn.conv2d(x, kernel_3, [1, 1, 1, 1], padding='SAME')
    out_31 = bn_layer(out_31,training=training)
    
    out = tf.add(tf.add(out_33, out_13), out_31)    
    out = tf.nn.relu(out)
   
    kernel_1 = tf.Variable(initializer([3, 3, chanel, chanel], dtype=tf.float32))
    out_33 = tf.nn.conv2d(out, kernel_1, [1, 1, 1, 1], padding='SAME')
    out_33 = bn_layer(out_33 , training=training)

    kernel_2 = tf.Variable(initializer([1, 3, chanel, chanel], dtype=tf.float32))
    out_13 = tf.nn.conv2d(out, kernel_2, [1, 1, 1, 1], padding='SAME')
    out_13 = bn_layer(out_13,training=training)

    kernel_3 = tf.Variable(initializer([3, 1, chanel, chanel], dtype=tf.float32))
    out_31 = tf.nn.conv2d(out, kernel_3, [1, 1, 1, 1], padding='SAME')
    out_31 = bn_layer(out_31 ,training=training)

    out = tf.add(tf.add(out_33, out_13), out_31)    
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