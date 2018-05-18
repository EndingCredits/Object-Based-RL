import numpy as np
import tensorflow as tf
#from ops import linear, conv2d, flatten
from ops import linear, conv2d, equiv_submax, pool, get_mask


def deepmind_CNN(state, output_size=128):

    initializer = tf.truncated_normal_initializer(0, 0.1) #seed=seed
    activation_fn = tf.nn.relu
    
    state = tf.transpose(state, perm=[0, 2, 3, 1])
    state = tf.truediv(state, 255.0)
    
    conv1 = conv2d(state, 32, [8, 8], [4, 4],
      kernel_initializer=initializer, activation=activation_fn,
      data_format='channels_last', name='conv1')
    conv2 = conv2d(conv1, 64, [4, 4], [2, 2],
      kernel_initializer=initializer, activation=activation_fn,
      data_format='channels_last', name='conv2')
    conv3 = conv2d(conv2, 64, [3, 3], [1, 1],
      kernel_initializer=initializer, activation=activation_fn,
      data_format='channels_last', name='conv3')

    shape = conv3.get_shape().as_list()
    conv3_flat = tf.reshape(conv3, [-1, reduce(lambda x, y: x * y, shape[1:])])

    out = linear(conv3_flat, output_size,
      activation=tf.identity, name='value_hid')

    # Returns the network output
    return out


def feedforward(x, layers = [128], **kwargs):
    fc = x
    for i, layer in enumerate(layers):
        fc = linear(fc, layer, name='l_' + str(i), **kwargs)
    
    out = fc

    return out
    

def feedforward_network(state):
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    out = feedforward(state, [128]*2,
        activation=activation_fn, kernel_initializer=initializer)

    # Returns the network output
    return out


def object_embedding_network(state):
    mask = get_mask(state)
    net = embedding_network(state, mask)
    return net
    
    
def embedding_network(state,
                      mask=None,
                      skip_style=False,
                      d_e = [[128]*2]*2,
                      d_o = [128]*3):
                      
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    # Build graph:
    initial_elems = state

    # Embedding Part
    el = initial_elems
    for i, block in enumerate(d_e):
        if skip_style:
            # Use original input as input for block
            el = initial_elems
        for j, layer in enumerate(block):
            el = linear(el, layer, name='l' + str(i) + '_'  + str(j),
                        kernel_initializer=initializer)
            if j==0 and not i==0: # Add in context
                el = el + linear(c, layer, name='l' + str(i) + '_c',
                                 kernel_initializer=initializer)
            el = activation_fn(el)

        c = pool(el, mask, keepdims=True) # pool to get context for next block
    
    # Fully connected part
    fc = tf.squeeze(c, axis=-2)
    for i, layer in enumerate(d_o):
        fc = linear(fc, layer, activation=activation_fn, 
                    kernel_initializer=initializer, name='lO_' + str(i))
    
    # Returns the network output
    return fc
    
    
def object_embedding_network2(state, l_e=[128]*4, l_o=[128]*3):

    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu
    
    mask = get_mask(state)

    # Embedding Part
    el = state
    
    el = linear(el, l_e[0], name='l' + str(0))
    for i, layer in enumerate(l_e[1:]):
        el = equiv_submax(el, mask)
        el = linear(el, layer, activation=activation_fn, name='l' + str(i+1),
                    kernel_initializer=initializer )
        
    c = pool(el, mask)
    
    # Fully connected part
    fc = c
    for i, layer in enumerate(l_o):
        fc = linear(fc, layer, activation=activation_fn, 
                    kernel_initializer=initializer, name='lO_' + str(i))
    
    return fc
