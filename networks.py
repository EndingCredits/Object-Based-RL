import numpy as np
import tensorflow as tf
from ops import * #sorry


def deepmind_CNN(state,
                 output_size=128):

    # Keeping track of which initilizer and activation funcs are used
    initializer = tf.truncated_normal_initializer(0, 0.1) #seed=seed
    activation_fn = tf.nn.relu
    
    # reshape to channels last
    state = tf.transpose(state, perm=[0, 2, 3, 1])
    
    # Mormalise from raw values to range 0->1
    state = tf.truediv(state, 255.0)
    
    # Convolutional part
    conv1 = conv2d(state, 32, [8, 8], [4, 4],
      kernel_initializer=initializer, activation=activation_fn,
      data_format='channels_last', name='conv1')
    conv2 = conv2d(conv1, 64, [4, 4], [2, 2],
      kernel_initializer=initializer, activation=activation_fn,
      data_format='channels_last', name='conv2')
    conv3 = conv2d(conv2, 64, [3, 3], [1, 1],
      kernel_initializer=initializer, activation=activation_fn,
      data_format='channels_last', name='conv3')

    # Flatten final layer
    shape = conv3.get_shape().as_list()
    flat_shape = 1
    for dim in shape[1:]:
        flat_shape *= dim
    conv3_flat = tf.reshape(conv3, [-1, flat_shape])

    # Apply a final linear layer
    out = linear(conv3_flat, output_size,
      activation=tf.identity, name='value_hid')

    # Returns the network output
    return out
    

def fully_connected_network(state,
                            layers=[128]*2):
    """
    Simple FC network.
    """
    # Keeping track of which initilizer and activation funcs are used
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    # Stack layers
    out = feedforward(state, layers,
        activation=activation_fn, kernel_initializer=initializer)

    # Returns the network output
    return out


    
def object_embedding_network(state,
                             mask=None,
                             skip_style=False,
                             embedding_layers=[[128]*2]*2,
                             output_layers=[128]*3):
    """
    The original embedding network we used, which is different to the one we
    used for our paper.
    
    Rather than using equivariant 'submax' layers this uses a 'context
    concatenation' approach, where the context (given by the max_pool) is
    concatenated with each element at the input to each block of layers.
    
    Rough diagram:
    
        x_i           Single element
      ___|___
     |       |
     |--128--|        1st 'block'
     |--128--|
     |_______|
      ___|___        
     |       |
     |    max_pool    
     |       |        Equivariant part
  f(x_i) o max_j x_j
      ___|___
     |       |
     |--128--|        2nd block
     |--128--|
     |_______|
         |
        ...
        
    This is then pooled to get a single vector representation of all objects and
    used as input for a final 'task' network.
    """
    
    
    # Keeping track of which initilizer and activation funcs are used
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    # Rename for brevity
    d_e = embedding_layers ; l_o = output_layers
    
    
    if mask is None:
       mask = get_mask(state) # Get mask directly from state

    # Build graph:
    initial_elems = state
    
    # Embedding Part:
    ##################################
    el = initial_elems
    for i, block in enumerate(d_e):
        if skip_style:
            # If skip-style we use the original elements for our input
            el = initial_elems
        for j, layer in enumerate(block):
            el = linear(el, layer, name='l' + str(i) + '_'  + str(j),
                        kernel_initializer=initializer)
            if j==0 and not i==0: # If start of the next block we add in context
                el = el + linear(c, layer, name='l' + str(i) + '_c',
                                 kernel_initializer=initializer)
            el = activation_fn(el)

        c = pool(el, mask, keepdims=True) # pool to get context
        
    ##################################
        
        
    # Fully connected (task) part:
    ##################################
    fc = tf.squeeze(c, axis=-2)
    for i, layer in enumerate(l_o):
        fc = linear(fc, layer, activation=activation_fn, 
                    kernel_initializer=initializer, name='lO_' + str(i))
    ##################################
    
    out = fc
    
    # Returns the network output
    return out
    
    
    
    
def object_embedding_network2(state,
                              mask=None,
                              embedding_layers=[128]*4,
                              output_layers=[128]*3,
                              use_equivariant=True):

    """
    The OEN used for our paper. It is broadly the same architecture used in
    "Deep Sets", but with different activations and layer sizes.
    
    It consists of a number of linear element-wise transformations interspersed
    with 'submax' (i.e. f(x_i) = x_i - max_j{x_j}) equivariant transformations.
    
    This is then pooled to get a single vector representation of all objects and
    used as input for a final 'task' network.
    """


    # Keeping track of which initilizer and activation funcs are used
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu
    
    l_e = embedding_layers ; l_o = output_layers
    
    if mask is None:
       mask = get_mask(state) # Get mask directly from state
    
    # Embedding Part:
    ##################################
    el = state
    
    # Do the first layer without equivariant transform
    el = linear(el, l_e[0], name='l' + str(0))
    
    # Do the rest of the embedding layers with 'submax' equivariant transform
    for i, layer in enumerate(l_e[1:]):
        if use_equivariant:
            el = equiv_submax(el, mask)
        el = linear(el, layer, activation=activation_fn, name='l' + str(i+1),
                    kernel_initializer=initializer )
    
    # Pool final elements to get input to task network
    c = pool(el, mask)
    
    ##################################
    
    # Fully connected (task) part:
    ##################################
    
    fc = c
    for i, layer in enumerate(l_o):
        fc = linear(fc, layer, activation=activation_fn, 
                    kernel_initializer=initializer, name='lO_' + str(i))
                    
    ##################################
    
    out = fc
    
    # Returns the network output
    return out
    



def relational_object_network(state,
                              mask=None,):

    """
    A relational network. Currently testing.
    
    Mimics the architecture given in Relational Deep Reinforcement Learning. 
    """
    #TODO: Pick a design that more closely relates to the above architecture


    # Keeping track of which initilizer and activation funcs are used
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu
    
    if mask is None:
       mask = get_mask(state) # Get mask directly from state
    
    # Embedding Part:
    ##################################
    el = state
    
    # Do the first layer without equivariant transform
    el = position_preserving_embedding(el, 128, name='initial_embedding')
    
    # Add a number of attention blocks
    for i in range(2):
        r = None if i == 0 else True
        el_ = self_attn_qkv(el, 64, 64, num_heads=4, mask=mask, reuse=r)
        el_ = linear(el_, 128, activation=activation_fn, name='l1',
                    kernel_initializer=initializer, reuse=r)
        el_ = linear(el_, 128, activation=activation_fn, name='l2',
                    kernel_initializer=initializer, reuse=r)
        el = el + el_
    
    # Pool final elements to get input to task network
    c = pool(el, mask)
    
    ##################################
    
    
    # Fully connected (task) part:
    ##################################
    
    l_o = [128]*3
    
    fc = c
    for i, layer in enumerate(l_o):
        fc = linear(fc, layer, activation=activation_fn, 
                    kernel_initializer=initializer, name='lO_' + str(i))
                    
    ##################################
    
    out = fc
    
    # Returns the network output
    return out


def relational_object_network2(state,
                               mask=None,):

    # Keeping track of which initilizer and activation funcs are used
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu
    
    if mask is None:
       mask = get_mask(state) # Get mask directly from state
    
    # Embedding Part:
    ##################################
    el = state
    
    # Calculate weight matrix
    def distance_matrix(x_1, x_2):
        x_1 = tf.expand_dims(x_1, -3)
        x_2 = tf.expand_dims(x_2, -2)
        sq_diff = tf.square(x_1 - x_2)
        return tf.reduce_sum(sq_diff, -1)
    
    positions = el[..., :2]
    dists = distance_matrix(positions, positions) + 1e-6
    logits = tf.math.reciprocal(dists)
    logits = logits + (1-mask)*10e9
    weights = tf.nn.softmax(logits)
    
    
    # Do the first layer without equivariant transform
    el = position_preserving_embedding(el, 128, name='initial_embedding')
    
    # Add a number of attention blocks
    def weighted_attention(weights, v):
        return tf.matmul(weights, v)
    
    for i in range(2):
        v = linear(el, 128, activation=activation_fn, name='l'+str(i)+'.1',
                    kernel_initializer=initializer)
        el_ = weighted_attention(weights, v)
        el_ = linear(el_, 128, activation=activation_fn, name='l'+str(i)+'.2',
                    kernel_initializer=initializer)
        el = el + el_
    
    # Pool final elements to get input to task network
    c = pool(el, mask)
    
    ##################################
    
    
    # Fully connected (task) part:
    ##################################
    
    l_o = [128]*3
    
    fc = c
    for i, layer in enumerate(l_o):
        fc = linear(fc, layer, activation=activation_fn, 
                    kernel_initializer=initializer, name='lO_' + str(i))
                    
    ##################################
    
    out = fc
    
    # Returns the network output
    return out
    
