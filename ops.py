import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers 


################################################################################
###################### Standard network layers #################################
################################################################################


def linear(x, out_size, **kwargs):
    return tf.layers.dense(x, out_size, **kwargs)
    
def feedforward(x, layers = [128], **kwargs):
    """
    Concatenates a number of linear layers.
    """
    fc = x
    for i, layer in enumerate(layers):
        fc = linear(fc, layer, name='l_' + str(i), **kwargs)
    
    out = fc

    return out


def conv2d(*args, **kwargs):
    return tf.layers.conv2d(*args, **kwargs)


################################################################################
######################## Set network layers ####################################
################################################################################

def equiv_submax(x,
                 mask=None,
                 name='submax'):
    """
    The 'equivariant' transformation used in the "Deep Sets" paper. N.B: In the
    paper it is combined with a linear transformation (and is actually a special
    case of the more general 'context-concatenation' equivariant layer) but we
    find it more useful to use this as a separate layer.
    
    It is assumed that x is of shape .... x N x C where N is the number of
    elements and C is the number of channels. 
    """
    
    out = x - pool(x, mask, keepdims=True)
    
    return out


def self_attn_qkv(x,
                  key_size,
                  value_size=None,
                  num_heads=1,
                  mask=None,
                  reuse=None,
                  name='self_attn_qkv'):
    """
    Adapted from tensor2tensor library.
    
    It is assumed that x is of shape .... x N x C where N is the number of
    elements and C is the number of channels. 
    
    TODO: add initializer, and activation functions
    """
    value_size = key_size if value_size is None else value_size
    
    with tf.variable_scope(name):
      q = linear(x, key_size, use_bias=False, reuse=reuse, name='query')
      k = linear(x, key_size, use_bias=False, reuse=reuse, name='key')
      v = linear(x, value_size, use_bias=False, reuse=reuse, name='value')
      
    def split_heads(x, num_heads):
        return reshape_range(x, [num_heads, -1])
    
    if num_heads != 1:
      q = split_heads(q, num_heads)
      k = split_heads(k, num_heads)
      v = split_heads(v, num_heads)
      
      if mask is not None:
          mask = tf.expand_dims(mask, -1) #Will be the same mask for every head
      
      key_depth_per_head = key_size // num_heads
      q *= key_depth_per_head**-0.5
      
    def dot_product_attn(q, k, v, mask=None):
        logits = tf.matmul(q, k, transpose_b=True) # ... x l_q x l_k
        if mask is not None:
            bias = (1-mask)*10e9
            logits = logits + bias
        weights = tf.nn.softmax(logits)
        return tf.matmul(weights, v)
    
    v_out = dot_product_attn(q, k, v, mask)
    
    def join_heads(x):
        return reshape_range(x, [value_size], -2, 2)
        
    if num_heads != 1:
        v_out = join_heads(v_out)
      
    out = v_out
    
    return out


def pool(x,
         mask=None,
         keepdims=False,
         pool_type='max'):
    """
    Applies some pooling function along the penultimate dimension, and applies a
    mask where appropriate.
    
    It is assumed that x is of shape .... x N x C where N is the number of
    elements and C is the number of channels. 
    """

    if pool_type == 'max':
        if mask is not None:
            x = x * mask - (1.0 - mask)*10e9
        out = tf.reduce_max(x, -2, keepdims)
        
    elif pool_type == 'sum':
        if mask is not None:
            x = x * mask
        out = tf.reduce_sum(x, -2, keepdims)
            
    elif pool_type == 'mean':
        if mask is not None:
            x = x * mask
            x_sum = tf.reduce_sum(x, -2, keepdims)
            out = x_sum / tf.reduce_sum(mask, -2, keepdims)
        else:
            out = tf.reduce_mean(x, -2, keepdims)

    return out
    


################################################################################
########################### General utils ######################################
################################################################################


def shape_list(x):
    """
    Taken from tensor2tensor library.
    
    Return list of dims, statically where possible.
    """
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def reshape_range(x, new_shape, dim_start=-1, num_dims=1):
    """
    Split's the dim_start'th dimension into a new set of dimensions of shape
    new_shape.
    A range of dims can be reshaped at once by changing num_dims
    """
    in_shape = shape_list(x)
    dim_start = dim_start % len(in_shape)
    start_dims = in_shape[:dim_start]
    end_dims = in_shape[dim_start+num_dims:]
    
    out_shape = start_dims + new_shape + end_dims
    
    out = tf.reshape(x, out_shape)
    
    return out
    

def get_mask(x):
    """
    Returns a matrix with values set to 1 where elements aren't padding
    Assumes input is of the form [...] x C, and that empy inputs are all 0 hence
    we return a matrix of shape [...] x 1 with 0's in locations where last
    dimension is all 0, and 1 elsewhere. (We keep dim for broadcasting).
    """
    
    emb_sum = tf.reduce_sum(tf.abs(x), axis=-1, keep_dims=True)
    mask = 1.0 - tf.to_float(tf.equal(emb_sum, 0.0))
    return tf.stop_gradient(mask)
    
    
def combine_weights(in_list):
    """
    Returns a 1D tensor of the input list of (nested lists of) tensors, useful
    for doing things like comparing current weights with old weights for EWC.
    
    1.) For all elements in input list, (ln 3)
          if a list combine it recursively 
          else leave it alone
    2.) From resulting list, get all non-none elements and flatten them (ln 2)
    3.) If resulting list is empty return None (ln 1)
          else return concatenation of list
    ( All on one line :) )
    """
    
    return (lambda x: None if not x else tf.concat(x, axis=0)) (
        [ tf.reshape(x, [-1]) for x in
        [ combine_weights(x) if isinstance(x, list) else x for x in in_list ]
        if x is not None])
        
        
def position_preserving_embedding(x, out_size, num_preserve=4, **kwargs):
    """
    Returns the concatenation of the first n=num_preserve channels of the input
    with a linear transformation of the rest of the channels. I.e. performs a
    linear transformation preserving the first n channels.
    """
    n = num_preserve
    return tf.concat([x[..., :n], linear(x[..., n:], out_size - n)], axis=-1)
    
################################################################################
############################ Legacy code #######################################
################################################################################

def __linear(x,
           out_size,
           initializer=tf.contrib.layers.xavier_initializer(),
           bias_start=0.0,
           activation_fn=None,
           name='linear'):
    """
    Applies a linear transformation to input and returns the result.
    Assumes input is of the form ... x C where C is the number of channels.
    TODO: Use tf.layers.dense instead
    """
    
    # Get weights and bias
    in_size = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable('w', [in_size, out_size], tf.float32,
            initializer=initializer)
        b = tf.get_variable('b', [out_size],
            initializer=tf.constant_initializer(bias_start))

    # Apply weights and biases
    out = tf.nn.bias_add(tf.matmul(x, w), b)

    if activation_fn != None:
        out = activation_fn(out)
      
    return out
    
def ___combine_weights(in_list):
    # Returns a 1D tensor of the input list of tensors
    
    # 1.) For all elements in input list, if a list, combine it (via recursion)
    # 2.) From resulting list, get all non-none elements and flatten them
    # 3.) If resulting list is empty return None
    #       else return concatenation of list
    
    # Method 1
    #new_list = [ combine_weights(x) if type(x) is list else tf.reshape(x, [-1]) for x in in_list if x is not None ]
    #return None if all(x is None for x in new_list) else tf.concat( [ x for x in new_list if x is not None ], axis=0)
    
    # Method 2
    #return map(lambda x: tf.concat(x, axis=0), filter(None, ([combine_weights(x) if isinstance(x, list) else tf.reshape(x, [-1]) for x in in_list if x is not None],)))
    
    # Method 3
    #elements = [ tf.reshape(x, [-1]) for x in  [ combine_weights(x)
    #    if isinstance(x, list) else x for x in in_list ] if x is not None]
    #return None if not elements else tf.concat(elements, axis=0)
    
    # Method 4
    return (lambda x: None if not x else tf.concat(x, axis=0)) ([ tf.reshape(x, [-1]) for x in  [ combine_weights(x) if isinstance(x, list) else x for x in in_list ] if x is not None])
    
    
