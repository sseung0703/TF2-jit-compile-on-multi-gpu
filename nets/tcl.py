import tensorflow as tf

def arg_scope(func):
    def func_with_args(self, *args, **kwargs):
        if hasattr(self, 'pre_defined'):
            for k in self.pre_defined.keys():
                if k not in kwargs.keys():
                    kwargs[k] = self.pre_defined[k]
        return func(self, *args, **kwargs)
    return func_with_args

class Conv2d(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, kernel_size, num_outputs, strides = 1, dilations = 1, padding = 'SAME',
                 kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 2., mode='fan_out'),
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = None,
                 name = 'conv',
                 trainable = True,
                 **kwargs):
        super(Conv2d, self).__init__(name = name, trainable = trainable, **kwargs)
        
        self.kernel_size = kernel_size
        self.num_outputs = num_outputs
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn
        
    def build(self, input_shape):
        super(Conv2d, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = self.kernel_size + [input_shape[-1], self.num_outputs],
                                      initializer=self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,1,1,self.num_outputs],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)
        self.ori_shape = self.kernel.shape

    def call(self, input):
        kernel = self.kernel

        conv = tf.nn.conv2d(input, kernel, self.strides, self.padding,
                            dilations=self.dilations, name=None)
        if self.use_biases:
            conv += self.biases

        if self.activation_fn:
            conv = self.activation_fn(conv)

        return conv

class DepthwiseConv2d(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, kernel_size, multiplier = 1, strides = [1,1,1,1], dilations = [1,1], padding = 'SAME',
                 kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 2., mode='fan_in'),
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = None,
                 name = 'conv',
                 trainable = True,
                 **kwargs):
        super(DepthwiseConv2d, self).__init__(name = name, trainable = trainable, **kwargs)
        
        self.kernel_size = kernel_size
        self.strides = strides if isinstance(strides, list) else [1, strides, strides, 1]
        self.padding = padding
        self.dilations = dilations if isinstance(dilations, list) else [dilations, dilations]
        self.multiplier = multiplier
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn

    def build(self, input_shape):
        super(DepthwiseConv2d, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = self.kernel_size + [input_shape[-1], self.multiplier],
                                      initializer=self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,1,1, input_shape[-1]*self.multiplier],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)
        self.ori_shape = self.kernel.shape

    def call(self, input):
        kernel = self.kernel
        conv = tf.nn.depthwise_conv2d(input, kernel, strides = self.strides, padding = self.padding, dilations=self.dilations)
        if self.use_biases:
            conv += self.biases
        if self.activation_fn:
            conv = self.activation_fn(conv)
        return conv

class FC(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, num_outputs, 
                 kernel_initializer = tf.keras.initializers.random_normal(stddev = 1e-2),
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = None,
                 name = 'fc',
                 trainable = True, **kwargs):
        super(FC, self).__init__(name = name, trainable = trainable, **kwargs)
        self.num_outputs = num_outputs
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn
        
    def build(self, input_shape):
        super(FC, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = [int(input_shape[-1]), self.num_outputs],
                                      initializer=self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,self.num_outputs],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)
        self.ori_shape = self.kernel.shape
    def call(self, input):
        kernel = self.kernel

        fc = tf.matmul(input, kernel)
        if self.use_biases:
            fc += self.biases
        if self.activation_fn:
            fc = self.activation_fn(fc)

        return fc

class BatchNorm(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, param_initializers = None,
                       scale = True,
                       center = True,
                       alpha = 0.9,
                       epsilon = 1e-5,
                       activation_fn = None,
                       name = 'bn',
                       trainable = True,
                       **kwargs):
        super(BatchNorm, self).__init__(name = name, trainable = trainable, **kwargs)
        if param_initializers == None:
            param_initializers = {}
        if not(param_initializers.get('moving_mean')):
            param_initializers['moving_mean'] = tf.keras.initializers.Zeros()
        if not(param_initializers.get('moving_variance')):
            param_initializers['moving_variance'] = tf.keras.initializers.Ones()
        if not(param_initializers.get('gamma')) and scale:
            param_initializers['gamma'] = tf.keras.initializers.Ones()
        if not(param_initializers.get('beta')) and center:
            param_initializers['beta'] = tf.keras.initializers.Zeros()
        
        self.param_initializers = param_initializers
        self.scale = scale
        self.center = center
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_fn = activation_fn

    def build(self, input_shape):
        super(BatchNorm, self).build(input_shape)
        self.moving_mean = self.add_weight(name  = 'moving_mean', trainable = False,
                                      shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                      initializer=self.param_initializers['moving_mean'],
                                      aggregation=tf.VariableAggregation.MEAN,
                                      )
        self.moving_variance = self.add_weight(name  = 'moving_variance', trainable = False,
                                      shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                      initializer=self.param_initializers['moving_variance'],
                                      aggregation=tf.VariableAggregation.MEAN,
                                      )
        if self.scale:
            self.gamma = self.add_weight(name  = 'gamma', 
                                         shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                         initializer=self.param_initializers['gamma'],
                                         trainable = self.trainable)
        else:
            self.gamma = 1.
        if self.center:
            self.beta = self.add_weight(name  = 'beta', 
                                        shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                        initializer=self.param_initializers['beta'],
                                        trainable = self.trainable)
        else:
            self.beta = 0.
        self.ori_shape = self.moving_mean.shape[-1]
           
    def EMA(self, variable, value):
        update_delta = (variable - value) * (1-self.alpha)
        variable.assign_sub(update_delta)

    def update(self, update_var):
        mean, var = update_var
        self.EMA(self.moving_mean, mean)
        self.EMA(self.moving_variance, var)

    def call(self, input, training=None):
        if training:
            mean, var = tf.nn.moments(input, list(range(len(input.shape)-1)), keepdims=True)
            self.update_var = [mean, var]
        else:
            mean = self.moving_mean
            var = self.moving_variance

        gamma, beta = self.gamma, self.beta
        bn = tf.nn.batch_normalization(input, mean, var, offset = beta, scale = gamma, variance_epsilon = self.epsilon)

        if self.activation_fn:
            bn = self.activation_fn(bn)

        return bn

