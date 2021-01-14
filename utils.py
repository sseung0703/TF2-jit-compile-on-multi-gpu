import os, shutil, glob, pickle, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
tf.debugging.set_log_device_placement(False)

from nets import ResNet
import asyncio

def scheduler(args, epoch):
    lr = args.learning_rate
    for dp in args.decay_points:
        if epoch >= dp:
            lr *= args.decay_rate
    return lr

def save_code_and_augments(args):
    if os.path.isdir(os.path.join(args.train_path,'codes')): 
        print ('============================================')
        print ('The folder already is. It will be overwrited')
        print ('============================================')
    else:
        os.mkdir(os.path.join(args.train_path,'codes'))

    for code in glob.glob(args.home_path + '/*.py'):
        shutil.copyfile(code, os.path.join(args.train_path, 'codes', os.path.split(code)[-1]))
    
    with open(os.path.join(args.train_path, 'arguments.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

class Evaluation:
    def __init__(self, args, model, strategy, dataset, loss_object):
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        @tf.function(jit_compile=args.compile)
        def compiled_step(images, labels, training):
            pred = model(images, training = training)
            loss = loss_object(labels, pred)/args.val_batch_size
            return pred, loss

        def eval_step(images, labels, training):
            pred, loss = compiled_step(images, labels, training)
            self.test_loss.update_state(loss)
            self.test_accuracy.update_state(labels, pred)


        @tf.function
        def eval_step_dist(images, labels, training):
            strategy.run(eval_step, args=(images, labels, training))

        self.dataset = dataset
        self.step = eval_step_dist

    def run(self, training):
        for images, labels in self.dataset:
            self.step(images, labels, training)
        loss = self.test_loss.result().numpy()
        acc = self.test_accuracy.result().numpy()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        return acc, loss

def load_model(args, num_class, trained_param = None):
    if 'ResNet' in args.arch:
        arch = int(args.arch.split('-')[1])
        model = ResNet.Model(num_layers = arch, num_class = num_class, name = 'ResNet', trainable = True)

    if trained_param is not None:
        with open(trained_param, 'rb') as f:
            trained = pickle.load(f)
        n = 0
        for k in model.Layers.keys():
            layer = model.Layers[k]
            if 'conv' in k or 'fc' in k:
                kernel = trained[layer.name + '/kernel:0']
                layer.kernel_initializer = tf.constant_initializer(kernel)
                n += 1
                if layer.use_biases:
                    layer.biases_initializer = tf.constant_initializer(trained[layer.name + '/biases:0'])
                    n += 1
                layer.num_outputs = kernel.shape[-1]
                
            elif 'bn' in k:
                moving_mean = trained[layer.name + '/moving_mean:0']
                moving_variance = trained[layer.name + '/moving_variance:0']
                param_initializers = {'moving_mean' : tf.constant_initializer(moving_mean),
                                      'moving_variance': tf.constant_initializer(moving_variance)}
                n += 2
    
                if layer.scale:
                    param_initializers['gamma'] = tf.constant_initializer(trained[layer.name + '/gamma:0'])
                    n += 1
                if layer.center:
                    param_initializers['beta'] = tf.constant_initializer(trained[layer.name + '/beta:0'])
                    n += 1
                layer.param_initializers = param_initializers
        print (n, 'params loaded')
    return model

def build_dataset_providers(args, strategy):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
 
    train_ds = ILSVRC(args, 'train', shuffle = True)
    train_ds = train_ds.map(pre_processing(is_training = True, contrastive = args.Knowledge), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(100*args.batch_size).batch(args.batch_size).map(pre_processing_batched(is_training = True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.with_options(options)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = ILSVRC(args, 'val', shuffle = False)
    test_ds = test_ds.map(pre_processing(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(args.val_batch_size).map(pre_processing_batched(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.with_options(options)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    datasets = {
        'train': train_ds.repeat( args.train_epoch ),
        'test': test_ds
    }

    datasets = {k:strategy.experimental_distribute_dataset(datasets[k]) for k in datasets}
    datasets['train_len'] = train_ds.cardinality().numpy()

    print('Datasets are built')
    return datasets

def save_model(args, model, name):
    params = {}
    for v in model.variables:
        if model.name in v.name:
            params[v.name[len(model.name)+1:]] = v.numpy()
    with open(os.path.join(args.train_path, name + '.pkl'), 'wb') as f:
        pickle.dump(params, f)

def check_complexity(model, args):
    model(np.zeros([1]+args.input_size, dtype=np.float32), training = False)
    total_params = []
    total_flops = []
    for k in model.Layers.keys():
        layer = model.Layers[k]
        if hasattr(layer, 'params'):
            p = layer.params
            if isinstance(p, tf.Tensor):
                p = p.numpy()
            total_params.append(p)
        if hasattr(layer, 'flops'):
            f = layer.flops
            if isinstance(f, tf.Tensor):
                f = f.numpy()
            total_flops.append(f)
    return sum(total_params), sum(total_flops)
