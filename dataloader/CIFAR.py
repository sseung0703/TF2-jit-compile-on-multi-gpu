import tensorflow as tf
import numpy as np

def build_dataset_providers(args, strategy, test_only = False):
    if args.dataset == 'CIFAR10':
        train_images, train_labels, test_images, test_labels, pre_processing = Cifar10(args)
    if args.dataset == 'CIFAR100':
        train_images, train_labels, test_images, test_labels, pre_processing =  Cifar100(args)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.map(pre_processing(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(args.val_batch_size).cache()
    test_ds = test_ds.with_options(options)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    if test_only:
        return {'test': test_ds, 'num_classes' : int(args.dataset[5:])}
 
    train_ds = [train_images, train_labels]

    train_ds = tf.data.Dataset.from_tensor_slices(tuple(train_ds)).cache()
    train_ds = train_ds.map(pre_processing(is_training = True),  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(100*args.batch_size).batch(args.batch_size)
    train_ds = train_ds.with_options(options)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    datasets = {
        'train': train_ds.repeat( args.train_epoch ),
        'test': test_ds,
    }

    datasets = {k:strategy.experimental_distribute_dataset(datasets[k]) for k in datasets}
    datasets['train_len'] = train_ds.cardinality().numpy()
    datasets['num_classes'] = int(args.dataset[5:])
    args.input_size = [32,32,3]

    print('Datasets are built')
    return datasets

def Cifar10(args):
    from tensorflow.keras.datasets.cifar10 import load_data
    (train_images, train_labels), (val_images, val_labels) = load_data()
    
    def pre_processing(is_training = False):
        def training(image, *argv):
            sz = tf.shape(image)

            image = tf.cast(image, tf.float32)

            image0 = image

            image0 = (image0-np.array([113.9,123.0,125.3]))/np.array([66.7,62.1,63.0])
            image0 = tf.image.random_flip_left_right(image0)
            image0 = tf.pad(image0, [[4,4],[4,4],[0,0]], 'REFLECT')
            image0 = tf.image.random_crop(image0,sz)
            
            return [image0] + [arg for arg in argv]
        
        def inference(image, label):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([113.9,123.0,125.3]))/np.array([66.7,62.1,63.0])
            return image, label
        
        return training if is_training else inference
    return train_images, train_labels, val_images, val_labels, pre_processing

def Cifar100(args):
    from tensorflow.keras.datasets.cifar100 import load_data
    (train_images, train_labels), (val_images, val_labels) = load_data()

    def pre_processing(is_training = False):
        @tf.function
        def training(image, *argv):
            sz = tf.shape(image)

            image = tf.cast(image, tf.float32)
            image0 = image

            image0 = (image0-np.array([112,124,129]))/np.array([70,65,68])
            image0 = tf.image.random_flip_left_right(image0)
            image0 = tf.pad(image0, [[4,4],[4,4],[0,0]], 'REFLECT')
            image0 = tf.image.random_crop(image0,sz)
            
            return [image0] + [arg for arg in argv]

        @tf.function
        def inference(image, label):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([112,124,129]))/np.array([70,65,68])
            return image, label
        
        return training if is_training else inference
    return train_images, train_labels, val_images, val_labels, pre_processing
