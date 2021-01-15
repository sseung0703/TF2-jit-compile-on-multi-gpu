import glob, os
import tensorflow as tf
import numpy as np
from PIL import Image

JPEG_OPT = {'fancy_upscaling': True, 'dct_method': 'INTEGER_ACCURATE'}

def build_dataset_providers(args, strategy, test_only = False):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    test_ds = ILSVRC(args, 'val', shuffle = False)
    test_ds = test_ds.map(pre_processing(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(args.val_batch_size).map(pre_processing_batched(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.with_options(options)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    if test_only:
        return {'test': test_ds, 'num_classes': 1000}
 

    train_ds = ILSVRC(args, 'train', shuffle = True)
    train_ds = train_ds.map(pre_processing(is_training = True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(100*args.batch_size).batch(args.batch_size).map(pre_processing_batched(is_training = True), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.with_options(options)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
   
    datasets = {
        'train': train_ds.repeat( args.train_epoch ),
        'test': test_ds
    }

    datasets = {k:strategy.experimental_distribute_dataset(datasets[k]) for k in datasets}
    datasets['train_len'] = train_ds.cardinality().numpy()
    datasets['num_classes'] = 1000
    args.input_size = [224,224,3]

    print('Datasets are built')
    return datasets

def ILSVRC(args, split = 'train', sample_rate = None, shuffle = False, seed = None, sub_ds = None, saved = False):
    if split == 'train':
        with open(os.path.join(args.data_path, 'class_to_label.txt'),'r') as f:
            CLASSES = f.readlines()
        CLASSES = { name.replace('\n','') : l for l, name in enumerate(CLASSES) }

        label_pathes = glob.glob(os.path.join(args.data_path, split, '*'))

        if sample_rate is not None:
            if abs(sample_rate) < 1:
                min_num_label =  min([len(glob.glob(os.path.join(l, '*'))) for l in label_pathes])
                sampled_data_len = int(abs(sample_rate) * min_num_label)
            else:
                sampled_data_len = abs(sample_rate)

        image_paths = []
        labels = []
        class_names = []
        for name in label_pathes:
            image_path = glob.glob(os.path.join(name, '*'))
            image_path = [p for p in image_path if 'n02105855_2933.JPEG' not in p]

            if sample_rate is not None:
                np.random.seed(seed)
                np.random.shuffle(image_path)
                if sample_rate < 0:
                    image_path = image_path[::-1]
                image_path = image_path[:sampled_data_len]
            image_paths += image_path
            labels += [CLASSES[os.path.split(name)[1]]] * len(image_path)
            class_names.append(os.path.split(name)[1])

        if shuffle:
            np.random.seed(seed)
            idx = np.arange(len(image_paths))
            np.random.shuffle(idx)
            image_paths = [image_paths[i] for i in idx]
            labels = [labels[i] for i in idx]

    elif split == 'val':
        image_paths = glob.glob(os.path.join(args.data_path, split, '*'))
        image_paths.sort()

        with open(os.path.join(args.home_path, 'val_gt.txt'),'r') as f:
            labels = f.readlines()

    print (split + ' dataset length :', len(image_paths))
    label_arry = np.int64(labels)
    img_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(label_arry)
    dataset = tf.data.Dataset.zip((img_ds, label_ds))

    return dataset

def get_size(path):
    img = Image.open(path.decode("utf-8"))
    w,h = img.size
    return np.int32(h), np.int32(w)

def random_resize_crop(image, height, width):
    shape = tf.stack([height, width, 3])
    bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=tf.zeros(shape=[0, 0, 4]),
        min_object_covered=0,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.08, 1.0],
        max_attempts=10,
        use_image_if_no_bounding_boxes=True)

    is_bad = tf.reduce_sum(tf.cast(tf.equal(bbox_size, shape), tf.int32)) >= 2

    if is_bad:
        image = tf.image.decode_jpeg(image, channels = 3, **JPEG_OPT)
        newh, neww = tf.numpy_function(resizeshortest, [tf.shape(image, tf.int32), 256], [tf.int32, tf.int32])
        image = tf.image.resize(image, (newh,neww), method='bicubic')
        image = tf.slice(image, [newh//2-112,neww//2-112,0],[224,224,-1])
    else:
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

        image = tf.image.decode_and_crop_jpeg(image, crop_window, channels = 3, **JPEG_OPT)
        image = tf.image.resize(image, (224,224), method='bicubic')
        image.set_shape((224,224,3))

    return image

def resizeshortest(shape, size):
    h, w = shape[:2]
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, int(scale * w + 0.5)
    else:
        newh, neww = int(scale * h + 0.5), size
    return np.int32(newh), np.int32(neww)

def lighting(image, std, eigval, eigvec):
    v = tf.random.normal(shape=[3], stddev=std) * eigval
    inc = tf.matmul(eigvec, tf.reshape(v, [3, 1]))
    image = image + tf.reshape(inc, [3])
    return image

def pre_processing(is_training = False, contrastive = False):
    @tf.function
    def training(path, label):
        height, width = tf.numpy_function(get_size, [path], [tf.int32, tf.int32])
        image = tf.io.read_file(path)

        height, width = tf.numpy_function(get_size, [path], [tf.int32, tf.int32])
        image = tf.io.read_file(path)
        image = random_resize_crop(image, height, width)
        return image, label

    @tf.function
    def test(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image, channels = 3, **JPEG_OPT)

        newh,neww = tf.numpy_function(resizeshortest, [tf.shape(image, tf.int32), 256], [tf.int32, tf.int32])
        image = tf.image.resize(image, (newh,neww), method='bicubic')
        image = tf.slice(image, [newh//2-112,neww//2-112,0],[224,224,-1])
        return image, label
    return training if is_training else test

def pre_processing_batched(is_training = False, contrastive = False, mode = 0):
    @tf.function
    def training(image, *argv):
        image = tf.image.random_flip_left_right(image)
        image = (image-np.array([123.675, 116.28 , 103.53 ]))/np.array([58.395, 57.12, 57.375])
        if len(argv) == 2:
            image = tf.reshape(image, [B,N,H,W,D])
        return [image] + [arg for arg in argv]

    @tf.function
    def test(image, *argv):
        shape = image.shape
        image = (image-np.array([123.675, 116.28 , 103.53 ]))/np.array([58.395, 57.12, 57.375])
        return [image] + [arg for arg in argv]
    return training if is_training else test

