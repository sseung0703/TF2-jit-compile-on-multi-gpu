import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import tensorflow as tf

from dataloader import ILSVRC, CIFAR
import utils

home_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='')

parser.add_argument("--arch", default='ResNet-50', type=str)
parser.add_argument("--dataset", default='ILSVRC', type=str)

parser.add_argument("--val_batch_size", default=256, type=int)
parser.add_argument("--trained_param", default = 'res50_ilsvrc.pkl',type=str)
parser.add_argument("--data_path", default = '/home/cvip/nas/ssd/ILSVRC2012',type=str)

parser.add_argument("--gpu_id", default= [0], type=int, nargs = '+')
parser.add_argument("--compile", default = False, action = 'store_true')

args = parser.parse_args()

args.home_path = os.path.dirname(os.path.abspath(__file__))
args.input_size = [224,224,3]

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([tf.config.list_physical_devices('GPU')[i] for i in args.gpu_id], 'GPU')
    for gpu_id in args.gpu_id:
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    devices = ['/gpu:{}'.format(i) for i in args.gpu_id]
    strategy = tf.distribute.MirroredStrategy(devices, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    with strategy.scope():
        if args.dataset == 'ILSVRC':
            datasets = ILSVRC.build_dataset_providers(args, strategy, test_only = True)
        elif 'CIFAR' in args.dataset:
            datasets = CIFAR.build_dataset_providers(args, strategy, test_only = True)

        model = utils.load_model(args, datasets['num_classes'], args.trained_param)

        top1_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='top1_accuracy')
        top5_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')

        @tf.function(experimental_compile = args.compile)
        def compiled_step(images):
            return model(images, training = False)

        def test_step(images, labels):
            pred = compiled_step(images, labels)
            top1_accuracy.update_state(labels, pred)
            top5_accuracy.update_state(labels, pred)

        @tf.function
        def test_step_dist(images, labels):
            strategy.run(test_step, args=(images, labels))
            
        for i, (test_images, test_labels) in enumerate(datasets['test']):
            test_step_dist(test_images, test_labels)

        top1_acc = top1_accuracy.result().numpy()
        top5_acc = top5_accuracy.result().numpy()
        top1_accuracy.reset_states()
        top5_accuracy.reset_states()
        print ('Test ACC. Top-1: %.4f, Top-5: %.4f'%(top1_acc, top5_acc))