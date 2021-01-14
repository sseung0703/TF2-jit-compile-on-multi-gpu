import tensorflow as tf
import numpy as np

def Optimizer(args, model, strategy):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.SUM)
    optimizer = tf.keras.optimizers.SGD(args.learning_rate, .9, nesterov=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function(jit_compile = args.compile)
    def compiled_step(images, labels):
        with tf.GradientTape() as tape:
            pred = model(images, training = True)
            total_loss = loss_object(labels, pred)/args.batch_size
        gradients = tape.gradient(total_loss, model.trainable_variables)
        update_vars = [model.Layers[k].update_var if hasattr(model.Layers[k], 'update_var') else None for k in model.Layers ]
        return total_loss, pred, gradients, update_vars

    def train_step(images, labels):
        total_loss, pred, gradients, update_vars = compiled_step(images, labels)
        if args.weight_decay > 0.:
            gradients = [g+v*args.weight_decay for g,v in zip(gradients, model.trainable_variables)]
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        for k, v in zip(model.Layers, update_vars):
            if hasattr(model.Layers[k], 'update'):
                model.Layers[k].update(v)

        train_loss.update_state(total_loss)
        train_accuracy.update_state(labels, pred)

    @tf.function()
    def train_step_dist(image, labels):
        strategy.run(train_step, args= (image, labels))

    return train_step_dist, train_loss, train_accuracy, optimizer

