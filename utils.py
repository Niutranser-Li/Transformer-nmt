import os, sys
import time, datetime

import tensorflow as tf

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BLEU_CALCULATOR_PATH = os.path.join(CURRENT_DIR_PATH, 'multi-bleu.perl')

class Mask:
    @classmethod
    def create_padding_mask(cls, sequences):
        sequences = tf.cast(tf.math.equal(sequences, 0), dtype=tf.float32)
        return sequences[:, tf.newaxis, tf.newaxis, :]
    
    @classmethod
    def create_look_ahead_mask(cls, seq_len):
        return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    
    @classmethod
    def create_masks(cls, inputs, target):
        encoder_padding_mask = Mask.create_padding_mask(inputs)
        decoder_padding_mask = Mask.create_padding_mask(inputs)

        look_ahead_mask = tf.maximum(
            Mask.create_look_ahead_mask(tf.shape(target)[1]),
            Mask.create_padding_mask(target)
        )

        return encoder_padding_mask, look_ahead_mask, decoder_padding_mask

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, dtype=tf.float32)
        self.warmup_steps = warmup_steps
    
    def call(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def label_smoothing(target_data, depth, epsilon=0.1):
    target_data_one_hot = tf.one_hot(target_data, depth=depth)
    n = target_data_one_hot.get_shape().as_list()[-1]
    return ((1-epsilon) * target_data_one_hot) + (epsilon/n)

class Trainer:
    def __init__(self, model, dataset, loss_object=None, optimizer=None, checkpoint_dir='./checkpoints', batch_size=None, distribute_strategy=None, vocab_size=32000, epoch=20):
        self.model = model
        self.dataset = dataset
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.vocab_size = vocab_size
        self.epoch = epoch
        self.dataset = dataset

        os.makedirs(checkpoint_dir, exist_ok=True)
        if self.optimizer is None:
            self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), model=self.model)
        else:
            self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

        # metrics
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        self.validation_loss = tf.keras.metrics.Mean('validation_loss', dtype=tf.float32)
        self.validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('validation_accuracy')
    
    def trainer(self, reset_checkpoint, is_distributed=False):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d')
        train_log_dir = './log/graient_tape/' + current_time + '/train'
        os.makedirs(train_log_dir, exist_ok=True)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if not reset_checkpoint:
            if self.checkpoint_manager.latest_checkpoint:
                print('Retored from {}'.format(self.checkpoint_manager.latest_checkpoint))
            else:
                print('Initializing from scratch.')
            self.checkpoint.restore(
                self.checkpoint_manager.latest_checkpoint
            )
        else:
            print('reset and initializing from scratch.')
        
        batch_count = len(self.dataset)

        for epoch in range(self.epoch):
            start_time = time.time()
            print('trainer - start learning.')

            for (batch, (inputs, target)) in enumerate(self.dataset):
                if is_distributed: self.distributed_train_step(inputs, target)
                else: self.basic_train_step(inputs, target)

                self.checkpoint.step.assign_add(1)
                if batch % 50 == 0:
                    print(
                        "epoch: {} batch: {}/{} loss: {} accuracy: {}".format(
                            epoch, batch, batch_count, self.train_loss.result(), self.train_accuracy.result()
                        )
                    )
                if batch % 10000 == 0 and batch != 0:
                    self.checkpoint_manager.save()
            print(
                '{} | epoch: {} loss: {} accuracy: {} time: {} sec.'.format(
                    datetime.datetime.now(), epoch, self.train_loss.result(), self.train_accuracy.result(),
                    time.time()-start_time
                )
            )
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('train_accuracy', self.train_accuracy.result(), step=epoch)

            self.checkpoint_manager.save()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.validation_loss.reset_states()
            self.validation_accuracy.reset_states()
        self.checkpoint_manager.save()
    
    def basic_train_step(self, inputs, target):
        target_include_start = target[:, :-1]
        target_include_end = target[:, 1:]
        encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_masks(inputs, target)

        with tf.GradientTape() as tape:
            pred = self.model.call(
                inputs=inputs, target = target_include_start,
                inputs_padding_mask = encoder_padding_mask,
                look_ahead_mask = look_ahead_mask,
                target_padding_mask = decoder_padding_mask,
                training = True
            )
            loss = self.loss_function(target_include_end, pred)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(target_include_end, pred)

        if self.distribute_strategy is None:
            return tf.reduce_mean(loss)
        return loss
    
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        real_one_hot = label_smoothing(real, depth=self.vocab_size, epsilon=0.1)
        loss = self.loss_object(real_one_hot, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_mean(loss)
    
    @tf.function
    def train_step(self, inputs, target):
        return self.basic_train_step(inputs, target)

    @tf.function
    def distributed_train_step(self, inputs, target):
        loss = self.distribute_strategy.experimental_run_v2(self.basic_train_step, args=(inputs, target))
        loss_value = self.distribute_strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        return tf.reduce_mean(loss_value)
