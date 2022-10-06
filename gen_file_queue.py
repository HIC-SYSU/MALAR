import tensorflow as tf
import numpy as np


class GERFileQueue(object):
    def __init__(self, end_epoch=1, batch_size=1, num_threads=1):

        # training params
        self.end_epoch = end_epoch
        self.batch_size = batch_size

        # batch generator parameters
        self.num_threads = num_threads
        self.capacity = 2000 * self.num_threads * self.batch_size

    def input_pipeline(self, filename_LD, filename_ND):
        filename_LD.sort()
        filename_ND.sort()
        img_batch, img_name_batch, label_batch, label_name_batch = self.gen_batch(filename_LD, filename_ND)
        return tf.expand_dims(img_batch, axis=-1), tf.expand_dims(label_batch, axis=-1)

    # generate image and label batch
    def gen_batch(self, filename_LD, filename_ND, is_validation=False):
        # read image and label
        img, img_name, state = self.read_and_decode(filename_LD, is_validation)
        img = tf.reshape(img, [512, 512])
        label, label_name, state = self.read_and_decode(filename_ND, is_validation, state)
        label = tf.reshape(label, [512, 512])

        # crop ROI in training process
        img = img[79:463, 66:450]
        label = label[79:463, 66:450]

        # generate batch
        if is_validation:
            batch_size = 1
            capacity = 2
        else:
            batch_size = self.batch_size
            capacity = self.capacity

        img_batch, img_name_batch, label_batch, label_name_batch = tf.train.batch([img, img_name, label, label_name],
                                                                                  batch_size=batch_size,
                                                                                  capacity=capacity,
                                                                                  num_threads=self.num_threads
                                                                                  )
        return img_batch, img_name_batch, label_batch, label_name_batch

    # use queue to read .tfrecords
    def read_and_decode(self, filename, is_validation, state=np.random.get_state()):
        if is_validation:
            np.random.set_state(state)  # sync shuffle in validation process, for visualization comparison
            np.random.shuffle(filename)
        num_epochs = self.end_epoch

        # read file queue, and decode to image
        filename_queue = tf.train.string_input_producer(filename, num_epochs, shuffle=False)  # generate file queue based on filename
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={'image': tf.FixedLenFeature([], tf.string),
                                                     'name': tf.FixedLenFeature([], tf.string)
                                                     }
                                           )
        img = tf.decode_raw(features['image'], tf.float32)
        img_name = features['name']

        # customized windowing
        img = tf.clip_by_value(img, -300, 700)
        img = (img + 300) / float(1000)
        img = 2 * img - 1

        return img, img_name, state
