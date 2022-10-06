from __future__ import division
import os
import numpy as np
from glob import glob
from collections import namedtuple
import time

import tensorflow as tf

import network as net
import read_filename as rf
import gen_file_queue as gfq
import inout_util as ut


class MALAR(object):
    def __init__(self, sess, args):
        self.sess = sess

        # save directory
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = os.path.join('.', 'logs')

        # network init
        self.generator = net.generator
        self.discriminator = net.discriminator
        self.vgg = net.vgg_net

        # network options
        OPTIONS = namedtuple('OPTIONS', 'gf_dim glf_dim df_dim')
        self.options = OPTIONS._make((args.ngf, args.nglf, args.ndf))

        # read file path
        dir_patient_UD = os.path.join(args.dataset_path, args.UDCT_path)
        dir_patient_LD = os.path.join(args.dataset_path, args.LDCT_path)
        dir_patient_UD_train = os.path.join(dir_patient_UD, "trainset")
        dir_patient_LD_train = os.path.join(dir_patient_LD, "trainset")
        dir_patient_UD_test = os.path.join(dir_patient_UD, "testset")
        dir_patient_LD_test = os.path.join(dir_patient_LD, "testset")
        train_patient_UD = [d for d in glob(dir_patient_UD_train + '/*')]
        train_patient_LD = [d for d in glob(dir_patient_LD_train + '/*')]
        test_patient_UD = [d for d in glob(dir_patient_UD_test + '/*')]
        test_patient_LD = [d for d in glob(dir_patient_LD_test + '/*')]
        train_patient_UD.sort()
        train_patient_LD.sort()
        test_patient_UD.sort()
        test_patient_LD.sort()
        # generate filename
        self.filename_generator = rf.ReadFilename()
        self.train_UDCT_filename, self.train_LDCT_filename = self.filename_generator(train_patient_UD, train_patient_LD)
        self.test_UDCT_filename, self.test_LDCT_filename = self.filename_generator(test_patient_UD, test_patient_LD)
        # filename save to .txt/.csv
        f1 = open('./train_UDCT_filename.txt', 'w'); f1.writelines(fn + '\n' for fn in self.train_UDCT_filename); f1.close()
        f2 = open('./train_LDCT_filename.txt', 'w'); f2.writelines(fn + '\n' for fn in self.train_LDCT_filename); f2.close()
        f3 = open('./test_UDCT_filename.txt', 'w'); f3.writelines(fn + '\n' for fn in self.test_UDCT_filename); f3.close()
        f4 = open('./test_LDCT_filename.txt', 'w'); f4.writelines(fn + '\n' for fn in self.test_LDCT_filename); f4.close()

        # use queue to read .tfrecords
        self.file_queue_generator = gfq.GERFileQueue(end_epoch=args.end_epoch, batch_size=args.batch_size, num_threads=args.num_threads)
        self.patch_X_big_batch, self.patch_Y_big_batch = self.file_queue_generator.input_pipeline(self.train_UDCT_filename, self.train_LDCT_filename)
        self.patch_X_mg = tf.split(self.patch_X_big_batch, args.gpu_num, axis=0)
        self.patch_Y_mg = tf.split(self.patch_Y_big_batch, args.gpu_num, axis=0)
        self.test_data_UD, self.test_name_UD, self.test_data_LD, self.test_name_LD = self.file_queue_generator.gen_batch(self.test_UDCT_filename, self.test_LDCT_filename, is_validation=True)
        print('Input complete !!!')

        """
        build model
        """
        # image placeholder(for validation)
        self.test_X = tf.placeholder(tf.float32, [1, args.img_size, args.img_size, 1], name='X')
        self.test_Y = tf.placeholder(tf.float32, [1, args.img_size, args.img_size, 1], name='Y')

        # multi-gpu setting
        self.tower_D_grads, self.tower_G_grads = [], []
        self.tower_D_loss_Y_1, self.tower_D_loss_Y_2, self.tower_G_loss_X2Y, self.tower_reconstruction_loss_X = [], [], [], []
        self.tower_D_loss_X_1, self.tower_D_loss_X_2, self.tower_G_loss_Y2X, self.tower_reconstruction_loss_Y = [], [], [], []

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.opt = tf.train.AdamOptimizer(self.lr, beta1=args.beta1)
        with tf.variable_scope(tf.get_variable_scope()):
            for i, d in enumerate(args.gpu_no_2):
                with tf.device('/gpu: %s' % d):
                    with tf.name_scope('tower_%s' % d):
                        self.patch_X, self.patch_Y = self.patch_X_mg[i], self.patch_Y_mg[i]

                        if i == 0:
                            False_OR_True = False  # first GPU, i.e. firstly initialize variables
                        else:
                            False_OR_True = True  # other GPUs, i.e. repeatedly initialize variables

                        # Generator
                        self.G_X = self.generator(self.patch_X, self.options, reuse=False_OR_True, name="generatorX2Y")
                        self.F_GX = self.generator(self.G_X, self.options, reuse=False_OR_True, name="generatorY2X")
                        self.F_Y = self.generator(self.patch_Y, self.options, reuse=True, name="generatorY2X")
                        self.G_FY = self.generator(self.F_Y, self.options, reuse=True, name="generatorX2Y")

                        # Discriminator
                        self.D_GX = self.discriminator(self.G_X, self.options, reuse=False_OR_True, name="discriminatorY")
                        self.D_FY = self.discriminator(self.F_Y, self.options, reuse=False_OR_True, name="discriminatorX")
                        self.D_Y = self.discriminator(self.patch_Y, self.options, reuse=True, name="discriminatorY")
                        self.D_X = self.discriminator(self.patch_X, self.options, reuse=True, name="discriminatorX")

                        # vgg
                        self.rm_GX = self.vgg(self.G_X)
                        self.rm_Y = self.vgg(self.patch_Y)
                        self.rm_FY = self.vgg(self.F_Y)
                        self.rm_X = self.vgg(self.patch_X)

                        # Loss
                        # generator loss
                        self.G_loss_X2Y = net.least_square(self.D_GX, tf.ones_like(self.D_GX))
                        self.G_loss_Y2X = net.least_square(self.D_FY, tf.ones_like(self.D_FY))

                        # reconstruction loss
                        self.reconstruction_loss_X = net.reconstruction_loss(self.patch_X, self.F_GX, args.L1_lambda)
                        self.reconstruction_loss_Y = net.reconstruction_loss(self.patch_Y, self.G_FY, args.L1_lambda)

                        # discriminator loss
                        self.D_loss_patch_Y = net.least_square(self.D_Y, tf.ones_like(self.D_Y))
                        self.D_loss_patch_GX = net.least_square(self.D_GX, tf.zeros_like(self.D_GX))
                        self.D_loss_patch_X = net.least_square(self.D_X, tf.ones_like(self.D_X))
                        self.D_loss_patch_FY = net.least_square(self.D_FY, tf.zeros_like(self.D_FY))
                        self.D_loss_Y_1 = (self.D_loss_patch_Y + self.D_loss_patch_GX)
                        self.D_loss_X_1 = (self.D_loss_patch_X + self.D_loss_patch_FY)
                        self.D_loss_1 = (self.D_loss_X_1 + self.D_loss_Y_1) / 2

                        # intra-correlation loss
                        self.D_loss_Y_2 = net.least_square(net.intra_feature(self.rm_GX), net.intra_feature(self.rm_Y))
                        self.D_loss_X_2 = net.least_square(net.intra_feature(self.rm_FY), net.intra_feature(self.rm_X))
                        self.D_loss_2 = (self.D_loss_X_2 + self.D_loss_Y_2) / 2

                        # total G_loss, D_loss
                        self.G_loss = self.G_loss_X2Y + self.G_loss_Y2X + self.reconstruction_loss_X + self.reconstruction_loss_Y
                        self.D_loss = self.D_loss_1 + self.D_loss_2

                        # variable list
                        t_vars = tf.trainable_variables()
                        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
                        self.g_vars = [var for var in t_vars if 'generator' in var.name]

                        # cal gradients for the batch on this gpu
                        self.D_grads = self.opt.compute_gradients(self.D_loss, var_list=self.d_vars)
                        self.G_grads = self.opt.compute_gradients(self.G_loss, var_list=self.g_vars)

                        # gradient
                        self.tower_D_grads.append(self.D_grads)
                        self.tower_G_grads.append(self.G_grads)

                        # loss list
                        self.tower_D_loss_Y_1.append(self.D_loss_Y_1)
                        self.tower_D_loss_Y_2.append(self.D_loss_Y_2)
                        self.tower_G_loss_X2Y.append(self.G_loss_X2Y)
                        self.tower_reconstruction_loss_X.append(self.reconstruction_loss_X)
                        self.tower_D_loss_X_1.append(self.D_loss_X_1)
                        self.tower_D_loss_X_2.append(self.D_loss_X_2)
                        self.tower_G_loss_Y2X.append(self.G_loss_Y2X)
                        self.tower_reconstruction_loss_Y.append(self.reconstruction_loss_Y)

                        """
                        Summary
                        """
                        # generator loss
                        self.G_loss_X2Y_sum = tf.summary.scalar("1_G_loss_X2Y", self.G_loss_X2Y, family='Generator_loss')
                        self.G_loss_Y2X_sum = tf.summary.scalar("2_G_loss_Y2X", self.G_loss_Y2X, family='Generator_loss')
                        self.reconstruction_loss_X_sum = tf.summary.scalar("1_reconstruction_loss_X", self.reconstruction_loss_X, family='reconstruction_loss')
                        self.reconstruction_loss_Y_sum = tf.summary.scalar("2_reconstruction_loss_Y", self.reconstruction_loss_Y, family='reconstruction_loss')
                        self.g_sum = tf.summary.merge([self.G_loss_X2Y_sum, self.G_loss_Y2X_sum, self.reconstruction_loss_X_sum, self.reconstruction_loss_Y_sum])

                        # discriminator loss
                        self.D_loss_Y_1_sum = tf.summary.scalar("1_D_loss_Y_1", self.D_loss_Y_1, family='Discriminator_loss')
                        self.D_loss_X_1_sum = tf.summary.scalar("2_D_loss_X_1", self.D_loss_X_1, family='Discriminator_loss')
                        self.D_loss_Y_2_sum = tf.summary.scalar("3_D_loss_Y_2", self.D_loss_Y_2, family='Discriminator_loss')
                        self.D_loss_X_2_sum = tf.summary.scalar("4_D_loss_X_2", self.D_loss_X_2, family='Discriminator_loss')
                        self.d_sum = tf.summary.merge([self.D_loss_Y_1_sum, self.D_loss_X_1_sum, self.D_loss_Y_2_sum, self.D_loss_X_2_sum])

                        # image summary
                        self.test_G_X = self.generator(self.test_X, self.options, reuse=True, name="generatorX2Y")
                        self.test_F_GX = self.generator(self.test_G_X, self.options, reuse=True, name="generatorY2X")
                        self.test_F_Y = self.generator(self.test_Y, self.options, reuse=True, name="generatorY2X")
                        self.test_G_FY = self.generator(self.test_F_Y, self.options, reuse=True, name="generatorX2Y")
                        self.train_img_summary = tf.concat([self.patch_X, self.G_X, self.G_X - self.patch_X, self.F_GX, self.patch_Y, self.F_Y, self.patch_Y - self.F_Y, self.G_FY], axis=2)
                        self.summary_image_1 = tf.summary.image('1_train_patch_image', self.train_img_summary)
                        self.test_img_summary = tf.concat([self.test_X, self.test_G_X, self.test_G_X - self.test_X, self.test_F_GX, self.test_Y, self.test_F_Y, self.test_Y - self.test_F_Y, self.test_G_FY], axis=2)
                        self.summary_image_2 = tf.summary.image('2_test_whole_image', self.test_img_summary)

        # compute mean grad
        self.mean_D_grads = self.average_gradients(self.tower_D_grads)
        self.mean_G_grads = self.average_gradients(self.tower_G_grads)
        # update var based on mean_grad
        self.d_optim = self.opt.apply_gradients(self.mean_D_grads)
        self.g_optim = self.opt.apply_gradients(self.mean_G_grads)

        # multi-gpu mean loss
        self.mean_D_loss_Y_1 = tf.reduce_mean(self.tower_D_loss_Y_1)
        self.mean_D_loss_Y_2 = tf.reduce_mean(self.tower_D_loss_Y_2)
        self.mean_G_loss_X2Y = tf.reduce_mean(self.tower_G_loss_X2Y)
        self.mean_reconstruction_loss_X = tf.reduce_mean(self.tower_reconstruction_loss_X)
        self.mean_D_loss_X_1 = tf.reduce_mean(self.tower_D_loss_X_1)
        self.mean_D_loss_X_2 = tf.reduce_mean(self.tower_D_loss_X_2)
        self.mean_G_loss_Y2X = tf.reduce_mean(self.tower_G_loss_Y2X)
        self.mean_reconstruction_loss_Y = tf.reduce_mean(self.tower_reconstruction_loss_Y)

        # model saver
        self.saver = tf.train.Saver(max_to_keep=None)

        print('--------------------------------------------\n# of parameters : {} '.format(np.sum([np.prod(v.get_shape().as_list()) for v in (self.d_vars + self.g_vars)])))

    # function of computing mean grad
    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    # training process
    def train(self, args):
        init_1 = tf.global_variables_initializer()
        init_2 = tf.local_variables_initializer()
        self.sess.run(init_1)
        self.sess.run(init_2)

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # pre-trained model load
        self.start_step = 0
        if args.continue_train:
            if self.load():
                print(" [*] Load trained model SUCCESS")
            else:
                print(" [!] Load trained model failed...")
        batch_idxs = int(len(self.train_UDCT_filename) / args.batch_size)
        self.epoch = int(self.start_step / batch_idxs)
        self.iter = self.start_step % batch_idxs
        print('Start point, epoch : {}, iter : {}'.format(self.epoch, self.iter))

        lr = args.lr
        t1 = 0  # training time spent
        for epoch in range(self.epoch, args.end_epoch_es):

            # decay learning rate
            if epoch > args.decay_epoch:
                lr = args.lr - (epoch - args.decay_epoch) * (args.lr / (args.end_epoch - args.decay_epoch))

            for _ in range(0, batch_idxs):
                # Update G network
                X, Y, _, summary_str = self.sess.run([self.patch_X_big_batch, self.patch_Y_big_batch, self.g_optim, self.g_sum], feed_dict={self.lr: lr})
                self.writer.add_summary(summary_str, self.start_step)

                # Update D network
                D_loss_Y_1, D_loss_Y_2, G_loss_X2Y, reconstruction_loss_X, \
                D_loss_X_1, D_loss_X_2, G_loss_Y2X, reconstruction_loss_Y, _, summary_str = \
                    self.sess.run([self.mean_D_loss_Y_1, self.mean_D_loss_Y_2, self.mean_G_loss_X2Y, self.mean_reconstruction_loss_X,
                                   self.mean_D_loss_X_1, self.mean_D_loss_X_2, self.mean_G_loss_Y2X, self.mean_reconstruction_loss_Y, self.d_optim, self.d_sum],
                                  feed_dict={self.patch_X_big_batch: X,
                                             self.patch_Y_big_batch: Y,
                                             self.lr: lr})
                self.writer.add_summary(summary_str, self.start_step)

                if (self.start_step + 1) % args.print_freq == 0:
                    currt_step = self.start_step % batch_idxs if epoch != 0 else self.start_step
                    print(("\r(epoch: {}/{} {}/{} "
                           "D_Y_1: {:.3f} G_X2Y: {:.3f} D_Y_2: {:.3f} cycle_X: {:.3f} "
                           "D_X_1: {:.3f} G_Y2X: {:.3f} D_X_2: {:.3f} cycle_Y: {:.3f}; "
                           "Time: {:.2f}s "
                           .format(epoch, args.end_epoch, currt_step, batch_idxs,
                                   D_loss_Y_1, G_loss_X2Y, D_loss_Y_2, reconstruction_loss_X,
                                   D_loss_X_1, G_loss_Y2X, D_loss_X_2, reconstruction_loss_Y,
                                   round(time.time() - t1, 2))))
                    t1 = time.time()

                    # summary training sample image
                    summary_str1 = self.sess.run(self.summary_image_1)
                    self.writer.add_summary(summary_str1, self.start_step)

                # check sample image
                if (self.start_step + 1) % (args.print_freq * 5) == 0:
                    self.check_sample(args, self.start_step)

                # save model
                if (self.start_step + 1) % batch_idxs == 0:
                    self.save(args, self.start_step)

                self.start_step += 1

            if epoch >= args.end_epoch_es:
                self.coord.request_stop()
                self.coord.join(self.threads)

        self.coord.request_stop()
        self.coord.join(self.threads)

    # validation process
    def check_sample(self, args, idx):
        UD_batch, UD_name_batch, LD_batch, LD_name_batch = self.sess.run([self.test_data_UD, self.test_name_UD, self.test_data_LD, self.test_name_LD])

        x = UD_batch.reshape([1] + self.test_X.get_shape().as_list()[1:])
        y = LD_batch.reshape([1] + self.test_X.get_shape().as_list()[1:])

        G_X, F_GX, F_Y, G_FY = self.sess.run([self.test_G_X, self.test_F_GX, self.test_F_Y, self.test_G_FY],
                                             feed_dict={self.test_X: x, self.test_Y: y})
        G_X = np.array(G_X).astype(np.float32)
        F_GX = np.array(F_GX).astype(np.float32)
        F_Y = np.array(F_Y).astype(np.float32)
        G_FY = np.array(G_FY).astype(np.float32)

        # save img
        validation_save_dir = args.validation_save_dir
        if not os.path.exists(validation_save_dir):
            os.makedirs(validation_save_dir)
        save_dir = os.path.join(validation_save_dir, bytes.decode(UD_name_batch[0]))
        ut.save_image(x[0, :, :, 0], G_X[0, :, :, 0], F_GX[0, :, :, 0], y[0, :, :, 0], F_Y[0, :, :, 0], G_FY[0, :, :, 0], save_dir)

        # summary validation sample image
        if (idx + 1) % (args.print_freq * 5) == 0:
            summary_str2 = self.sess.run([self.summary_image_2],
                                         feed_dict={self.test_X: x,
                                                    self.test_Y: y,
                                                    self.test_G_X: G_X.reshape([1] + self.test_X.get_shape().as_list()[1:]),
                                                    net.is_training: False})
            self.writer.add_summary(summary_str2, idx)

    # save model
    def save(self, args, step):
        model_name = args.model + ".model"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name), global_step=step)

    # load model
    def load(self):
        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(ckpt_name.split('-')[-1])
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False
