## Multiple Adversarial Learning based Angiography Reconstruction for Ultra-low-dose Contrast Medium CT
## Weiwei Zhang, Zhen Zhou, Zhifan Gao, Guang Yang, Lei Xu, Weiwen Wu, and Heye Zhang

## --------------------------------------------------------------
## Train MALAR on your own dataset, you need to convert your dataset into a tfrecords format first.
## --------------------------------------------------------------

import argparse
import os
import tensorflow as tf
from model import MALAR
import inout_util as ut


parser = argparse.ArgumentParser(description='')
# -------------------------------------
# set load directory
parser.add_argument('--dataset_path', dest='dataset_path', default='TFRecords', help='.tfrecords file directory')
parser.add_argument('--UDCT_path', dest='UDCT_path', default='ultra_low_dose_data', help='UDCT TFRecords folder name')
parser.add_argument('--LDCT_path', dest='LDCT_path', default='low_dose_data', help='LDCT TFRecords folder name')

# set save directory
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='checkpoint', help='pre-trained model dir')
parser.add_argument('--validation_save_dir', dest='validation_save_dir', default='validation_result', help='validation result save dir')

# image info
parser.add_argument('--img_size', dest='img_size', type=int,  default=384, help='input image size, h=w')
parser.add_argument('--img_vmax', dest='img_vmax', type=int, default=3071, help='max value in image')
parser.add_argument('--img_vmin', dest='img_vmin', type=int, default=-1024,  help='max value in image')

# train, test
parser.add_argument('--model', dest='model', default='MALAR', help='MALAR, single_adv, w/o_adaptive_fusion')

# train detail
parser.add_argument('--end_epoch', dest='end_epoch', type=int, default=300, help='end epoch')
parser.add_argument('--end_epoch_es', dest='end_epoch_es', type=int, default=40, help='end epoch for early stopping')
parser.add_argument('--decay_epoch', dest='decay_epoch', type=int, default=30, help='epoch to decay lr')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--batch_size', dest='batch_size', type=int,  default=16, help='batch size')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight of reconstruction loss')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='The exponential decay rate for the 1st moment estimates.')
parser.add_argument('--beta2', dest='beta2', type=float, default=0.999, help='The exponential decay rate for the 2nd moment estimates.')
parser.add_argument('--ngf', dest='ngf', type=int, default=32, help='# of gen filters in first conv layer')
parser.add_argument('--nglf', dest='nglf', type=int, default=15, help='# of gen filters in last conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')

# others
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print_freq (iterations)')
parser.add_argument('--continue_train', dest='continue_train', type=ut.ParseBoolean, default=True, help='load the latest model: true, false')
parser.add_argument('--gpu_no_1', dest='gpu_no_1', type=str,  default='0, 1', help='gpu no 1')
parser.add_argument('--gpu_no_2', dest='gpu_no_2', type=list,  default=[0, 1], help='gpu no 2')
parser.add_argument('--gpu_num', dest='gpu_num', type=str,  default=2, help='gpu num')
parser.add_argument('--unpair', dest='unpair', type=ut.ParseBoolean, default=False, help='unpaired image : True|False')
parser.add_argument('--num_threads', dest='num_threads', type=int, default=1, help='thread num. of loaded data : single|multiple')
# -------------------------------------

args, unknown = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no_1

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
model = MALAR(sess, args)
model.train(args)
