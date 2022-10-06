## Multiple Adversarial Learning based Angiography Reconstruction for Ultra-low-dose Contrast Medium CT
## Weiwei Zhang, Zhen Zhou, Zhifan Gao, Guang Yang, Lei Xu, Weiwen Wu, and Heye Zhang

## ------- Demo file to test MALAR on the sampled dicom files---------

import argparse
import os
import tensorflow as tf
from test_model import MALAR
from test_functions import get_info


parser = argparse.ArgumentParser(description='')
# -------------------------------------
# set load directory
parser.add_argument('--UDCT_path', dest='UDCT_path', default='test_UDCT', help='ultra-low-dose .dcm file directory')
parser.add_argument('--pi_path', dest='pi_path', default='pixel_info', help='truncated pixel information directory of ultra-low-dose .dcm')

# set save directory
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',  default='checkpoint', help='check point dir')
parser.add_argument('--test_save_dir', dest='test_save_dir',  default='reconstruction_result', help='test process save dir')

# image info
parser.add_argument('--img_size', dest='img_size', type=int,  default=384, help='input image size, h=w')

# train detail
parser.add_argument('--ngf', dest='ngf', type=int, default=32, help='# of gen filters in first conv layer')
parser.add_argument('--nglf', dest='nglf', type=int, default=15, help='# of gen filters in last conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
# -------------------------------------

args, unknown = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)

# save truncated pixels info
get_info(args.UDCT_path)
print('Saving pixel position and CT values are completed...!!!')

# reconstruction
model = MALAR(sess, args)
model.test(args)
