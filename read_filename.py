## read filename of all .tfrecords
## used to generate the file queue.

import os
from glob import glob
from random import shuffle


class ReadFilename(object):
    def __init__(self, is_unpair=False):
        self.is_unpair = is_unpair

    def __call__(self, patient_UD_list, patient_LD_list):

        UDCT_image_name, LDCT_image_name = [], []
        for patient_UD, patient_LD in zip(patient_UD_list, patient_LD_list):

            # absolute_path
            p_UDCT_path = glob(os.path.join(patient_UD, '*'))
            p_LDCT_path = glob(os.path.join(patient_LD, '*'))
            p_UDCT_path.sort()
            p_LDCT_path.sort()

            # CT slice name
            if self.is_unpair:
                UDCT_image_name.extend(p_UDCT_path)
                shuffle(p_LDCT_path)
                LDCT_image_name.extend(p_LDCT_path)
            else:
                UDCT_image_name.extend(p_UDCT_path)
                LDCT_image_name.extend(p_LDCT_path)

        return UDCT_image_name, LDCT_image_name
