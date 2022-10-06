import os
import sys
import numpy as np
from glob import glob
import pydicom


def read_dicom(dcm_path):
    dicom = pydicom.read_file(dcm_path, force=True)
    image = dicom.pixel_array.astype(np.int16)
    image[image < 0] = 0
    image += np.int16(dicom.RescaleIntercept)

    return dicom, image


# save the position and CT value of the pixels ranged outside [WL-WW/2, WL+WW/2]
def get_pixel_info_outside_ranges(img):
    p_v = list()
    for i in range(79, 463):
        for j in range(66, 450):
            if img[i, j] < -300 or img[i, j] > 700:
                p_v.append([i, j, img[i, j]])
    return p_v


# operate on multiple .dcm files
def get_info(data_path):

    # 1.dicom file path
    dcm_path = [d for d in glob(data_path + '/*')]
    dcm_path.sort()

    for i in range(len(dcm_path)):

        # 2.read .dcm
        _slice = pydicom.read_file(dcm_path[i], force=True)
        image = _slice.pixel_array.astype(np.int16)
        image[image < 0] = 0
        image += np.int16(_slice.RescaleIntercept)
        _name = dcm_path[i].replace('\\', '/').split('/')[-1]

        # 3.get position and CT value, save to .npy
        print('Saving truncated pixel information: {}'.format(_name)); sys.stdout.flush()
        pi = get_pixel_info_outside_ranges(image)

        if not os.path.exists('pixel_info'):
            os.makedirs('pixel_info')
        np.save(os.path.join('pixel_info', _name), pi)


# restore truncated pixels
def load_pixel_info(img, img_pi):
    for i in range(len(img_pi)):
        p_v = img_pi[i]
        x = p_v[0]
        y = p_v[1]
        value = p_v[2]
        img[x-79, y-66] = value + 1024
    return img
