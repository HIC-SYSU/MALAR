import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


# save visualization img in validation process
def save_image(LDCT, LDCT_pre, LDCT_cycle, NDCT, NDCT_pre, NDCT_cycle, save_dir='.', max_=1, min_=-1):

    f, axes = plt.subplots(2, 4, figsize=(30, 20))

    axes[0, 0].imshow(LDCT, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[0, 1].imshow(LDCT_pre, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[0, 2].imshow(LDCT_pre - LDCT, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[0, 3].imshow(LDCT_cycle, cmap=plt.cm.gray, vmax=max_, vmin=min_)

    axes[1, 0].imshow(NDCT, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[1, 1].imshow(NDCT_pre, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[1, 2].imshow(NDCT - NDCT_pre, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[1, 3].imshow(NDCT_cycle, cmap=plt.cm.gray, vmax=max_, vmin=min_)

    axes[0, 0].title.set_text('LDCT')
    axes[0, 1].title.set_text('LDCT output')
    axes[0, 2].title.set_text('LDCT output - LDCT')
    axes[0, 3].title.set_text('LDCT cycle')

    axes[1, 0].title.set_text('NDCT')
    axes[1, 1].title.set_text('NDCT output')
    axes[1, 2].title.set_text('NDCT - NDCT output')
    axes[1, 3].title.set_text('NDCT cycle')

    if save_dir != '.':
        f.savefig(save_dir)
        plt.close()


# argparser string -> boolean type
def ParseBoolean(b):
    b = b.lower()
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError('Cannot parse string into boolean.')


# argparser string -> boolean type
def ParseList(l):
    return l.split(',')
