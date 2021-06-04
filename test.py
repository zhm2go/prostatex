import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numpy import where
import prostatex


def test(config):
    data = tfds.load('prostatex/{0}'.format(config), split="train")
    plt_num = 0
    if config == 'stack':
        for dp in data:
            image, DCMSerDescr = dp["image"], dp["DCMSerDescr"]
            image = image.numpy()
            fig, axes = plt.subplots(4, 1, figsize=(25, 25))
            image = np.split(image, 4, axis=-1)
            axes[0].imshow(image[0], cmap='gray')
            axes[0].set_title('diff')
            axes[1].imshow(image[1], cmap='gray')
            axes[1].set_title('t2')
            axes[2].imshow(image[2], cmap='gray')
            axes[2].set_title('PD')
            img3 = image[3] / 1000 - 1000
            axes[3].imshow(img3, cmap='gray')
            axes[3].set_title('ktran')
            ijk = dp["ijk"].numpy().decode("utf-8").split()
            for i in range(4):
                rect = patches.Rectangle((round(int(ijk[1])) - 25, round(int(ijk[0])) - 25), 50, 50, linewidth=1, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect)
            ProxID, fid, pos = dp["ProxID"].numpy().decode("utf-8"), dp["fid"].numpy().decode("utf-8"), dp["position"].numpy().decode("utf-8")
            fig.suptitle("{0}'s Number {1} finding at position {2}\n".format(ProxID, fid, pos))
            fig.subplots_adjust(wspace=0.05)
            fig.show()
            plt_num += 1
            if plt_num >= 30:
                break
    elif config == 'nostack':
        for dp in data:
            image, DCMSerDescr = dp["image"], dp["DCMSerDescr"]
            image = image.numpy()
            fig, axes = plt.subplots(1, 1, figsize=(25, 25))
            axes.imshow(image, cmap='gray')
            ijk = dp["ijk"].numpy().decode("utf-8").split()
            rect = patches.Rectangle((int(ijk[1]) - 25, int(ijk[0]) - 5), 10, 10, linewidth=3, edgecolor='r',facecolor='none')
            axes.add_patch(rect)
            ProxID, fid, pos = dp["ProxID"].numpy().decode("utf-8"), dp["fid"].numpy().decode("utf-8"), dp["position"].numpy().decode("utf-8")
            axes.set_title("{0}'s Number {1} finding at position {2} on {3}".format(ProxID, fid, pos, DCMSerDescr.numpy().decode("utf-8")))
            #fig.suptitle("{0}'s Number {1} finding at position {2} on {3}".format(ProxID, fid, pos, DCMSerDescr.numpy().decode("utf-8")))
            fig.show()
            plt_num += 1
            if plt_num >= 10:
                break

def build():
    builder = tfds.builder('prostatex')
    # 1. Create the tfrecord files (no-op if already exists)
    builder.download_and_prepare()
    # 2. Load the `tf.data.Dataset`
    ds = builder.as_dataset(split='train', shuffle_files=True)
    #print(ds)

def main():
    config = 'stack' #@param ["stack", "nostack"]
    test(config)


if __name__ == "__main__":
    main()