import csv

import pydicom

from pydicom.data import get_testdata_file
import os
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from pydicom import dcmread
from pydicom.data import get_testdata_file
import SimpleITK as sitk
import scipy
from prostatex_colab.wasabi import wasabi
import boto3
import matplotlib.pyplot as plt
import pandas as pd
import tempfile



# os.chdir("/Users/zhanghaimeng/tfds")

class MyDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="""
            SPIE-AAPM-NCI PROSTATEx Challenges
            """,
            homepage="https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges#23691656e9a7bb92dcee4b419511436cc3a364b3",
            # TODO: decide features
            features=tfds.features.FeaturesDict({
                'image_description': tfds.features.Text(),
                'image': tfds.features.Image(),
            }),
            supervised_keys=('image', 'label'),
            citation=r"""
             @ARTICLE {,
                author  = "Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F",
                title   = "The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository",
                journal = "Journal of Digital Imaging",
                year    = "2013",
                volume  = "26",
                number  = "6",
                pages   = "1045-1057",
                month   = "dec"
                }
              """,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        extracted_path = dl_manager.manual_dir / 'manifest-A3Y4AE4o5818678569166032044/PROSTATEx/ProstateX-0000/07-07-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-05711'

        # Specify the splits
        # TODO: decide split
        return {
            'train': self._generate_examples(
                images_path=extracted_path / '3.000000-t2tsesag-87368',
            ),
            'test': self._generate_examples(
                images_path=extracted_path / '4.000000-t2tsetra-00702',
            ),
        }

    def _generate_examples(self, images_path):
        return


def test():

    path = 's3://gradient-public-data/prostatex/KTran/ProstateXKtrains-train-fixed/ProstateX-0001/ProstateX-0001-Ktrans.mhd'
    pathz = 's3://gradient-public-data/prostatex/KTran/ProstateXKtrains-train-fixed/ProstateX-0001/ProstateX-0001-Ktrans.zraw'
    f = tf.io.gfile.GFile(path, mode='rb')
    temp = open('tmp.mhd', 'ab')
    temp.write(f.read())
    temp.close()
    fz = tf.io.gfile.GFile(pathz, mode='rb')
    tempz = open('ProstateX-0001-Ktrans.zraw', 'ab')
    tempz.write(fz.read())
    tempz.close()
    img = sitk.ReadImage('tmp.mhd', imageIO="MetaImageIO")
    image = sitk.GetArrayFromImage(img)
    os.remove('tmp.mhd')
    os.remove('ProstateX-0001-Ktrans.zraw')

    image = image[6]
    process_img = np.where(image != 0, np.log10(image), 1000)
    process_img = (((process_img + 1000) * 1000).astype('uint16'))/1000-1000

    plt.imshow(process_img, cmap='gray')
    plt.show()
    return

    filename = 's3://gradient-public-data/prostatex/manifest/ProstateX-0000/07-07-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-05711/7.000000-ep2ddifftraDYNDISTADC-48780/1-10.dcm'
    f = tf.io.gfile.GFile(filename, mode='rb')
    ds = pydicom.dcmread(f)
    img = ds.pixel_array
    plt.imshow(img)
    plt.show()
    f = tf.io.gfile.GFile(
        's3://gradient-public-data/prostatex/metadata/ProstateX-Findings-Train.csv',
        'rb+')
    df = pd.read_csv(f)
    iter = df.iterrows()
    row = next(iter)
    """
    path = '/Users/zhanghaimeng/tensorflow_datasets/downloads/manual'
    FINDINGS_PATH = path + '/ProstateX-TrainingLesionInformationv2/ProstateX-Findings-Train.csv'
    KTRAN_PATH = path + '/ProstateX-TrainingLesionInformationv2/ProstateX-Images-KTrans-Train.csv'
    IMAGES_PATH = path + '/ProstateX-TrainingLesionInformationv2/ProstateX-Images-Train.csv'
    DICOM_METADATA_PATH = path + '/manifest-hjL8tlLc1556886850502670511/metadata.csv'

    image_bytes = tf.io.read_file(
        path + '/manifest-hjL8tlLc1556886850502670511/PROSTATEx/ProstateX-0000/07-07-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-05711/7.000000-ep2ddifftraDYNDISTADC-48780/1-10.dcm')

    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)

    image_bytes2 = tf.io.read_file(
        path + '/manifest-hjL8tlLc1556886850502670511/PROSTATEx/ProstateX-0000/07-07-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-05711/4.000000-t2tsetra-00702/1-10.dcm')

    image2 = tfio.image.decode_dicom_image(image_bytes2, dtype=tf.uint16)
    image_bytes3 = tf.io.read_file(
        path + '/manifest-hjL8tlLc1556886850502670511/PROSTATEx/ProstateX-0000/07-07-2011-MR prostaat kanker detectie WDSmc MCAPRODETW-05711/9.000000-tfl3d PD reftra1.5x1.5t3-67276/1-08.dcm')

    image3 = tfio.image.decode_dicom_image(image_bytes3, dtype=tf.uint16)
    #lossy_image = tfio.image.decode_dicom_image(image_bytes, scale='auto', on_error='lossy', dtype=tf.uint8)

    itkimage = sitk.ReadImage(
        path + '/ProstateXKtrains-train-fixed/{0}/{0}-Ktrans.mhd'.format('ProstateX-0006')
    )
    scan = sitk.GetArrayFromImage(itkimage)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))

    img = scipy.ndimage.interpolation.zoom(np.squeeze(image)[22:106], 384 / 84)
    img2 = scipy.ndimage.interpolation.zoom(np.squeeze(image2), 384 / 384)
    img3 = scipy.ndimage.interpolation.zoom(np.squeeze(image3), 384 / 128)
    img4 = scipy.ndimage.interpolation.zoom(np.squeeze(scan[9]), 384 / 128)
    print(img.shape)
    print(img2.shape)
    print(img3.shape)
    print(np.log10(img4))
    axes[0].imshow(np.squeeze(img), cmap='gray')
    axes[0].set_title('diff')
    axes[1].imshow(np.squeeze(img2), cmap='gray')
    axes[1].set_title('t2')
    axes[2].imshow(np.squeeze(img3), cmap='gray')
    axes[2].set_title('PD')
    axes[3].imshow(np.squeeze(np.log10(img4)), cmap='gray')
    axes[3].set_title('ktran')

    fig.show()
    stack_img = np.stack([img, img2, img3], axis=-1)
    print(stack_img.shape)"""


def main():
    test()


if __name__ == "__main__":
    main()
