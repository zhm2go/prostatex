"""prostatex dataset."""
import os
from collections import defaultdict

import pydicom
import scipy
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt
from numpy import where
import SimpleITK as sitk
import csv
import pandas as pd
from wasabi import wasabi

# TODO(prostatex): Markdown description  that will appear on the catalog page.


_DESCRIPTION = """
SPIE-AAPM-NCI PROSTATEx Challenges
"""

# TODO(prostatex): BibTeX citation
_CITATION = """
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
"""

class ProstateXConfig(tfds.core.BuilderConfig):
  """BuilderConfig for DeeplesionConfig."""
  def __init__(self,
               name=None,
               **kwargs):
    super(ProstateXConfig,
          self).__init__(name=name,
                         version=tfds.core.Version('1.0.0'),
                         **kwargs)

class Prostatex(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for prostatex dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
      Download from https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges#23691656d4622c5ad5884bdb876d6d441994da38
      to the `manual_dir/` by NBIA data retriever.
      """
    BUILDER_CONFIGS = [
        ProstateXConfig(
            name='volume',
            description=_DESCRIPTION,
        ),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(prostatex): Specifies the tfds.core.DatasetInfo object
        if self.builder_config.name == 'stack' or self.builder_config.name == 'nostack':
            features = tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, None), dtype=tf.uint16),
                'significant': tfds.features.ClassLabel(names=['TRUE', 'FALSE']),
                'zone': tfds.features.ClassLabel(names=['PZ', 'AS', 'TZ', 'SV']),
                # TODO: image type should be in configure builder
                'DCMSerDescr': tfds.features.Text(),
                'ProxID': tfds.features.Text(),
                'fid': tfds.features.Text(),
                'position': tfds.features.Text(),
                'ijk': tfds.features.Text(),
                # TODO: ijk to tfds.feature.bbox; alignment; deploy in colab
            })
        elif self.builder_config.name == 'volume':
            features = tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Sequence(tfds.features.Image(shape=(None, None, None), dtype=tf.uint16)),
                'significant': tfds.features.Sequence(tfds.features.ClassLabel(names=['TRUE', 'FALSE'])),
                'zone': tfds.features.Sequence(tfds.features.ClassLabel(names=['PZ', 'AS', 'TZ', 'SV'])),
                # TODO: image type should be in configure builder
                'DCMSerDescr': tfds.features.Text(),
                'ProxID': tfds.features.Text(),
                'fid': tfds.features.Sequence(tfds.features.Text()),
                'position': tfds.features.Sequence(tfds.features.Text()),
                'ijk': tfds.features.Sequence(tfds.features.Text()),
                # TODO: ijk to tfds.feature.bbox; alignment; deploy in colab
            })
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=features,
                # TODO: ijk to tfds.feature.bbox; alignment; deploy in colab
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'significant'),  # Set to `None` to disable
            homepage='https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+PROSTATEx+Challenges#23691656d4622c5ad5884bdb876d6d441994da38',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(prostatex): Downloads the data and defines the splits
        path = "s3://gradient-public-data/prostatex"

        # TODO(prostatex): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path),
            # 'test': self._generate_examples(path / 'ProstateX-0001'),
        }

    @staticmethod
    def get_image(images_row, path, images_location):
        dimension = images_row[-3]
        dimension = dimension.split('x')  # get total number of slices
        ijk = images_row['ijk']  # k means kth slices in the series

        ijk = ijk.split()
        k = int(ijk[-1]) + 1 if int(ijk[-1]) < int(dimension[2]) else int(dimension[2])
        k = 1 if k < 1 else k
        if int(k) < 10 and int(dimension[2]) >= 10:
            file_name = "1-0{0}.dcm".format(k)
        else:
            file_name = "1-{0}.dcm".format(k)
        whole_path = path + '/manifest' + images_location[
            images_row['DCMSerDescr'].replace("_", "").replace("=", "")] + '/' + file_name
        image = tf.io.gfile.GFile(whole_path, mode='rb')
        image_dc = pydicom.dcmread(image)
        img = image_dc.pixel_array
        return img

    @staticmethod
    def add_to_overlay(image_overlay, image_overlay_ijk, images_row, image):  # add appropriate images to be overlaid
        image_name = images_row['Name']
        if 'diff' in image_name and 'ADC' in image_name:
            image_overlay['diff'] = np.squeeze(image)
            image_overlay_ijk['diff'] = images_row['ijk']
        elif 't2_tse_tra' in image_name:
            image_overlay['t2'] = np.squeeze(image)
            image_overlay_ijk['t2'] = images_row['ijk']
        elif 'tfl_3d_PD' in image_name:
            image_overlay['PD'] = np.squeeze(image)
            image_overlay_ijk['PD'] = images_row['ijk']

    @staticmethod
    def overlay_images(image_overlay, image_overlay_ijk):  # resize images and overlay them together
        def to_square(tp):
            img = image_overlay[tp]
            ijk = image_overlay_ijk[tp].split()
            row = img.shape[0]
            col = img.shape[1]
            mid = row - col
            for i in range(len(ijk)):
                ijk[i] = int(ijk[i])
            if row == col:
                return
            elif row > col:
                j = ijk[1] - mid // 2
                image_overlay_ijk[tp] = "{0} {1} {2}".format(ijk[0], j, ijk[2])
                image_overlay[tp] = img[mid // 2:row - mid // 2]
            else:
                i = ijk[0] + mid // 2
                image_overlay_ijk[tp] = "{0} {1} {2}".format(i, ijk[1], ijk[2])
                image_overlay[tp] = np.array([row[- mid // 2: col + mid // 2] for row in img])

        def zoom_and_shift(tp, max_resolution, ref_ijk):
            img_resize = scipy.ndimage.interpolation.zoom(np.squeeze(image_overlay[tp]), max_resolution / image_overlay[tp].shape[0])
            ijk = image_overlay_ijk[tp].split()
            i = round(int(ijk[0]) * max_resolution / image_overlay[tp].shape[0])
            j = round(int(ijk[1]) * max_resolution / image_overlay[tp].shape[0])
            image_overlay_ijk[tp] = "{0} {1} {2}".format(i, j, ijk[2])
            ref_i = round(int(ref_ijk[0]))
            ref_j = round(int(ref_ijk[1]))
            dj = ref_j - j
            di = ref_i - i
            img_shift = np.roll(img_resize, dj, axis=0)
            img_shift = np.roll(img_shift, di, axis=1)
            if dj > 0:
                img_shift[:dj, :] = 0
            elif dj < 0:
                img_shift[dj:, :] = 0
            if di > 0:
                img_shift[:, :di] = 0
            elif di < 0:
                img_shift[:, di:] = 0
            image_overlay[tp] = img_shift

        def get_largest_image(max_resolution):
            if image_overlay['diff'].shape[0] == max_resolution:
                return image_overlay_ijk['diff'].split()
            if image_overlay['t2'].shape[0] == max_resolution:
                return image_overlay_ijk['t2'].split()
            if image_overlay['PD'].shape[0] == max_resolution:
                return image_overlay_ijk['PD'].split()
            if image_overlay['ktran'].shape[0] == max_resolution:
                return image_overlay_ijk['ktran'].split()

        def resize_and_overlay():
            diff_img, t2_img, PD_img, ktran_img = image_overlay['diff'], image_overlay['t2'], image_overlay['PD'], image_overlay['ktran']
            max_resolution = max(diff_img.shape[0], t2_img.shape[0], PD_img.shape[0], ktran_img.shape[0])
            ref_ijk = get_largest_image(max_resolution)
            zoom_and_shift('diff', max_resolution, ref_ijk)
            zoom_and_shift('t2', max_resolution, ref_ijk)
            zoom_and_shift('PD', max_resolution, ref_ijk)
            zoom_and_shift('ktran', max_resolution, ref_ijk)
            image_overlay_ijk['ref'] = ref_ijk
            return np.stack([image_overlay['diff'], image_overlay['t2'], image_overlay['PD'], image_overlay['ktran']], axis=-1)

        to_square('diff')
        to_square('t2')
        to_square('PD')
        to_square('ktran')
        return resize_and_overlay()

    @staticmethod
    def get_ktran_image(ktran_row, path):
        ProxID = ktran_row['ProxID']
        mhd_path = path + '/KTran/ProstateXKtrains-train-fixed/{0}/{0}-Ktrans.mhd'.format(ProxID)
        zraw_path = path + '/KTran/ProstateXKtrains-train-fixed/{0}/{0}-Ktrans.zraw'.format(ProxID)
        ktran_mhd = tf.io.gfile.GFile(mhd_path, mode='rb')
        temp = open('{0}-Ktrans.mhd'.format(ProxID), 'ab')
        temp.write(ktran_mhd.read())
        temp.close()
        ktran_zraw = tf.io.gfile.GFile(zraw_path, mode='rb')
        temp2 = open('{0}-Ktrans.zraw'.format(ProxID), 'ab')
        temp2.write(ktran_zraw.read())
        temp2.close()
        itkimage = sitk.ReadImage(
            '{0}-Ktrans.mhd'.format(ProxID), imageIO="MetaImageIO"
        )
        scan = sitk.GetArrayFromImage(itkimage)
        os.remove('{0}-Ktrans.mhd'.format(ProxID))
        os.remove('{0}-Ktrans.zraw'.format(ProxID))
        return scan

    @staticmethod
    def get_csv_iter(path_to_csv):
        metadata = tf.io.gfile.GFile(path_to_csv, 'rb+')
        metadata_csv = pd.read_csv(metadata)
        reader = metadata_csv.iterrows()
        return reader

    @staticmethod
    def get_image_series(num_slice, path, images_location, DCMSerDescr):
        img_list = []
        for k in range(1, num_slice + 1):
            if int(k) < 10 and int(num_slice) >= 10:
                file_name = "1-0{0}.dcm".format(k)
            else:
                file_name = "1-{0}.dcm".format(k)
            whole_path = path + '/manifest' + images_location[
                DCMSerDescr.replace("_", "").replace("=", "")] + '/' + file_name
            print(whole_path)
            image = tf.io.gfile.GFile(whole_path, mode='rb')
            image_dc = pydicom.dcmread(image)
            img = image_dc.pixel_array
            img_list.append(np.expand_dims(img, axis=2))
        return img_list

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(prostatex): Yields (key, example) tuples from the dataset
        path = str(path)
        FINDINGS_PATH = path + '/metadata/ProstateX-Findings-Train.csv'
        KTRAN_PATH = path + '/metadata/ProstateX-Images-KTrans-Train.csv'
        IMAGES_PATH = path + '/metadata/ProstateX-Images-Train.csv'
        DICOM_METADATA_PATH = path + '/metadata/metadata.csv'

        # open all .csv files to locate desired images
        findings_reader = self.get_csv_iter(FINDINGS_PATH)
        dicom_metadata_reader = self.get_csv_iter(DICOM_METADATA_PATH)
        images_reader = self.get_csv_iter(IMAGES_PATH)
        ktran_reader = self.get_csv_iter(KTRAN_PATH)

        itera = 0
        prevID = 'ProstateX--1'  # fake prevID as an initializer
        images_location = {}
        images_number = {}
        images_ijk = defaultdict(list)
        metadata_buf_stack = []
        image_buf_stack = []

        fid_list = []
        pos_list = []
        zone_list = []
        significance_list = []
        ktran_ijk_list = []
        scan = None

        # for each finding, get its corresponding images and their paths
        for f_idx, findings_row in findings_reader:
            ProxID, fid, pos, zone, significance = findings_row['ProxID'], findings_row['fid'], findings_row['pos'], findings_row['zone'], findings_row['ClinSig']

            # set up the dictionary to map DCMSerDescr to path referred in metadata.csv

            if prevID != ProxID and len(images_location) > 0:  # implicitly ktran scan is not None
                if self.builder_config.name == 'volume':
                    for key in images_location:
                        image_series = self.get_image_series(images_number[key], path, images_location, key)
                        yield prevID + key + '-volume', {
                            'image': image_series,
                            'significant': significance_list,
                            'zone': zone_list,
                            'DCMSerDescr': key,
                            'ProxID': prevID,
                            'fid': fid_list,
                            'position': pos_list,
                            'ijk': images_ijk[key],
                        }
                    ktran_images = []
                    for slice in scan:
                        ktran_slice = np.squeeze(slice)
                        ktran_slice = where(ktran_slice != 0, np.log10(ktran_slice), 1000)
                        ktran_slice = ((ktran_slice + 1000) * 1000).astype('uint16')
                        ktran_images.append(np.expand_dims(ktran_slice, axis=2))
                    yield prevID + '-ktran' + '-volume', {
                        'image': ktran_images,
                        'significant': significance_list,
                        'zone': zone_list,
                        'DCMSerDescr': 'ktranFromDCE',
                        'ProxID': prevID,
                        'fid': fid_list,
                        'position': pos_list,
                        'ijk': ktran_ijk_list,
                    }
                    significance_list, zone_list, fid_list, pos_list, images_ijk, ktran_ijk_list = [], [], [], [], defaultdict(list), []  # clean the data
                    # of prevID

            fid_list.append(str(fid))
            pos_list.append(pos)
            zone_list.append(zone)
            significance_list.append(significance)

            if prevID != ProxID:  # if ProxID equals to prevID, use the last dictionary
                images_location = {}
                if metadata_buf_stack:
                    metadata_buf = metadata_buf_stack[-1]
                    if metadata_buf['Subject ID'] == ProxID:
                        images_location[metadata_buf['Series Description']] = metadata_buf['File Location'][11:]
                        images_number[metadata_buf['Series Description']] = int(metadata_buf['Number of Images'])
                        metadata_buf_stack.pop()
                        for d_idx, dicom_metadata_row in dicom_metadata_reader:
                            if dicom_metadata_row['Subject ID'] != ProxID:
                                metadata_buf_stack.append(dicom_metadata_row)
                                break
                            images_location[dicom_metadata_row['Series Description']] = dicom_metadata_row['File Location'][11:]
                            images_number[dicom_metadata_row['Series Description']] = int(dicom_metadata_row['Number of Images'])
                else:
                    for d_idx, dicom_metadata_row in dicom_metadata_reader:
                        if dicom_metadata_row['Subject ID'] != ProxID:
                            metadata_buf_stack.append(dicom_metadata_row)
                            break
                        images_location[dicom_metadata_row['Series Description']] = dicom_metadata_row['File Location'][11:]
                        images_number[dicom_metadata_row['Series Description']] = int(dicom_metadata_row['Number of Images'])
            # for each finding image, decode it and yield to tfds

            image_overlay = defaultdict(lambda: np.ndarray(0))
            image_overlay_ijk = {}
            if image_buf_stack:  # if the image_buf_stack contains elements, it must be the first one of this ProxID and fid and pos
                images_row = image_buf_stack.pop()
                assert images_row['ProxID'] == ProxID and images_row['fid'] == fid and images_row['pos'] == pos
                if ProxID != 'ProstateX-0025':  # 25 has a unique corner case
                    image = self.get_image(images_row, path, images_location)
                    self.add_to_overlay(image_overlay, image_overlay_ijk, images_row, image)
                    if images_row['ijk'] not in images_ijk[images_row['DCMSerDescr'].replace('_', '').replace('=', '')]:
                        images_ijk[images_row['DCMSerDescr'].replace('_', '').replace('=', '')].append(images_row['ijk'])
                    if self.builder_config.name == 'nostack':
                        yield str(ProxID) + str(fid) + images_row['Name'] + str(images_row['DCMSerNum']) + images_row['pos'], {
                            'image': np.expand_dims(image, axis=2),
                            'significant': significance,
                            'zone': zone,
                            'DCMSerDescr': images_row['DCMSerDescr'],
                            'ProxID': ProxID,
                            'fid': str(fid),
                            'position': pos,
                            'ijk': images_row['ijk'],
                        }
            for i_idx, images_row in images_reader:
                if images_row['ProxID'] != ProxID or images_row['fid'] != fid or images_row['pos'] != pos:  # check image ProxID and fid
                    image_buf_stack.append(images_row)
                    break
                if ProxID == 'ProstateX-0025':  # 25 has a unique corner case
                    continue
                image = self.get_image(images_row, path, images_location)
                self.add_to_overlay(image_overlay, image_overlay_ijk, images_row, image)
                if images_row['ijk'] not in images_ijk[images_row['DCMSerDescr'].replace('_', '').replace('=', '')]:
                    images_ijk[images_row['DCMSerDescr'].replace('_', '').replace('=', '')].append(images_row['ijk'])
                if self.builder_config.name == 'nostack':
                    yield str(ProxID) + str(fid) + images_row['Name'] + str(images_row['DCMSerNum']) + images_row['pos'], {
                        'image': np.expand_dims(image, axis=2),
                        'significant': significance,
                        'zone': zone,
                        'DCMSerDescr': images_row['DCMSerDescr'],
                        'ProxID': ProxID,
                        'fid': str(fid),
                        'position': pos,
                        'ijk': images_row['ijk'],
                    }

            #yield KTRAN to tfds

            if ProxID == 'ProstateX-0025':  # escape 25 which is corner case
                next(ktran_reader)  # skip 2 lines as 2 entries for one finding in case 25
                next(ktran_reader)
            else:
                k_idx, ktran_row = next(ktran_reader)
                assert ktran_row['ProxID'] == ProxID and ktran_row['fid'] == fid
                ijk = ktran_row['ijk'].split()
                scan = self.get_ktran_image(ktran_row, path)
                num_slices = scan.shape[0]
                k = int(ijk[-1]) if int(ijk[-1]) < int(num_slices) else int(num_slices) - 1
                k = 0 if k < 0 else k
                ktran_img = np.squeeze(scan[k])
                ktran_img = where(ktran_img != 0, np.log10(ktran_img), 1000)
                image_overlay['ktran'] = ((ktran_img + 1000) * 1000).astype('uint16')  # make ktran scan more contrast
                image_overlay_ijk['ktran'] = ktran_row['ijk']
                if self.builder_config.name == 'nostack':
                    yield ProxID + str(fid) + pos + '-ktran', {
                        'image': np.expand_dims(image_overlay['ktran'], axis=2),
                        'significant': significance,
                        'zone': zone,
                        'DCMSerDescr': 'ktranFromDCE',
                        'ProxID': ProxID,
                        'fid': str(fid),
                        'position': pos,
                        'ijk': ktran_row['ijk'],
                    }
                ktran_ijk_list.append(ktran_row['ijk'])
            #  overlay images and yield
            if ProxID != 'ProstateX-0025' and self.builder_config.name == 'stack':
                assert len(image_overlay.keys()) == 4
                overlay_image = self.overlay_images(image_overlay, image_overlay_ijk)
                yield ProxID + str(fid) + pos + '-stack', {
                    'image': overlay_image,
                    'significant': significance,
                    'zone': zone,
                    'DCMSerDescr': 'stackFromDiffT2PDKtran',
                    'ProxID': ProxID,
                    'fid': str(fid),
                    'position': pos,
                    'ijk': ' '.join(image_overlay_ijk['ref']),
                }

            prevID = ProxID  # update prevID
            itera += 1
            if itera >= 1:
                break

        if self.builder_config.name == 'volume':
            for key in images_location:
                image_series = self.get_image_series(images_number[key], path, images_location, key)
                yield prevID + key + '-volume', {
                    'image': image_series,
                    'significant': significance_list,
                    'zone': zone_list,
                    'DCMSerDescr': key,
                    'ProxID': prevID,
                    'fid': fid_list,
                    'position': pos_list,
                    'ijk': images_ijk[key],
                }
            ktran_images = []
            for slice in scan:
                ktran_slice = np.squeeze(slice)
                ktran_slice = where(ktran_slice != 0, np.log10(ktran_slice), 1000)
                ktran_slice = ((ktran_slice + 1000) * 1000).astype('uint16')
                ktran_images.append(np.expand_dims(ktran_slice, axis=2))
            yield prevID + '-ktran' + '-volume', {
                'image': ktran_images,
                'significant': significance_list,
                'zone': zone_list,
                'DCMSerDescr': 'ktranFromDCE',
                'ProxID': prevID,
                'fid': fid_list,
                'position': pos_list,
                'ijk': ktran_ijk_list,
            }

