"""prostatex dataset."""
from collections import defaultdict

import pandas as pd
import pydicom
import scipy
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt
import csv
import SimpleITK as sitk

# TODO(prostatex): Markdown description  that will appear on the catalog page.
from numpy import where

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
            name='stack',
            description=_DESCRIPTION,
        ),
        ProstateXConfig(
            name='nostack',
            description=_DESCRIPTION,
        ),
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
                'significant': tfds.features.ClassLabel(names=['TRUE', 'FALSE', 'TBD']),
                'ggg': tfds.features.ClassLabel(names=['N/A', '1', '2', '3', '4', '5',  'TBD']),
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
                'significant': tfds.features.Sequence(tfds.features.ClassLabel(names=['TRUE', 'FALSE', 'TBD'])),
                'ggg': tfds.features.Sequence(tfds.features.ClassLabel(names=['N/A', '1', '2', '3', '4', '5', 'TBD'])),
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
        path = dl_manager.manual_dir
        train_paths = ['/manifest-train1', '/ProstateX-TrainingLesionInformation', '/ProstateXKtrans-train-fixed', 'Train']
        test_paths = ['/manifest-test1', '/ProstateX-TestLesionInformation', '/ProstateXKtrans-test-fixed', 'Test']
        # TODO(prostatex): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path, train_paths),
            'test': self._generate_examples(path, test_paths),
        }

    @staticmethod
    def get_csv_iter(path_to_csv):
        metadata = tf.io.gfile.GFile(path_to_csv, 'rb+')
        metadata_csv = pd.read_csv(metadata)
        reader = metadata_csv.iterrows()
        return reader

    @staticmethod
    def get_image(DCMSerDescr, ijk, path, images_location, images_number):
        #DCMSerDescr = images_row[-2].replace("_", "").replace("=", "")
        #ijk = images_row[-9]  # k means kth slices in the series
        ijk = ijk.split()

        k = int(ijk[-1]) + 1 if int(ijk[-1]) < int(images_number[DCMSerDescr]) else int(images_number[DCMSerDescr])
        k = 1 if k < 1 else k
        if int(k) < 10 and int(images_number[DCMSerDescr]) >= 10:
            file_name = "1-0{0}.dcm".format(k)
        else:
            file_name = "1-{0}.dcm".format(k)
        whole_path = path + images_location[DCMSerDescr] + '/' + file_name
        f = tf.io.gfile.GFile(whole_path, mode='rb')
        ds = pydicom.dcmread(f)
        img = ds.pixel_array
        spacing = ds[0x28, 0x30].value
        return img, spacing

    @staticmethod
    def get_image_series(num_slice, path, images_location, DCMSerDescr):
        img_list = []
        for k in range(1, num_slice + 1):
            if int(k) < 10 and int(num_slice) >= 10:
                file_name = "1-0{0}.dcm".format(k)
            else:
                file_name = "1-{0}.dcm".format(k)
            whole_path = path + images_location[
                DCMSerDescr.replace("_", "").replace("=", "")] + '/' + file_name
            f = tf.io.gfile.GFile(whole_path, mode='rb')
            ds = pydicom.dcmread(f)
            img = ds.pixel_array
            img_list.append(np.expand_dims(img, axis=2))
        return img_list

    @staticmethod
    def get_ktran_series(scan):
        ktran_images = []
        for slice in scan:
            ktran_slice = np.squeeze(slice)
            ktran_slice = where(ktran_slice != 0, np.log10(ktran_slice), 1000)
            ktran_slice = ((ktran_slice + 1000) * 1000).astype('uint16')
            ktran_images.append(np.expand_dims(ktran_slice, axis=2))
        return ktran_images

    @staticmethod
    def add_to_overlay(image_overlay, image_overlay_ijk, image_spacing, image_name, ijk, image, spacing):  # add appropriate images to be overlaid
        if ('diff' in image_name or 'DIFF' in image_name) and 'ADC' in image_name:
            image_overlay['diff'] = np.squeeze(image)
            image_overlay_ijk['diff'] = ijk
            image_spacing['diff'] = float(spacing[0])
        elif 't2_tse_tra' in image_name:
            image_overlay['t2'] = np.squeeze(image)
            image_overlay_ijk['t2'] = ijk
            image_spacing['t2'] = float(spacing[0])
        elif 'PD' in image_name and '3d' in image_name:
            image_overlay['PD'] = np.squeeze(image)
            image_overlay_ijk['PD'] = ijk
            image_spacing['PD'] = float(spacing[0])
    @staticmethod
    def overlay_images(image_overlay, image_overlay_ijk, image_spacing):  # resize images and overlay them together
        def to_square(tp):
            img = image_overlay[tp]
            ijk = image_overlay_ijk[tp].split()
            row = img.shape[0]
            col = img.shape[1]
            mid = row - col

            if row == col:
                return
            elif row > col:
                i = int(ijk[0]) + mid // 2
                image_overlay_ijk[tp] = "{0} {1} {2}".format(i, ijk[1], ijk[2])
                zeros = np.zeros((row, mid // 2), dtype=img.dtype)
                new_img = np.concatenate((zeros, img, zeros), axis=1)
                image_overlay[tp] = new_img
            else:
                j = int(ijk[1]) - mid // 2
                image_overlay_ijk[tp] = "{0} {1} {2}".format(ijk[0], j, ijk[2])
                zeros = np.zeros((-mid // 2, col), dtype=img.dtype)
                new_img = np.concatenate((zeros, img, zeros), axis=0)
                image_overlay[tp] = new_img

        def zoom_and_shift(tp, max_resolution, ref_ijk, max_mode):
            coefficient = image_spacing[tp] * np.shape(image_overlay[tp])[0] / (max_resolution * image_spacing[max_mode])
            img_resize = scipy.ndimage.zoom(np.squeeze(image_overlay[tp]),
                                                          max_resolution * coefficient / image_overlay[tp].shape[0])
            ijk = image_overlay_ijk[tp].split()
            i = round(int(ijk[0]) * max_resolution * coefficient / image_overlay[tp].shape[0])
            j = round(int(ijk[1]) * max_resolution * coefficient / image_overlay[tp].shape[0])
            image_overlay_ijk[tp] = "{0} {1} {2}".format(i, j, ijk[2])
            ref_i = int(ref_ijk[0])
            ref_j = int(ref_ijk[1])
            if coefficient == 1:
                img_shift = img_resize
            elif coefficient > 1:
                this_resolution = np.shape(img_resize)[0]
                zeros = np.zeros((this_resolution, 50), dtype=img_resize.dtype)
                img_add_col = np.concatenate((zeros, img_resize, zeros), axis=1)
                zeros = np.zeros((50, this_resolution + 100), dtype=img_resize.dtype)
                img_add_row = np.concatenate((zeros, img_add_col, zeros), axis=0)
                new_j, new_i = j + 50, i + 50
                img_cut_row = img_add_row[new_j - ref_j:new_j - ref_j + max_resolution] #cut rows
                img_shift = np.array([row[new_i - ref_i:new_i - ref_i + max_resolution] for row in img_cut_row]) # cut columns
            elif coefficient < 1:
                this_resolution = np.shape(img_resize)[0]
                zeros = np.zeros((this_resolution, (max_resolution + 100 - this_resolution) // 2), dtype=img_resize.dtype)
                img_add_col = np.concatenate((zeros, img_resize, zeros), axis=1)
                zeros = np.zeros(((max_resolution + 100 - this_resolution) // 2, max_resolution + 100), dtype=img_add_col.dtype)
                img_add_row = np.concatenate((zeros, img_add_col, zeros), axis=0)
                new_i, new_j = i + max_resolution + 50 - this_resolution, j + max_resolution + 50 - this_resolution
                dj, di = ref_j - new_j, ref_i - new_i
                img_cut_row = img_add_row[new_j - ref_j:new_j - ref_j + max_resolution]  # cut rows
                img_shift = np.array([row[new_i - ref_i:new_i - ref_i + max_resolution] for row in img_cut_row])  # cut columns
            image_overlay[tp] = img_shift

        def get_largest_image(max_resolution):
            if image_overlay['diff'].shape[0] == max_resolution:
                return image_overlay_ijk['diff'].split(), 'diff'
            if image_overlay['t2'].shape[0] == max_resolution:
                return image_overlay_ijk['t2'].split(), 't2'
            if image_overlay['PD'].shape[0] == max_resolution:
                return image_overlay_ijk['PD'].split(), 'PD'
            if image_overlay['ktran'].shape[0] == max_resolution:
                return image_overlay_ijk['ktran'].split(), 'tran'

        def resize_and_overlay():
            diff_img, t2_img, PD_img, ktran_img = image_overlay['diff'], image_overlay['t2'], image_overlay['PD'], \
                                                  image_overlay['ktran']
            max_resolution = max(diff_img.shape[0], t2_img.shape[0], PD_img.shape[0], ktran_img.shape[0])
            ref_ijk, max_mode = get_largest_image(max_resolution)
            zoom_and_shift('diff', max_resolution, ref_ijk, max_mode)
            zoom_and_shift('t2', max_resolution, ref_ijk, max_mode)
            zoom_and_shift('PD', max_resolution, ref_ijk, max_mode)
            zoom_and_shift('ktran', max_resolution, ref_ijk, max_mode)
            image_overlay_ijk['ref'] = ref_ijk
            try:
                assert image_overlay['diff'].shape == image_overlay['t2'].shape == image_overlay['PD'].shape == image_overlay['ktran'].shape
            except AssertionError:
                print(max_resolution)
                print(max_mode)
                print(ref_ijk)
            return np.stack([image_overlay['diff'], image_overlay['t2'], image_overlay['PD'], image_overlay['ktran']], axis=-1)
        to_square('diff')
        to_square('t2')
        to_square('PD')
        to_square('ktran')
        return resize_and_overlay()

    def _generate_examples(self, path, file_paths):
        #path the the absolute path to parent dir of all files, file_paths includes: 1.path to dycom images 2.path to metadata 3.path to mhd images
        """Yields examples."""
        # TODO(prostatex): Yields (key, example) tuples from the dataset
        path = str(path)
        dycom_path, metadata_path, mhd_path, split = file_paths
        FINDINGS_PATH = path + metadata_path + '/ProstateX-Findings-{0}.csv'.format(split)
        GGG_FINDINGS_PATH = path + metadata_path + '/ProstateX-2-Findings-{0}.csv'.format(split)
        KTRAN_PATH = path + metadata_path + '/ProstateX-Images-KTrans-{0}.csv'.format(split)
        IMAGES_PATH = path + metadata_path + '/ProstateX-Images-{0}.csv'.format(split)
        DICOM_METADATA_PATH = path + dycom_path + '/metadata.csv'


        # open all .csv files to locate desired images
        findings_reader = self.get_csv_iter(FINDINGS_PATH)
        ggg_findings_reader = self.get_csv_iter(GGG_FINDINGS_PATH)
        images_reader = csv.reader(open(IMAGES_PATH, newline=''), delimiter=',', quotechar='|')
        next(images_reader)
        dicom_metadata_reader = csv.reader(open(DICOM_METADATA_PATH, newline=''), delimiter=',', quotechar='|')
        next(dicom_metadata_reader)
        ktran_reader = csv.reader(open(KTRAN_PATH, newline=''), delimiter=',', quotechar='|')
        next(ktran_reader)
        iter = 0
        prevID = 'ProstateX--1'  # fake prevID as an initializer
        images_location, images_number, images_ijk = {}, {}, defaultdict(list)
        metadata_buf_stack, image_buf_stack = [], []
        fid_list, pos_list, zone_list, significance_list, ktran_ijk_list, ggg_list = [], [], [], [], [], []
        scan = None
        ggg_exist, ggg_dic = set(), {}

        # read the ggg of some findings from prostatex2
        for g_idx, ggg_findings_row in ggg_findings_reader:
            key = '{0};{1};{2}'.format(ggg_findings_row['ProxID'], ggg_findings_row['fid'], ggg_findings_row['pos'])
            ggg_exist.add(key)
            ggg_dic[key] = str(ggg_findings_row['ggg']) if 'ggg' in ggg_findings_row else 'TBD'

        # for each finding, get its corresponding images and their paths
        for f_idx, findings_row in findings_reader:
            ProxID, fid, pos, zone = findings_row['ProxID'], str(findings_row['fid']), findings_row['pos'], findings_row['zone']
            significance = findings_row['ClinSig'] if 'ClinSig' in findings_row else 'TBD'
            ggg_key = '{0};{1};{2}'.format(ProxID, fid, pos)
            ggg = ggg_dic[ggg_key] if ggg_key in ggg_dic else 'N/A'
            print('{0} {1} {2} {3}'.format(ProxID, fid, pos, ggg))
            # generate volume images
            if prevID != ProxID and len(images_location) > 0:  # implicitly ktran scan is not None
                if self.builder_config.name == 'volume':
                    for key in images_location:
                        image_series = self.get_image_series(images_number[key], path + dycom_path, images_location, key)
                        yield prevID + key + '-volume', {
                            'image': image_series,
                            'significant': significance_list,
                            'ggg': ggg_list,
                            'zone': zone_list,
                            'DCMSerDescr': key,
                            'ProxID': prevID,
                            'fid': fid_list,
                            'position': pos_list,
                            'ijk': images_ijk[key],
                        }
                    ktran_images = self.get_ktran_series(scan)
                    yield prevID + '-ktran' + '-volume', {
                        'image': ktran_images,
                        'significant': significance_list,
                        'ggg': ggg_list,
                        'zone': zone_list,
                        'DCMSerDescr': 'ktranFromDCE',
                        'ProxID': prevID,
                        'fid': fid_list,
                        'position': pos_list,
                        'ijk': ktran_ijk_list,
                    }
                    significance_list, zone_list, fid_list, pos_list, images_ijk, ktran_ijk_list, ggg_list = [], [], [], [], defaultdict(
                        list), [], []  # clean the data
                    # of prevID

            fid_list.append(fid)
            pos_list.append(pos)
            zone_list.append(zone)
            significance_list.append(significance)
            ggg_list.append(ggg)

            # set up the dictionary to map DCMSerDescr to path referred in metadata.csv

            if prevID != ProxID:  # if ProxID equals to prevID, use the last dictionary
                images_location, tmp_dicom_metadata = {}, None
                for dicom_metadata_row in dicom_metadata_reader:
                    if dicom_metadata_row[4] != ProxID:
                        tmp_dicom_metadata = dicom_metadata_row
                        break
                    metadata_buf_stack.append(dicom_metadata_row)
                len_metadata_buf_stack = len(metadata_buf_stack)
                for i in range(len_metadata_buf_stack):
                    metadata_buf = metadata_buf_stack.pop()
                    images_location[metadata_buf[8]] = metadata_buf[15][1:]
                    images_number[metadata_buf[8]] = int(metadata_buf[13])
                if tmp_dicom_metadata is not None: metadata_buf_stack.append(tmp_dicom_metadata)
            # for each finding nostack image, decode it and yield to tfds

            image_overlay, image_overlay_ijk, image_spacing = defaultdict(lambda: np.ndarray(0)), {}, {}
            temp_image = None
            for images_row in images_reader:
                if images_row[0] != ProxID or images_row[2] != fid or images_row[3] != pos:
                    #  when reading image of next ProxID, process all the images_row in stack for this ProxID
                    temp_image = images_row
                    break
                image_buf_stack.append(images_row)
            len_image_buf_stack = len(image_buf_stack)
            for i in range(len_image_buf_stack):
                image_row = image_buf_stack.pop()
                assert image_row[0] == ProxID and image_row[2] == fid and image_row[3] == pos
                DCMSerDes, ijk, image_name = image_row[-2].replace('_', '').replace('=', ''), image_row[-9], image_row[1]
                image, spacing = self.get_image(DCMSerDes, ijk, path + dycom_path, images_location, images_number)
                self.add_to_overlay(image_overlay, image_overlay_ijk, image_spacing, image_name, ijk, image, spacing)
                if ijk not in images_ijk[DCMSerDes]:
                    images_ijk[DCMSerDes].append(image_row[-9])
                if self.builder_config.name == 'nostack':
                    yield ProxID + fid + image_row[1] + image_row[-1] + image_row[3], {
                            'image': np.expand_dims(image, axis=2),
                            'significant': significance,
                            'ggg': ggg,
                            'zone': zone,
                            'DCMSerDescr': image_row[-2],
                            'ProxID': ProxID,
                            'fid': fid,
                            'position': pos,
                            'ijk': image_row[-9],
                    }
            if temp_image is not None: image_buf_stack.append(temp_image)

            # yield KTRAN nostack to tfds


            ktran_row = next(ktran_reader)
            assert ktran_row[0] == ProxID and ktran_row[1] == fid
            ijk = ktran_row[-1].split()
            itkimage = sitk.ReadImage(
                path + mhd_path + '/{0}/{0}-Ktrans.mhd'.format(ProxID)
            )
            scan = sitk.GetArrayFromImage(itkimage)
            ktran_spacing = itkimage.GetSpacing()
            num_slices = scan.shape[0]
            k = int(ijk[-1]) if int(ijk[-1]) < int(num_slices) else int(num_slices) - 1
            k = 0 if k < 0 else k
            ktran_img = np.squeeze(scan[k])
            ktran_img = where(ktran_img != 0, np.log10(ktran_img), 1000)
            image_overlay['ktran'] = ((ktran_img + 1000) * 1000).astype('uint16')  # make ktran scan more contrast
            image_overlay_ijk['ktran'] = ktran_row[-1]
            image_spacing['ktran'] = ktran_spacing[0]
            if self.builder_config.name == 'nostack':
                yield ProxID + fid + pos + '-ktran', {
                    'image': np.expand_dims(image_overlay['ktran'], axis=-1),
                    'significant': significance,
                    'ggg': ggg,
                    'zone': zone,
                    'DCMSerDescr': 'ktranFromDCE',
                    'ProxID': ProxID,
                    'fid': fid,
                    'position': pos,
                    'ijk': ktran_row[-1],
                }
            ktran_ijk_list.append(ktran_row[-1])
                # TODO: correct ktran volume

            #  in test split, use ktran's ijk as PD's ijk as they have the same resolution and spacing

            if split == 'Test' and 'PD' not in image_overlay:
                PD_DCMSerDescr = None
                for key in images_location.keys():
                    if 'PD' in key:
                        PD_DCMSerDescr = key
                assert PD_DCMSerDescr is not None
                PD_ijk = image_overlay_ijk['ktran']
                image, spacing = self.get_image(PD_DCMSerDescr, PD_ijk, path + dycom_path, images_location, images_number)
                self.add_to_overlay(image_overlay, image_overlay_ijk, image_spacing, PD_DCMSerDescr, PD_ijk, image, spacing)
                if ijk not in images_ijk[PD_DCMSerDescr]:
                    images_ijk[PD_DCMSerDescr].append(PD_ijk)
                if self.builder_config.name == 'nostack':
                    yield ProxID + fid + PD_DCMSerDescr + pos, {
                        'image': np.expand_dims(image, axis=2),
                        'significant': significance,
                        'ggg': ggg,
                        'zone': zone,
                        'DCMSerDescr': PD_DCMSerDescr,
                        'ProxID': ProxID,
                        'fid': fid,
                        'position': pos,
                        'ijk': PD_ijk,
                    }


            #  overlay images and yield stack

            if self.builder_config.name == 'stack':
                assert len(image_overlay.keys()) == 4
                overlay_image = self.overlay_images(image_overlay, image_overlay_ijk, image_spacing)
                yield ProxID + fid + pos + '-stack', {
                    'image': overlay_image,
                    'significant': significance,
                    'ggg': ggg,
                    'zone': zone,
                    'DCMSerDescr': 'stackFromDiffT2PDKtran',
                    'ProxID': ProxID,
                    'fid': fid,
                    'position': pos,
                    'ijk': ' '.join(image_overlay_ijk['ref']),
                }

            prevID = ProxID  # update prevID
            iter += 1
            if iter >= 400:
                break

        # process the last ProxID's volume
        if self.builder_config.name == 'volume':
            for key in images_location:
                image_series = self.get_image_series(images_number[key], path + dycom_path, images_location, key)
                yield prevID + key + '-volume', {
                    'image': image_series,
                    'significant': significance_list,
                    'ggg': ggg_list,
                    'zone': zone_list,
                    'DCMSerDescr': key,
                    'ProxID': prevID,
                    'fid': fid_list,
                    'position': pos_list,
                    'ijk': images_ijk[key],
                }
            ktran_images = self.get_ktran_series(scan)
            yield prevID + '-ktran' + '-volume', {
                'image': ktran_images,
                'significant': significance_list,
                'ggg': ggg_list,
                'zone': zone_list,
                'DCMSerDescr': 'ktranFromDCE',
                'ProxID': prevID,
                'fid': fid_list,
                'position': pos_list,
                'ijk': ktran_ijk_list,
            }
