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
                'ggg': tfds.features.ClassLabel(names=['N/A', '1', '2', '3', '4', '5', 'TBD']),
                'zone': tfds.features.ClassLabel(names=['PZ', 'AS', 'TZ', 'SV']),
                # TODO: image type should be in configure builder
                'DCMSerDescr': tfds.features.Text(),
                'ProxID': tfds.features.Text(),
                'fid': tfds.features.Text(),
                'position': tfds.features.Text(),
                'ijk': tfds.features.Text(),
                'bbox': tfds.features.BBoxFeature(),
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
                'bbox': tfds.features.Sequence(tfds.features.BBoxFeature()),
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
        train_paths = ['/manifest', '/KTran/ProstateXKtrains-train-fixed', 'Train']
        test_paths = ['/manifest-test', '/KTran/ProstateXKtrans-test-fixed', 'Test']
        # TODO(prostatex): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'test': self._generate_examples(path, test_paths),
            'train': self._generate_examples(path, train_paths),

        }

    @staticmethod
    def get_image(DCMSerDescr, ijk, path, images_location, images_number):
        ijk = ijk.split()
        k = int(ijk[-1]) if int(ijk[-1]) < int(images_number[DCMSerDescr]) else int(images_number[DCMSerDescr]) - 1
        k = 0 if k < 0 else k
        if 'cor' in DCMSerDescr or 'sag' in DCMSerDescr:
            k = k + 1  # for coronal and sagital plane, the image sequence are straight
        else:
            k = int(images_number[DCMSerDescr]) - k  # for transverse, reverse the sequence
        if int(k) < 10 and int(images_number[DCMSerDescr]) >= 10:
            file_name = "1-0{0}.dcm".format(k)
        else:
            file_name = "1-{0}.dcm".format(k)
        whole_path = path + images_location[DCMSerDescr] + '/' + file_name
        f = tf.io.gfile.GFile(whole_path, mode='rb')
        ds = pydicom.dcmread(f)
        img = ds.pixel_array
        spacing = ds[0x28, 0x30].value
        origin = ds[0x20, 0x32].value
        return img, spacing, origin

    @staticmethod
    def add_to_overlay(image_overlay, image_overlay_ijk, image_spacing, image_origin, image_name, ijk, image, spacing,
                       origin):  # add appropriate images to be overlaid
        if ('diff' in image_name or 'DIFF' in image_name) and 'ADC' in image_name:
            image_overlay['diff'] = np.squeeze(image)
            image_overlay_ijk['diff'] = ijk
            image_spacing['diff'] = tuple([float(i) for i in spacing])
            image_origin['diff'] = tuple([float(i) for i in origin[:2]])
        elif 't2_tse_tra' in image_name:
            image_overlay['t2'] = np.squeeze(image)
            image_overlay_ijk['t2'] = ijk
            image_spacing['t2'] = tuple([float(i) for i in spacing])
            image_origin['t2'] = tuple([float(i) for i in origin[:2]])
        elif 'PD' in image_name and '3d' in image_name:
            image_overlay['PD'] = np.squeeze(image)
            image_overlay_ijk['PD'] = ijk
            image_spacing['PD'] = tuple([float(i) for i in spacing])
            image_origin['PD'] = tuple([float(i) for i in origin[:2]])
            image_origin['ktran'] = image_origin[
                'PD']  # since the offset is changing along the z-axis, use PD and ktran referring to each other is more accurate

    @staticmethod
    def get_ktran_image(ktran_row, path):
        ProxID = ktran_row['ProxID']
        mhd_path = path + '/{0}/{0}-Ktrans.mhd'.format(ProxID)
        zraw_path = path + '/{0}/{0}-Ktrans.zraw'.format(ProxID)
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
        ktran_spacing = itkimage.GetSpacing()
        os.remove('{0}-Ktrans.mhd'.format(ProxID))
        os.remove('{0}-Ktrans.zraw'.format(ProxID))
        return scan, ktran_spacing

    @staticmethod
    def get_csv_iter(path_to_csv):
        metadata = tf.io.gfile.GFile(path_to_csv, 'rb+')
        metadata_csv = pd.read_csv(metadata)
        reader = metadata_csv.iterrows()
        return reader

    @staticmethod
    def get_image_series(num_slice, path, images_location, DCMSerDescr):
        img_list = []
        start, end, step = 0, 0, 0
        if 'cor' or 'sag' in DCMSerDescr:
            start, end, step = 1, num_slice + 1, 1
        else:  # reverse the series of transverse
            start, end, step = num_slice, 0, -1
        for k in range(start, end, step):
            if int(k) < 10 and int(num_slice) >= 10:
                file_name = "1-0{0}.dcm".format(k)
            else:
                file_name = "1-{0}.dcm".format(k)

            whole_path = path + images_location[
                DCMSerDescr.replace("_", "").replace("=", "")] + '/' + file_name
            image = tf.io.gfile.GFile(whole_path, mode='rb')
            image_dc = pydicom.dcmread(image)
            img = image_dc.pixel_array
            img_list.append(np.expand_dims(img, axis=2))
        return img_list

    def resample_image(self, itk_image, ref_image):
        out_spacing = ref_image.GetSpacing()
        out_size = ref_image.GetSize()
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(ref_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
        resample.SetInterpolator(sitk.sitkLinear)  # sitkBSpline)
        return resample.Execute(itk_image)

    def get_itk(self, image, spacing, origin):
        itkimage = sitk.GetImageFromArray(image)
        itkimage.SetSpacing(spacing)
        itkimage.SetOrigin(origin)
        return itkimage

    def get_2d_resample(self, image_overlay, image_spacing, image_origin):
        diff_itk = self.get_itk(image_overlay['diff'], image_spacing['diff'], image_origin['diff'])
        t2_itk = self.get_itk(image_overlay['t2'], image_spacing['t2'], image_origin['t2'])
        PD_itk = self.get_itk(image_overlay['PD'], image_spacing['PD'], image_origin['PD'])
        ktran_itk = self.get_itk(image_overlay['ktran'], image_spacing['ktran'], image_origin['ktran'])
        diff_resample = self.resample_image(diff_itk, ktran_itk)
        t2_resample = self.resample_image(t2_itk, ktran_itk)
        PD_resample = self.resample_image(PD_itk, ktran_itk)
        return np.stack((sitk.GetArrayFromImage(diff_resample), sitk.GetArrayFromImage(t2_resample), sitk.GetArrayFromImage(PD_resample), sitk.GetArrayFromImage(ktran_itk)), axis=-1)

    def get_bbox(self, ijk, shape):
        ijk = ijk.split()
        ymin = (float(ijk[1]) - 5) / shape[0]
        xmin = (float(ijk[0]) - 5) / shape[1]
        ymax = (float(ijk[1]) + 5) / shape[0]
        xmax = (float(ijk[0]) + 5) / shape[1]
        if ymin < 0: ymin = 0.0
        if xmin < 0: xmin = 0.0
        if ymax > 1: ymax = 1.0
        if xmax > 1: xmax = 1.0
        return tfds.features.BBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)

    def get_bbox_list(self, ijk_list, shape):
        bbox_list = []
        for ijk in ijk_list:
            ymin, xmin, ymax, xmax = self.get_bbox(ijk, shape)
            bbox_list.append(tfds.features.BBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax))
        return bbox_list

    def _generate_examples(self, path, file_paths):
        """Yields examples."""
        # TODO(prostatex): Yields (key, example) tuples from the dataset
        path = str(path)
        dycom_path, mhd_path, split = file_paths
        FINDINGS_PATH = path + '/metadata/{0}/ProstateX-Findings-{0}.csv'.format(split)
        GGG_FINDINGS_PATH = path + '/metadata/{0}/ProstateX-2-Findings-{0}.csv'.format(split)
        KTRAN_PATH = path + '/metadata/{0}/ProstateX-Images-KTrans-{0}.csv'.format(split)
        IMAGES_PATH = path + '/metadata/{0}/ProstateX-Images-{0}.csv'.format(split)
        DICOM_METADATA_PATH = path + '/metadata/{0}/metadata.csv'.format(split)

        # open all .csv files to locate desired images
        findings_reader = self.get_csv_iter(FINDINGS_PATH)
        ggg_findings_reader = self.get_csv_iter(GGG_FINDINGS_PATH)
        dicom_metadata_reader = self.get_csv_iter(DICOM_METADATA_PATH)
        images_reader = self.get_csv_iter(IMAGES_PATH)
        ktran_reader = self.get_csv_iter(KTRAN_PATH)

        itera = 0
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
            ProxID, fid, pos, zone = findings_row['ProxID'], findings_row['fid'], findings_row['pos'], \
                                                   findings_row['zone']
            significance = findings_row['ClinSig'] if 'ClinSig' in findings_row else 'TBD'
            ggg_key = '{0};{1};{2}'.format(ProxID, fid, pos)
            ggg = ggg_dic[ggg_key] if ggg_key in ggg_dic else 'N/A'
            # set up the dictionary to map DCMSerDescr to path referred in metadata.csv

            if prevID != ProxID and len(images_location) > 0:  # implicitly ktran scan is not None
                if self.builder_config.name == 'volume':
                    for key in images_location:
                        image_series = self.get_image_series(images_number[key], path + dycom_path, images_location, key)
                        bbox_list = self.get_bbox_list(images_ijk[key], np.shape(image_series[0]))
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
                            'bbox': bbox_list,
                        }
                    ktran_images = []
                    for slice in scan:
                        ktran_slice = np.squeeze(slice)
                        ktran_slice = where(ktran_slice != 0, np.log10(ktran_slice), 1000)
                        ktran_slice = ((ktran_slice + 1000) * 1000).astype('uint16')
                        ktran_images.append(np.expand_dims(ktran_slice, axis=2))
                    bbox_list = self.get_bbox_list(ktran_ijk_list, np.shape(ktran_images[0]))
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
                        'bbox':bbox_list,
                    }
                    significance_list, zone_list, fid_list, pos_list, images_ijk, ktran_ijk_list = [], [], [], [], defaultdict(
                        list), []  # clean the data
                    # of prevID

            fid_list.append(str(fid))
            pos_list.append(pos)
            zone_list.append(zone)
            significance_list.append(significance)
            ggg_list.append(ggg)

            if prevID != ProxID:  # if ProxID equals to prevID, use the last dictionary
                images_location, tmp_dicom_metadata = {}, None
                for d_idx, dicom_metadata_row in dicom_metadata_reader:
                    if dicom_metadata_row['Subject ID'] != ProxID:
                        tmp_dicom_metadata = dicom_metadata_row
                        break
                    metadata_buf_stack.append(dicom_metadata_row)
                len_metadata_buf_stack = len(metadata_buf_stack)
                for i in range(len_metadata_buf_stack):
                    metadata_buf = metadata_buf_stack.pop()
                    images_location[metadata_buf['Series Description']] = metadata_buf['File Location'][11:]
                    images_number[metadata_buf['Series Description']] = int(metadata_buf['Number of Images'])
                if tmp_dicom_metadata is not None: metadata_buf_stack.append(tmp_dicom_metadata)
                # for each finding image, decode it and yield to tfds

            image_overlay = defaultdict(lambda: np.ndarray(0))
            image_overlay_ijk, image_spacing, image_origin = {}, defaultdict(tuple), defaultdict(tuple)
            temp_image = None
            for i_idx, images_row in images_reader:
                if images_row['ProxID'] != ProxID or images_row['fid'] != fid or images_row['pos'] != pos:
                    #  when reading image of next ProxID, process all the images_row in stack for this ProxID
                    temp_image = images_row
                    break
                image_buf_stack.append(images_row)
            len_image_buf_stack = len(image_buf_stack)
            for i in range(len_image_buf_stack):
                image_row = image_buf_stack.pop()
                assert image_row['ProxID'] == ProxID and image_row['fid'] == fid and image_row['pos'] == pos
                DCMSerDes, ijk, image_name = image_row['DCMSerDescr'].replace('_', '').replace('=', ''), image_row[
                    'ijk'], image_row['Name']
                image, spacing, origin = self.get_image(DCMSerDes, ijk, path + dycom_path, images_location, images_number)
                self.add_to_overlay(image_overlay, image_overlay_ijk, image_spacing, image_origin, image_name, ijk, image, spacing, origin)
                DCMSerDes = image_row[-2].replace('_', '').replace('=', '')
                if ijk not in images_ijk[DCMSerDes]:
                    images_ijk[DCMSerDes].append(ijk)
                if self.builder_config.name == 'nostack':
                    bbox = self.get_bbox(ijk, np.shape(image))
                    yield str(ProxID) + str(fid) + image_row['Name'] + str(image_row['DCMSerNum']) + image_row[
                            'pos'], {
                            'image': np.expand_dims(image, axis=2),
                            'significant': significance,
                            'ggg': ggg,
                            'zone': zone,
                            'DCMSerDescr': image_row['DCMSerDescr'],
                            'ProxID': ProxID,
                            'fid': str(fid),
                            'position': pos,
                            'ijk': image_row['ijk'],
                            'bbox': bbox
                    }
            if temp_image is not None: image_buf_stack.append(temp_image)

            # yield KTRAN to tfds

            k_idx, ktran_row = next(ktran_reader)
            assert ktran_row['ProxID'] == ProxID and ktran_row['fid'] == fid
            ijk = ktran_row['ijk'].split()
            scan, ktran_spacing = self.get_ktran_image(ktran_row, path + mhd_path)
            num_slices = scan.shape[0]
            k = int(ijk[-1]) if int(ijk[-1]) < int(num_slices) else int(num_slices) - 1
            k = 0 if k < 0 else k
            ktran_img = np.squeeze(scan[k])
            ktran_img = where(ktran_img != 0, np.log10(ktran_img), 1000)
            image_overlay['ktran'] = ((ktran_img + 1000) * 1000).astype('uint16')  # make ktran scan more contrast
            image_overlay_ijk['ktran'] = ktran_row['ijk']
            image_spacing['ktran'] = ktran_spacing
            if self.builder_config.name == 'nostack':
                bbox = self.get_bbox(ktran_row['ijk'], np.shape(image_overlay['ktran']))
                yield ProxID + str(fid) + pos + '-ktran', {
                    'image': np.expand_dims(image_overlay['ktran'], axis=2),
                    'significant': significance,
                    'ggg': ggg,
                    'zone': zone,
                    'DCMSerDescr': 'ktranFromDCE',
                    'ProxID': ProxID,
                    'fid': str(fid),
                    'position': pos,
                    'ijk': ktran_row['ijk'],
                    'bbox': bbox
                }
            ktran_ijk_list.append(ktran_row['ijk'])

            #  in test split, use ktran's ijk as PD's ijk as they have the same resolution and spacing

            if split == 'Test' and 'PD' not in image_overlay:
                PD_DCMSerDescr = None
                for key in images_location.keys():
                    if 'PD' in key:
                        PD_DCMSerDescr = key
                assert PD_DCMSerDescr is not None
                PD_ijk = image_overlay_ijk['ktran']
                image, spacing, origin = self.get_image(PD_DCMSerDescr, PD_ijk, path + dycom_path, images_location,
                                                images_number)
                self.add_to_overlay(image_overlay, image_overlay_ijk, image_spacing, image_origin, PD_DCMSerDescr, PD_ijk, image,
                                    spacing, origin)
                if ijk not in images_ijk[PD_DCMSerDescr]:
                    images_ijk[PD_DCMSerDescr].append(PD_ijk)
                if self.builder_config.name == 'nostack':
                    bbox = self.get_bbox(PD_ijk, np.shape(image))
                    yield ProxID + str(fid) + PD_DCMSerDescr + pos, {
                        'image': np.expand_dims(image, axis=2),
                        'significant': significance,
                        'ggg': ggg,
                        'zone': zone,
                        'DCMSerDescr': PD_DCMSerDescr,
                        'ProxID': ProxID,
                        'fid': str(fid),
                        'position': pos,
                        'ijk': PD_ijk,
                        'bbox': bbox,
                    }
            #  overlay images and yield
            if self.builder_config.name == 'stack':
                assert len(image_overlay.keys()) == 4
                overlay_image = self.get_2d_resample(image_overlay, image_spacing, image_origin)
                bbox = self.get_bbox(image_overlay_ijk['ktran'], np.shape(overlay_image))
                yield ProxID + str(fid) + pos + '-stack', {
                    'image': overlay_image,
                    'significant': significance,
                    'ggg': ggg,
                    'zone': zone,
                    'DCMSerDescr': 'stackFromDiffT2PDKtran',
                    'ProxID': ProxID,
                    'fid': str(fid),
                    'position': pos,
                    'ijk': image_overlay_ijk['ktran'],
                    'bbox': bbox,
                }

            prevID = ProxID  # update prevID
            itera += 1
            if itera >= 1:
                break

        if self.builder_config.name == 'volume':
            for key in images_location:
                image_series = self.get_image_series(images_number[key], path + dycom_path, images_location, key)
                bbox_list = self.get_bbox_list(images_ijk[key], np.shape(image_series[0]))
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
                    'bbox': bbox_list,
                }
            ktran_images = []
            for slice in scan:
                ktran_slice = np.squeeze(slice)
                ktran_slice = where(ktran_slice != 0, np.log10(ktran_slice), 1000)
                ktran_slice = ((ktran_slice + 1000) * 1000).astype('uint16')
                ktran_images.append(np.expand_dims(ktran_slice, axis=2))
            bbox_list = self.get_bbox_list(ktran_ijk_list, np.shape(ktran_images[0]))
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
                'bbox': bbox_list,
            }
