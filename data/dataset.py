import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import logging
import sys
import os
import json  # for dumping json serialized results
import zipfile  # for creating submission zip file
import torch
from pycocotools.coco import COCO  # import COCO class in /site-packages/pycocotools/coco.py
from torch.utils.data import Dataset
# from utils import dump
import cv2 as cv
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


root = 'cowboyoutfits'  # '../input/cowboyoutfits'
logger = logging.getLogger()  # create Logger instance/object(param:name, default:root)
logger.addHandler(
    logging.StreamHandler(sys.stderr))  # output logging information sys.stderr object which support write() and flush()


class CowboyDataset(Dataset):
    def __init__(self, data_dir, transforms=None, train=True, test=False, filter_empty_gt=True):
        super(CowboyDataset, self).__init__()

        self.is_train = train
        self.is_test = test
        self.data_dir = data_dir
        # load training data
        self.coco = COCO(os.path.join(self.data_dir, 'train.json'))
        # print('Data info:')
        # self.coco.info()  # call coco.info(), will get information directly(no need for print())
        # print('Images num:', len(self.coco.dataset['images']))
        # print('Annotations num:', len(self.coco.dataset['annotations']))
        # print(self.coco.dataset['categories'])

        self.categories = {cat_info['id']: cat_info['name'] for cat_info in self.coco.loadCats(self.coco.getCatIds())}
        # print('Categories:', self.categories)
        self.cat2label, self.label2cat = dict(), dict()
        label = 0
        for cat_id in self.categories:
            self.cat2label[cat_id] = label
            self.label2cat[label] = cat_id
            label += 1

        if self.is_train:
            self.img_ids = self.coco.getImgIds()  # get image ids: list in training set

            self.img_infos = []
            total_ann_ids = []
            for img_id in self.img_ids:
                img_info = self.coco.loadImgs([img_id])[0]  # load image information using image id
                self.img_infos.append(img_info)

                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                total_ann_ids.extend(ann_ids)  # extend: make list hashable, append: TypeError: unhashable type: 'list'?
            assert len(total_ann_ids) == len(set(total_ann_ids)), f"Annotation ids in '{data_dir}' are not unique!"

        if self.is_test:
            test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
            self.img_ids = test_df.id.values.tolist()  # test_df.id.values: numpy.ndarray
            self.img_file_names = test_df.file_name.values.tolist()
            self.id2name = {img_id: img_filename for (img_id, img_filename) in zip(self.img_ids, self.img_file_names)}

        self.transforms = transforms
        self.filter_empty_gt = filter_empty_gt

    # call all functions of CowboyDataset instance in main.py?
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox annotations in one image contained in info.

        Args:
            img_info (list[dict):
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes(numpy.ndarray, corner), bboxes_ignore(numpy.ndarray),\
                labels(numpy.ndarray), masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:  # gt_bbox width or height pixel < 1?
                continue
            if ann['category_id'] not in self.categories:
                continue
            '''
            ?
            '''
            inter_w = max(0, min(x1+w, img_info['width'])-max(x1, 0))
            inter_h = max(0, min(y1+h, img_info['height'])-max(y1, 0))
            if inter_w == 0 or inter_h == 0:
                continue
            # if x1 + w > img_info['width'] or y1 + h > img_info['height']:
            #     continue

            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):  # ann may not contain 'iscrowded'
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros(shape=(0, 4), dtype=np.float32)  # no element
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros(shape=(0, 4), dtype=np.float32)

        ann_result = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore
        )
        # ann_result = {
        #     'bboxes': gt_bboxes,
        #     'labels': gt_labels,
        #     'bboxes_ignore': gt_bboxes_ignore
        # }

        return ann_result

    def get_ann_info(self, img_idx):
        img_info = self.img_infos[img_idx]

        img_id = self.img_infos[img_idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])  # ids of all annotations in one image, imgIds: list
        ann_info = self.coco.loadAnns(ids=ann_ids)  # information of all annotations in one image

        return self._parse_ann_info(img_info, ann_info)

    def get_cat_id(self, img_idx):
        img_id = self.img_infos[img_idx]['id']
        ann_id = self.coco.getAnnIds(imgIds=[img_id])  # ids of all annotations in one image, imgIds: list
        ann_info = self.coco.loadAnns(ids=ann_id)  # information of all annotations in one image, ann_info(p.l. of ann)

        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_pixel_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []

        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())  # coco.anna.value() (list[dict]?): annotation

        ids_in_cat = set()
        ids_in_cat |= set(self.coco.catToImgs[class_id] for class_id in self.categories)

        ids_with_gt = ids_with_ann & ids_in_cat

        # ids_with_gt = set()
        # for _ in self.coco.anns.values():
        #     if _['category_id'] in self.categories:
        #         ids_with_gt.add(_['image_id'])

        valid_img_ids = []
        for img_idx, img_info in enumerate(self.img_infos):  # use img_info to calculate area
            img_id = self.img_ids[img_idx]  # self.img_id and self.img_infos have the same indices
            if self.filter_empty_gt and img_id not in ids_with_gt:
                continue
            if img_info['width'] >= min_pixel_size and img_info['height'] >= min_pixel_size:
                valid_inds.append(img_idx)
                valid_img_ids.append(img_id)

        self.img_ids = valid_img_ids  # only keep valid img_ids
        # get valid indices(0 to total_img_num) of images before filtering for choosing info from self.img_infos
        return valid_inds

    def corner2center(self, bbox):
        """Convert corner style bounding boxes to center style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]/ Tensor? : The converted bounding boxes, in ``xywh`` order.
        """
        _bbox = bbox.tolist()

        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1]
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style.

        Args:
            results (list[numpy.ndarray]): region proposals in all images

        Returns:
            list[dict]: dict: region proposals on one image
        """
        json_results = []

        for img_idx in range(len(results)):
            img_id = self.img_ids[img_idx]
            bboxes = results[img_idx]
            for bbox in bboxes:
                bbox_info = dict()
                bbox_info['image_id'] = img_id
                bbox_info['bbox'] = self.corner2center(bbox)
                bbox_info['score'] = float(bbox[4])
                bbox_info['category'] = 1
                json_results.append(bbox_info)

        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style.

        Args:
            results (list[list[numpy.ndarray]]): predicted bboxes in all images

        Returns:
            list[dict]: dict: predicted bboxes on one image
        """
        json_results = []

        for img_idx in range(len(results)):
            img_id = self.img_ids[img_idx]
            for label in range(len(results[img_idx])):
                result = results[img_idx]
                bboxes = result[label]  # result is according to class label order?
                for bbox in bboxes:
                    bbox_info = dict()
                    bbox_info['image_id'] = img_id
                    bbox_info['bbox'] = self.corner2center(bbox)
                    bbox_info['score'] = float(bbox[4])
                    bbox_info['category'] = self.label2cat[label]
                    json_results.append(bbox_info)

        return json_results

    def _results2json(self, results, file_name):
        """Dump/Write the detection results to a COCO style json file.

        Args:
            results (list[tuple]): predicted bboxes in all images
            file_name (str)

        Returns:
            list[dict]: dict: predicted bboxes on one image
        """
        json_results = []

        for result in results:
            img_id, bboxes, labels, scores = result
            # bboxes, labels, scores = bboxes.tolist(), labels.tolist(), scores.tolist()

            for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
                json_result = dict()

                json_result['image_id'] = img_id
                json_result['category_id'] = self.label2cat[label]
                json_result['bbox'] = bbox
                json_result['score'] = score
                json_results.append(json_result)

        with open(file_name, "w") as f:
            for json_result in json_results:
                json.dump(json_result, f)


    # def _results2json(self, results, outfile_prefix):
    #     """Dump/Write the detection results to a COCO style json file.
    #
    #     There are 2 types of results: proposals, bbox predictions,
    #     and they have different data types. This method will
    #     automatically recognize the type, and dump them to json files.
    #
    #     Args:
    #         results (list[list | ndarray]): Testing results of the dataset.
    #
    #         outfile_prefix (str): The filename prefix of the json files. If the
    #             prefix is "somepath/xxx", the json files will be named
    #             "somepath/xxx.bbox.json", "somepath/xxx.proposal.json".
    #
    #     Returns:
    #         dict[str: str]: Possible keys are "bbox", "proposal", and \
    #             values are corresponding filenames.
    #     """
    #     result_files = dict()
    #
    #     if isinstance(results[0], list):
    #         json_results = self._det2json(results)
    #         result_files['bbox'] = f'{outfile_prefix}.bbox.json'
    #         result_files['proposal'] = f'{outfile_prefix}.bbox.json'  # proposal and bbox share the same file_name?
    #         dump(json_results, result_files['bbox'])
    #
    #     elif isinstance(results[0], np.ndarray):
    #         json_results = self._proposal2json(results)
    #         result_files['proposal'] = f'{outfile_prefix}.proposal'
    #         dump(json_results, result_files['proposal'])
    #
    #     else:
    #         raise TypeError("results type is not valid")
    #
    #     return result_files

    def __getitem__(self, img_idx: int):
        '''
        Get one image and its ground truth bounding boxes.

        Args:
            img_idx (int): image index in self.img_infos and self.img_ids

        Returns:
            tuple: (img_id, image, target), image is a float32 numpy.ndarray, \
                representing normalized image pixel intensity, target is a dict \
                containing bboxes, labels and image_idx
        '''
        img_id = self.img_ids[img_idx]

        if self.is_train:
            img_info = self.img_infos[img_idx]
            img_name = img_info['file_name']
        if self.is_test:
            img_name = self.img_file_names[img_idx]

        image = cv.imread(f'{self.data_dir}/images/{img_name}', cv.IMREAD_COLOR)  # ignore alpha channel?
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # image.shape: (width, height, channel)
        # image = torch.tensor(image, dtype=torch.float32)  # only Tensor can be used on gpu(.to(device))

        if self.is_train:
            ann_info = self.get_ann_info(img_idx)

            bboxes = ann_info['bboxes']  # numpy.ndarray: shape=(num_bboxes, 4), corner
            bboxes = torch.tensor(bboxes, dtype=torch.float64)

            labels = ann_info['labels']  # numpy.ndarray: shape=(num_bboxes, )
            labels = torch.tensor(labels, dtype=torch.int64)  # Tensor

            # keys of target are fixed because of using torchvision
            target = dict()
            target['boxes'] = bboxes  # must be in corner representation
            target['labels'] = labels
            target['image_idx'] = torch.tensor([img_idx])

            if self.transforms:
                sample = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels,
                }
                sample = self.transforms(**sample)
                image = sample['image']
                # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

            return img_id, image, target

        if self.is_test:
            if self.transforms:
                sample = {
                    'image': image
                }
                sample = self.transforms(**sample)
                image = sample['image']

            return img_id, image

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def get_train_transform():
        return A.Compose([
            ToTensorV2(p=1.0)  # change image and bbox to Tensor?  A.Flip(p=0.5)?
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

    @staticmethod
    def get_test_transform():
        return A.Compose([
            ToTensorV2(p=1.0)
        ])


# class CowboyTestDataset(Dataset):
#     def __init__(self):
#         super(CowboyTestDataset, self).__init__()

if __name__ == '__main__':
    c = CowboyDataset('cowboyoutfits')
