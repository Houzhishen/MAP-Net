from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import torch
import pickle


def load_images(data_file):
    data = np.load(data_file)
    fields = data['features'], data['targets']

    return fields


def load_attributes(data_file):
    data = np.load(data_file)
    fields = data['features'], data['targets']

    return fields


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class cub_load(object):
    """
    Dataset statistics:
    # 64 * 600 (train) + 16 * 600 (val) + 20 * 600 (test)
    """
    dataset_dir = '/home/abc/Datasets/cub/CUB_200_2011/npz/'

    def __init__(self, **kwargs):
        super(cub_load, self).__init__()
        self.train_dir = os.path.join(self.dataset_dir, 'few-shot-train.npz')
        self.val_dir = os.path.join(self.dataset_dir, 'few-shot-val.npz')
        self.test_dir = os.path.join(self.dataset_dir, 'few-shot-test.npz')

        self.train_emb_dir = os.path.join(self.dataset_dir, 'attributes-train.npz')
        self.val_emb_dir = os.path.join(self.dataset_dir, 'attributes-val.npz')
        self.test_emb_dir = os.path.join(self.dataset_dir, 'attributes-test.npz')

        self.train, self.train_labels2inds, self.train_labelIds = self._process_dir(self.train_dir, self.train_emb_dir)
        self.val, self.val_labels2inds, self.val_labelIds = self._process_dir(self.val_dir, self.val_emb_dir)
        self.test, self.test_labels2inds, self.test_labelIds = self._process_dir(self.test_dir, self.test_emb_dir)

        self.num_train_cats = len(self.train_labelIds)
        num_total_cats = len(self.train_labelIds) + len(self.val_labelIds) + len(self.test_labelIds)
        num_total_imgs = len(self.train + self.val + self.test)

        print("=> CUB loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # cats | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(len(self.train_labelIds), len(self.train)))
        print("  val      | {:5d} | {:8d}".format(len(self.val_labelIds),   len(self.val)))
        print("  test     | {:5d} | {:8d}".format(len(self.test_labelIds),  len(self.test)))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_cats, num_total_imgs))
        print("  ------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _get_pair(self, images, labels, attributes, attributes_label):
        assert (images.shape[0] == len(labels))
        data_pair = []
        for i in range(images.shape[0]):
            a = np.argwhere(attributes_label == labels[i])[0][0]
            # print(a)
            data_pair.append((images[i], attributes[a], labels[i]))
        return data_pair

    def _process_dir(self, image_path, attribute_path):
        print(image_path)
        images = load_images(image_path)
        attributes = load_attributes(attribute_path)
        # print(set(images[1]))
        # print(wordembs.shape)

        data_pair = self._get_pair(images[0], images[1], attributes[0], attributes[1])
        labels2inds = buildLabelIndex(images[1])
        labelIds = sorted(labels2inds.keys())
        return data_pair, labels2inds, labelIds

if __name__ == '__main__':
    cub_load()
    # a = np.array([1,2,3,4,5,6])

    # n = np.argwhere(a==4)
    # print(a[n])
