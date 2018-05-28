#!/usr/bin/env python

# --------------------------------------------------------------------
# This file is part of
# Weakly-supervised Pedestrian Attribute Localization Network.
#
# Weakly-supervised Pedestrian Attribute Localization Network
# is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Weakly-supervised Pedestrian Attribute Localization Network
# is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Weakly-supervised Pedestrian Attribute Localization Network.
# If not, see <http://www.gnu.org/licenses/>.
# --------------------------------------------------------------------

import os.path as osp
import cv2

import numpy as np
import scipy.io as sio

import evaluate
from wpal_net.config import cfg


class RAP:
    def __init__(self, db_path, par_set_id):
        self._db_path = db_path

        rap = sio.loadmat(osp.join(self._db_path, 'RAP_annotation', 'RAP_annotation.mat'))['RAP_annotation']

        self._partition = rap[0][0][0]
        self.labels = rap[0][0][1]
        self.attr_ch = rap[0][0][2]
        self.attr_eng = rap[0][0][3]
        self.num_attr = self.attr_eng.shape[0]
        self.position = rap[0][0][4]
        self._img_names = rap[0][0][5]
        self.attr_exp = rap[0][0][6]

#        self.attr_ch = self.attr_ch[0:51]
#        self.attr_eng = self.attr_eng[0:51]
#        self.num_attr = 51
#        self.labels = []
#        for labels_i in range(0, len(self.labels_all)):
#            self.labels.append([])
#        for labels_i in range(0, len(self.labels_all)):
#            self.labels[labels_i] = self.labels_all[labels_i][0:51]

        self.attr_group = [range(1, 4), range(4, 7), range(7, 9), range(9, 11), range(30, 36), ]

        self.flip_attr_pairs = [(54, 55)]

        self.attr_position_ind = np.zeros(51)
        self.attr_position_ind[9:15] = 1
        self.attr_position_ind[15:24] = 2
        self.attr_position_ind[24:30] = 3
        self.attr_position_ind[30:35] = 3
        self.attr_position_ind[43] = 1

        self.expected_loc_centroids = np.ones(self.num_attr, dtype=int) * 2
        self.expected_loc_centroids[9:16] = 1
        self.expected_loc_centroids[35:43] = 1

        """In our model, labels should be all between 0 and 1.
        Some labels are set to 2 in the RAP database, usually meaning the label is unknown or unsure.
        We change it to 0.5 as a more reasonable value expression.
        """
        self.labels = np.array([[0.5 if x == 2 else x for x in line] for line in self.labels])

        self.test_ind = None
        self.train_ind = None
        self.train_classified_ind = []
        self.train_classified_pre = []
        self.train_classified_b = []
        self.path_tmp = None
        self.img_tmp = None
        self.height = None
        self.width = None
        self.size_ratio_tmp = None
        for i in xrange(11):
            self.train_classified_pre.append([])
        self.label_weight = None
        self.set_partition_set_id(par_set_id)

#    def evaluate_AP(self, attr, inds):
#        cut_attr = [x[0:cfg.TEST.MAX_NUM_ATTR] for x in attr]
#        cut_gt = [x[0:cfg.TEST.MAX_NUM_ATTR] for x in self.labels[inds]]
#        return evaluate.AP

    def evaluate_mA(self, attr, inds):
        cut_attr = [x[0:cfg.TEST.MAX_NUM_ATTR] for x in attr]
        cut_gt = [x[0:cfg.TEST.MAX_NUM_ATTR] for x in self.labels[inds]]
        return evaluate.mA(cut_attr, cut_gt)

    def evaluate_example_based(self, attr, inds):
        cut_attr = [x[0:cfg.TEST.MAX_NUM_ATTR] for x in attr]
        cut_gt = [x[0:cfg.TEST.MAX_NUM_ATTR] for x in self.labels[inds]]
        return evaluate.example_based(cut_attr, cut_gt)

    def set_partition_set_id(self, par_set_id):
        self.train_ind = self._partition[par_set_id][0][0][0][0][0] - 1
        for i in range(0, len(self.train_ind)):
            self.path_tmp = self.get_img_path(self.train_ind[i])
            self.img_tmp = cv2.imread(self.path_tmp)
            self.height, self.width = self.img_tmp.shape[:2]
            self.size_ratio_tmp = round(float(self.height) / float(self.width))
            if i % 1000 == 0:
                print i
            if self.size_ratio_tmp > 10:
                self.size_ratio_tmp = 10
            self.train_classified_pre[int(self.size_ratio_tmp)].append(i)
        print "The size of database is %d. " % (len(self.train_ind))
        for i in xrange(11):
            print "The class %d includes %d pictures. " % (i, len(self.train_classified_pre[i]))
        sumsize = 0
        l = len(self.train_classified_pre) - 1
        while l > -1:
            if len(self.train_classified_pre[l]) == 0:
                l -= 1
                continue
            else:
                break
        for i in range(l + 1):
            self.train_classified_b.append(self.train_classified_pre[i])
            sumsize += len(self.train_classified_b[i])
        batch = cfg.TRAIN.BATCH_SIZE
        if sumsize < batch:
            print "The Size of database is smaller than the batch size you set, try to re-set the param: Batch_Size"
            self.train_classified_ind = self.train_classified_b
        else:
            i = len(self.train_classified_b) - 1
            while i > -1:
                if batch > len(self.train_classified_b[i]) != 0:
                    if i > 1:
                        for j in xrange(len(self.train_classified_b[i])):
                            self.train_classified_b[i - 1].append(self.train_classified_b[i][j])
                        self.train_classified_b[i] = []
                    if i == 1:
                        for j in xrange(len(self.train_classified_b[i])):
                            self.train_classified_b[i + 1].append(self.train_classified_b[i][j])
                        self.train_classified_b[i] = []
                i -= 1
                if i == -1:
                    length = len(self.train_classified_b)
                    l = length - 1
                    while l > -1:
                        if len(self.train_classified_b[l]) == 0:
                            l -= 1
                            continue
                        else:
                            length = l + 1
                            break
                    for x in xrange(length):
                        if len(self.train_classified_b[x]) < batch and len(self.train_classified_b[x]) != 0:
                            i = length - 1
                            break
            for i in xrange(length):
                self.train_classified_ind.append(self.train_classified_b[i])
                print "The class %d in Ind includes %d pics." % (i, len(self.train_classified_ind[i]))

        self.test_ind = self._partition[par_set_id][0][0][0][1][0] - 1
        pos_cnt = sum(self.labels[self.train_ind])
        self.label_weight = pos_cnt / self.train_ind.size

    def get_img_path(self, img_id):
        return osp.join(self._db_path, 'RAP_dataset', self._img_names[img_id][0][0])


if __name__ == '__main__':
    db = RAP('data/dataset/RAP', 0)
    print db._partition.shape
    print db._partition[0][0][0][0][1].shape
    print db._partition[1][0][0][0][1].shape
    print "Labels:", db.labels.shape
    print db.train_ind.shape
    print 'Max training index: ', max(db.train_ind)
    print db.get_img_path(0)
    print db.num_attr
    print db.label_weight
    print db.attr_eng[0][0][0]
    print db.attr_eng[1][0][0]
