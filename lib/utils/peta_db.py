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
from wpal_net.config import cfg
import numpy as np
import scipy.io as sio

import evaluate


class PETA:
    """This tool requires the PETA to be processed into similar form as RAP."""

    def __init__(self, db_path, par_set_id):
        self._db_path = db_path

        try:
            self.labels = sio.loadmat(osp.join(self._db_path, 'attributeLabels.mat'))['DataLabel']
        except NotImplementedError:
            import h5py
            print h5py.File(osp.join(self._db_path, 'attributeLabels.mat')).keys()
            self.labels = np.array(h5py.File(osp.join(self._db_path, 'attributeLabels.mat'))['DataLabel']).transpose()

        try:
            self.name = sio.loadmat(osp.join(self._db_path, 'attributesName.mat'))['attributesName']
        except NotImplementedError:
            import h5py
            print h5py.File(osp.join(self._db_path, 'attributesName.mat')).keys()
            self.name = h5py.File(osp.join(self._db_path, 'attributesName.mat'))['attributesName']

        try:
            self._partition = sio.loadmat(osp.join(self._db_path, 'partition.mat'))['partition']
        except NotImplementedError:
            import h5py
            print h5py.File(osp.join(self._db_path, 'partition.mat')).keys()
            self.name = h5py.File(osp.join(self._db_path, 'partition.mat'))['partition']

        self.num_attr = self.name.shape[0]
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
        self.attr_group = [range(0, 4)]
        self.flip_attr_pairs = []  # The PETA database has no symmetric attribute pairs.

        self.expected_loc_centroids = np.ones(self.num_attr) * 2

    def evaluate_mA(self, attr, inds):
        return evaluate.mA(attr, self.labels[inds])

    def evaluate_example_based(self, attr, inds):
        return evaluate.example_based(attr, self.labels[inds])

    def set_partition_set_id(self, par_set_id):
        #self.train_ind = self._partition[par_set_id][0][0][0][0][0] - 1
        #self.test_ind = self._partition[par_set_id][0][0][0][1][0] - 1
        self.train_ind = filter(lambda x:x%5!=par_set_id, xrange(self.labels.shape[0]))
         
        for i in range(0, len(self.train_ind)):
            self.path_tmp = self.get_img_path(self.train_ind[i])
            self.img_tmp = cv2.imread(self.path_tmp)
            self.height, self.width = self.img_tmp.shape[:2]
            self.size_ratio_tmp = round(float(self.height) / float(self.width))
            if i%1000 == 0:
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

        self.test_ind = filter(lambda x:x%5==par_set_id, xrange(self.labels.shape[0]))
        
        pos_cnt = sum(self.labels[self.train_ind])
        self.label_weight = pos_cnt / len(self.train_ind)

    def get_img_path(self, img_id):
        return osp.join(self._db_path, 'Data', str(img_id + 1) + '.png')


if __name__ == '__main__':
    db = PETA('data/dataset/ProcessedPeta', 1)
    print "Labels:", db.labels.shape
    print len(db.train_ind)
    print len(db.test_ind)
    print 'Max training index: ', max(db.train_ind)
    print db.get_img_path(0)
    print db.num_attr
    print db.train_ind
    print db.test_ind
    print db.label_weight
