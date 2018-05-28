#!/usr/bin/env python
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

"""The data layer used during training to train a WPAL-network.
DataLayer implements a Caffe Python layer.
"""

from multiprocessing import Process, Queue

import numpy as np
from data_layer.minibatch import get_minibatch
from wpal_net.config import cfg

import caffe


class DataLayer(caffe.Layer):
    """WPAL-network data layer used for training."""

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next mini-batch.
        """
        return self._blob_queue.get()

    def set_db(self, db, do_flip):
        """Set the database to be used by this layer during training."""
        self._db = db

        """Enable prefetch."""
        self._blob_queue = Queue(32)
        self._prefetch_process = BlobFetcher(self._blob_queue, self._db, do_flip)
        self._prefetch_process.start()

        # Terminate the child process when the parent exists
        def cleanup():
            print 'Terminating BlobFetcher'
            self._prefetch_process.terminate()
            self._prefetch_process.join()

        import atexit
        atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the DataLayer."""
        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.BATCH_SIZE, 3, 224, 224)
        self._name_to_top_map['data'] = idx
        idx += 1

        top[idx].reshape(cfg.TRAIN.BATCH_SIZE, cfg.NUM_ATTR)
        self._name_to_top_map['attr'] = idx
        idx += 1

        top[idx].reshape(cfg.TRAIN.BATCH_SIZE, cfg.NUM_ATTR)
        self._name_to_top_map['weight'] = idx
        idx += 1

        print 'DataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, queue, db, do_flip=True):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._db = db
        self._perm = None
        self._perm_raw = None
        self._perm_classified = []
        for i in xrange(11):
            self._perm_classified.append([])
        self._cur = 0
        self._num_raw = 0
        self._do_flip = do_flip
        self._train_ind = self._db.train_ind
        self._train_classified_ind = self._db.train_classified_ind
        self._weight = db.label_weight

        self._shuffle_train_inds()

        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_train_inds(self):
        """Randomly permute the training database."""

        for i in xrange(len(self._train_classified_ind)):
            self._perm_classified[i] = np.random.permutation(xrange(len(self._train_classified_ind[i])))
        self._perm_raw = np.random.permutation(xrange(len(self._train_classified_ind)))
        # self._perm = np.random.permutation(xrange(len(self._train_ind[2]) * (2 if self._do_flip else 1)))
        self._cur = 0
        self._num_raw = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        while self._cur >= len(self._db.train_classified_ind[self._perm_raw[self._num_raw]]):
        # if self._cur >= len(self._db.train_classified_ind[self._perm_raw[self._num_raw]]):
            self._num_raw += 1
            if self._num_raw >= len(self._train_classified_ind):
                break
            self._cur = 0
        if self._num_raw >= len(self._train_classified_ind):
            self._shuffle_train_inds()
        minibatch_inds = []
        ratio = self._perm_raw[self._num_raw]
        for i in range(self._cur, self._cur + cfg.TRAIN.BATCH_SIZE
                       if self._cur+cfg.TRAIN.BATCH_SIZE <= len(self._perm_classified[self._perm_raw[self._num_raw]])
                       else len(self._perm_classified[self._perm_raw[self._num_raw]])):
            minibatch_inds.append(self._db.train_classified_ind[self._perm_raw[self._num_raw]][self._perm_classified[self._perm_raw[self._num_raw]][i]])
        #minibatch_inds = self._db.train_classified_ind[self._perm_raw[self._num_raw]][self._perm_classified[self._perm_raw[self._num_raw]][self._cur:self._cur + cfg.TRAIN.BATCH_SIZE]]
        self._cur += cfg.TRAIN.BATCH_SIZE
        return minibatch_inds, ratio

    def run(self):
        print 'BlobFetcher started'
        while True:
            minibatch_inds, img_ratio = self._get_next_minibatch_inds()

            if len(minibatch_inds) < cfg.TRAIN.BATCH_SIZE:
                continue

            #print "The ratio of this blob is: %d" % img_ratio
            #print "Size of minibatch %d" % len(minibatch_inds)
            #print minibatch_inds
            #print
            minibatch_img_paths = \
                [self._db.get_img_path(self._db.train_ind[i])
                 for i in minibatch_inds]
            minibatch_labels = \
                [self._db.labels[self._db.train_ind[i]]
                 for i in minibatch_inds]
            minibatch_flip = \
                np.random.random_integers(0, 1, len(minibatch_inds))
            blobs = get_minibatch(minibatch_img_paths, minibatch_labels, minibatch_flip, self._db.flip_attr_pairs,
                                  self._weight, img_ratio)
            self._queue.put(blobs)
