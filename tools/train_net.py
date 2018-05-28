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

import _init_path

import argparse
import os
import sys
import google.protobuf.text_format

import caffe
from wpal_net.config import cfg, cfg_from_file, cfg_from_list
from wpal_net.train import train_net


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='train Weakly-supervised Pedestrian Attribute Localization Network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device ID to use (default: -1 meaning using CPU only)',
                        default=-1, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of training iterations',
                        default=100000, type=int)
    parser.add_argument('--weights', dest='snapshot_path',
                        help='initialize with weights of a pretrained model or snapshot',
                        default=None, type=str)
    parser.add_argument('--db', dest='db',
                        help='name of the databse',
                        default=None, type=str)
    parser.add_argument('--setid', dest='par_set_id',
                        help='the index of training and testing data partition set',
                        default='0', type=int)
    parser.add_argument('--outputdir', dest='output_dir',
                        help='the directory to save outputs',
                        default='./output', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set cfg keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional cfg file',
                        default=None, type=str)

    args = parser.parse_args()

    if args.solver is None or args.db is None:
        parser.print_help()
        sys.exit()

    return args


if __name__ == '__main__':
    print 'Parsing system arguments:', sys.argv
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    # set up Caffe
    if args.gpu_id == -1:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    caffe.set_random_seed(cfg.RNG_SEED)

    if args.db == 'RAP':
        """Load RAP database"""
        from utils.rap_db import RAP
        db = RAP(os.path.join('data', 'dataset', args.db), args.par_set_id)
    else:
        """Load PETA dayanse"""
        from utils.peta_db import PETA
        db = PETA(os.path.join('data', 'dataset',  args.db), args.par_set_id)

    #cfg.NUM_ATTR = db.num_attr

    print 'Output will be saved to `{:s}`'.format(args.output_dir)
    try:
        os.makedirs(args.output_dir)
    except:
        pass

    train_net(args.solver, db, os.path.join(args.output_dir, args.db),
              snapshot_path=args.snapshot_path,
              max_iters=args.max_iters)
