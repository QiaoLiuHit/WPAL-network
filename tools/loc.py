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
import cPickle
import os
import pprint
import sys
import time
import numpy as np

import caffe
from wpal_net.config import cfg, cfg_from_file, cfg_from_list
from wpal_net.loc import test_localization, locate_in_video


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='test WPAL-network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device ID to use (default: -1)',
                        default=-1, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional cfg file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set cfg keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--db', dest='db',
                        help='the name of the database',
                        default=None, type=str)
    parser.add_argument('--setid', dest='par_set_id',
                        help='the index of training and testing data partition set',
                        default='0', type=int)
    parser.add_argument('--outputdir', dest='output_dir',
                        help='the directory to save outputs',
                        default='./output', type=str)
    parser.add_argument('--detector-weight', dest='dweight',
                        help='the cPickle file storing the weights of detectors',
                        default=None, type=str)
    parser.add_argument('--display', dest='display',
                        help='whether to display on screen',
                        default=1, type=int)
    parser.add_argument('--max-count', dest='max_count',
                        help='max number of images to perform localization',
                        default=-1, type=int)
    parser.add_argument('--attr-ids', dest='attr_id_list',
                        help='the IDs of the attributes to be located and visualized, '
                             'split by comma. '
                             '-1 for whole body shape. '
                             '-2 for all attributes. ',
                        default=-1, type=str)
    parser.add_argument('--video', dest='video',
                        help='specifying this argument means the program is to perform '
                             'attribute localization on a video but not the database pictures ',
                        default=None, type=str)
    parser.add_argument('--tracking-res', dest='tracking_res',
                        help='when conducting attribute localization on a video, '
                             'pedestrian tracking should be performed in advance, '
                             'and results are input as an input file.',
                        default=None, type=str)

    args = parser.parse_args()

    if args.prototxt is None or args.caffemodel is None or args.db is None or args.dweight is None:
        parser.print_help()
        sys.exit()

    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using cfg:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    if args.db == 'RAP':
        """Load RAP database"""
        from utils.rap_db import RAP

        db = RAP(os.path.join('data', 'dataset', args.db), args.par_set_id)
    else:
        """Load PETA dayanse"""
        from utils.peta_db import PETA

        db = PETA(os.path.join('data', 'dataset', args.db), args.par_set_id)

    f = open(args.dweight, 'rb')
    pack = cPickle.load(f)

    # set up Caffe
    if args.gpu_id == -1:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    if args.video is not None:
        locate_in_video(net,
                        db,
                        args.video, args.tracking_res,
                        args.output_dir,
                        pack['pos_ave'], pack['neg_ave'], pack['binding'],
                        args.attr_id_list)
    else:
        if args.attr_id_list == '-2':
            for a in xrange(db.num_attr):
                test_localization(net, db, args.output_dir, pack['pos_ave'], pack['neg_ave'], pack['binding'],
                                  attr_id=a,
                                  display=args.display,
                                  max_count=args.max_count)
            test_localization(net, db, args.output_dir, pack['pos_ave'], pack['neg_ave'], pack['binding'],
                              attr_id=-1,
                              display=args.display,
                              max_count=args.max_count)
        else:
            iou_all = []
            ior_all = []
            used_img_ind = []
            used_img_label = []
            used_img_pred = []
            pre_all = []
            recall_all = []
            AP_threshold_all = []
            for i in range(0, 51):
                iou_all.append([])
                ior_all.append([])
                used_img_ind.append([])
                used_img_label.append([])
                used_img_pred.append([])
                pre_all.append([])
                recall_all.append([])
                AP_threshold_all.append([])
            for attr_id in args.attr_id_list.split(','):
                # ior_sa, iou_sa, used_img_ind_sa, used_img_label_sa, used_img_pred_sa
                syn_inf = test_localization(net, db,
                                            args.output_dir,
                                            pack[
                                                'pos_ave'],
                                            pack[
                                                'neg_ave'],
                                            pack[
                                                'binding'],
                                            attr_id=int(
                                                attr_id),
                                            display=args.display,
                                            max_count=args.max_count)
                # if len(used_img_ind_sa) == 0:
                if len(syn_inf) == 0:
                    print "No satisfactory img for attribute %d" % attr_id
                else:

                    #                    iou_all[int(attr_id)] = iou_sa
                    #                    ior_all[int(attr_id)] = ior_sa
                    #                    used_img_ind[int(attr_id)] = used_img_ind_sa
                    #                    used_img_label[int(attr_id)] = used_img_label_sa
                    #                    used_img_pred[int(attr_id)] = used_img_pred_sa
                    #                    print "testtesttesttest"
                    #                    print iou_all[int(attr_id)]
                    #                    print ior_all[int(attr_id)]
                    #                    print used_img_ind[int(attr_id)]
                    #                    print used_img_label[int(attr_id)]
                    #                    print used_img_pred[int(attr_id)]

                    print syn_inf
                    syn_inf.sort(reverse=True)
                    print syn_inf
                   # syn_file = os.path.join('/home/yang.zhou/Work/WPAL-network/WPAL-network/output/', 'syn_' + attr_id +
                   #                         'inf.txt')
                   # with open(syn_file, 'w') as f:
                   #     f.write(syn_inf)
                    # AP_threshold = 0.99
                    count_sum = 100
                    if count_sum > len(syn_inf):
                        count_sum = len(syn_inf)
                    while count_sum <= len(syn_inf):
                        pre = 0
                        recall = 0
                        num_pre = 0
                        den_pre = 0
                        num_recall = 0
                        den_recall = 0
                        # for cnt_ap in range(0, len(used_img_ind[int(attr_id)])):
                        for cnt_ap in range(0, count_sum):
                            # if used_img_pred[int(attr_id)][cnt_ap] > AP_threshold:
                            den_pre += 1
                            if (syn_inf[cnt_ap][1] == 1) and (
                                        syn_inf[cnt_ap][2] >= 0.5):
                                num_pre += 1
                                num_recall += 1
                            if syn_inf[cnt_ap][1] == 1:
                                den_recall += 1
                        if den_recall * den_pre == 0:
                            print "TSkiped for error den"
                            if count_sum == len(syn_inf):
                                break
                            else:
                                count_sum += 100
                                if count_sum > len(syn_inf):
                                    count_sum = len(syn_inf)
                                # AP_threshold -= 0.01
                                continue
                        else:
                            pre = float(num_pre) / float(den_pre)
                            recall = float(num_recall) / float(den_recall)
                            pre_all[int(attr_id)].append(pre)
                            recall_all[int(attr_id)].append(recall)
                            # AP_threshold_all[int(attr_id)].append(AP_threshold)
                        # AP_threshold -= 0.01
                        if count_sum == len(syn_inf):
                            break
                        else:
                            count_sum += 100
                            if count_sum > len(syn_inf):
                                count_sum = len(syn_inf)
                                #            for iou_i in range(0, len(miou_all)):
                                #                if len(miou_all[iou_i]) != 0:
                                #                    print "The mean Iou of %d-th attribute in test images is %f" % (iou_i, float(miou_all[iou_i][0]))
                                #            for mop_i in range(0, len(mop_all)):
                                #                if len(mop_all[mop_i]) != 0:
                                #                    print "The mean OP of %d-th attribute in test images is %f" % (mop_i, float(mop_all[mop_i][0]))

            for a_id in range(0, len(pre_all)):
                if len(pre_all[a_id]) != 0:
                    print
                    print "For %d-th attr" % a_id
                    print "__________________________________________________________________________________________"
                    print "AP_threshold:"
                    print AP_threshold_all[int(a_id)]
                    print
                    print "Pre:"
                    print pre_all[int(a_id)]
                    print
                    print "Recall:"
                    print recall_all[int(a_id)]
                    print
                    print "Total: %d" % len(AP_threshold_all[int(a_id)])
