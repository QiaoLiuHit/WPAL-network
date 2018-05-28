#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/test_net.py --setid 0 --net ./data/snapshots/INCEPTION_BN_SPP_RAP/0/RAP/inception_bn_spp_rap_iter_950000.caffemodel --def ./models/INCEPTION_BN_SPP_RAP/test_net.prototxt --gpu 3 --db RAP --set TEST.MAX_NUM_ATTR 51

