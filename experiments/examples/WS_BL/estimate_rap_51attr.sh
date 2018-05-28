#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/estimate_param.py --setid 0 --net ./data/snapshots/INCEPTION_BN_SPP_RAP_51attr/0/RAP/inception_bn_spp_rap_51attr_iter_950000.caffemodel --def ./models/INCEPTION_BN_SPP_RAP_51attr/test_net.prototxt --gpu 3 --db RAP
