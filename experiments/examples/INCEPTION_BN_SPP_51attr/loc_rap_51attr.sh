#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/loc.py --setid 0 --net ./data/snapshots/INCEPTION_BN_SPP_RAP_51attr/0/inception_bn_spp_rap_51attr_iter_950000.caffemodel --def ./models/INCEPTION_BN_SPP_RAP_51attr/test_net.prototxt --db RAP --detector-weight ./output/rap_448_detector.pkl --attr-id -2 --cfg experiments/cfgs/spp.yml --display 0 --max-count 5
