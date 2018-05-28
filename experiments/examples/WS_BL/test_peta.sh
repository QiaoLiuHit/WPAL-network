#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/test_net.py --setid 0 --net ./data/snapshots/INCEPTION_BN_SPP_PETA/0/ProcessedPeta/inception_bn_spp_peta_iter_125000.caffemodel --def ./models/INCEPTION_BN_SPP_PETA/test_net.prototxt --gpu 3 --db ProcessedPeta
