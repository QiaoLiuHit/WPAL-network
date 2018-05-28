#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 3 GOOGLENET_BN_SPP_RAP ./data/pretrained/googlenet_bn_stepsize_6400_iter_1200000.caffemodel RAP 0 --set TRAIN.BATCH_SIZE 16
