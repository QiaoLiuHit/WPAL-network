#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 1 INCEPTION_BN_SPP_PETA ./data/pretrained/Inception21k.caffemodel ProcessedPeta 0 --set TRAIN.BATCH_SIZE 24
