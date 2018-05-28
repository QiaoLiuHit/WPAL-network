#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 3 INCEPTION_BN_SPP_PETA ./data/pretrained/Inception21k.caffemodel ProcessedPeta 1 --set TRAIN.BATCH_SIZE 16
