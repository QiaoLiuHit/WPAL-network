#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 0 INCEPTION_BN_GMP_PETA ./data/pretrained/Inception21k.caffemodel ProcessedPeta 0 --set TRAIN.BATCH_SIZE 48
