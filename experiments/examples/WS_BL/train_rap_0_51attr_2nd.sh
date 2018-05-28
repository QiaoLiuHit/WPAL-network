#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./experiments/scripts/wpal_net.sh 2 WS_BL_2th ./data/pretrained/bvlc_googlenet.caffemodel RAP 0 --set TRAIN.BATCH_SIZE 32
