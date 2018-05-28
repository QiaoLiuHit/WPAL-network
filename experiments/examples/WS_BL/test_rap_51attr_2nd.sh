#!/usr/bin/env bash
cd `dirname "${BASH_SOURCE[0]}"`/../../..
./tools/test_net.py --setid 0 --net ./data/snapshots/WS_BL_2th/0/RAP/ws_bl_2th_iter_950000.caffemodel --def ./models/WS_BL_2th/test_net.prototxt --gpu 3 --db RAP --set TEST.MAX_NUM_ATTR 51

