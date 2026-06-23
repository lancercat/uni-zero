#!/usr/bin/env python
# -*- coding:utf-8 -*-
###################################################
#      Filename: convert.py
#        Author: lzw.whu@gmail.com
#       Created: 2017-11-16 10:34:55
# Last Modified: 2017-11-23 18:41:54
###################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os;
from collections import defaultdict
from struct import pack

import numpy as np
from PIL import Image, ImageDraw
from neko_sdk.lmdbcvt import hwdb_sample_data

from neko_sdk.lmdb_wrappers.lmdb_wrapper import lmdb_wrapper;


def make_olhwdb (pot_dir, lmdb_dir):
    db=lmdb_wrapper(lmdb_dir);
    files = []
    all_tagcode = defaultdict(int)
    for tagcode, strokes,fn in hwdb_sample_data.read_from_pot_dir(pot_dir):
        all_tagcode[tagcode] += 1
        im = Image.new("L", (10240, 10240), 255)
        draw = ImageDraw.Draw(im)
        mins = []
        maxs = []
        for stroke in strokes:
            mins.append(np.min(stroke, 0))
            maxs.append(np.max(stroke, 0))
            draw.line(stroke, fill=0, width=4)
        del draw
        _min = np.min(mins, 0) - 2
        _max = np.max(maxs, 0) + 2
        box = (_min[0], _min[1], _max[0], _max[1])
        wrid=os.path.basename(fn).split(".")[0];
        im = im.crop(box)
        char = pack('>H', tagcode).decode('gb2312');
        db.adddata_kv({"image":im},
                      {"label":char,"lang":"Chinese","wrid":wrid},
                      {});

def make_hwdb (gnt_dir, lmdb_dir):
    db=lmdb_wrapper(lmdb_dir);
    files = []
    all_tagcode = defaultdict(int)
    for tagcode, im,fn in hwdb_sample_data.read_from_gnt_dir(gnt_dir):
        all_tagcode[tagcode] += 1
        wrid=os.path.basename(fn).split(".")[0];
        try:
            char = pack('>H', tagcode)[::-1].decode('gbk')
            db.adddata_kv({"image":im},
                          {"label":char,"lang":"Chinese","wrid":wrid},
                          {});
        except:
            print(fn,tagcode);

#

