import copy
import glob
import os.path
import shutil

import cv2
import torch
import tqdm

from osocr_tasks.tasks_gen3.char_classifier.mkdict import v3_dict_from_list
import json;
from neko_sdk.neko_framework_NG.data.supportdb.support_lmdb_builder import neko_support_db_builder
from osocrNG.data_utils.neko_im_label_lmdb_holder import neko_im_label_lmdb_holder,RN
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755

def convert_from_275_inplace(lmdbpath,meta,whitelist=None):
    dbh=neko_im_label_lmdb_holder({
        neko_im_label_lmdb_holder.PARAM_root:lmdbpath
    });
    idict={};
    tdict={};
    for i in tqdm.tqdm(dbh.all_valid_indexes()):
        record=dbh.fetch_item(i);
        text=record[RN.LABEL];
        if(whitelist):
            if(text not in whitelist):
                print(text,"not in whitelist");
                continue;

        img=record[RN.IMAGE];
        if(img is None):
            continue;
        if(text not in idict):
            idict[text]=[];
        idict[text].append(dbh.get_img_key(i));
    torch.save(idict,os.path.join(lmdbpath,"filtered_index.pt"));



if __name__ == '__main__':
    convert_from_275_inplace("/run/media/lasercat/f3a1698e-80ad-4473-8fc6-4df8c81c3831/ssddata/ctwch/ctwfsl20_train/","",t1_3755);
