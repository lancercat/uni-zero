import json;
import os;
import shutil;

import cv2;

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper;


def make_art_lmdb_wval(root,dst):
    valdst = os.path.join(dst, "val");
    trndst = os.path.join(dst, "train");

    trpath = os.path.join(root, "train_task2_labels.json");
    jdict = None

    shutil.rmtree(dst, True);
    db = im_lmdb_wrapper(trndst);
    valdb = im_lmdb_wrapper(valdst);
    idx = 0;

    with open(trpath, "r") as fp:
        jdict = json.load(fp);

    for i in jdict:
        idx += 1;
        impath = os.path.join(root, "train_task2_images", i + ".jpg");
        if (len(jdict[i]) > 1 or jdict[i][0]['illegibility']):
            print(len(jdict[i]));
            continue;
        im = cv2.imread(impath);
        gt = jdict[i][0]['transcription'];
        lang = jdict[i][0]['language'];
        if (idx % 9 == 0):
            valdb.add_data_utf(im, gt, lang);
        else:
            db.add_data_utf(im, gt, lang);
    cnt = 0;
    db.end_this();
    valdb.end_this();



def make_art_lmdb(root,dst):
    trndst = os.path.join(dst, "train");

    trpath = os.path.join(root, "train_task2_labels.json");

    shutil.rmtree(dst, True);
    db = im_lmdb_wrapper(trndst);
    idx = 0;

    with open(trpath, "r") as fp:
        jdict = json.load(fp);

    for i in jdict:
        idx += 1;
        impath = os.path.join(root, "train_task2_images", i + ".jpg");
        if (len(jdict[i]) > 1 or jdict[i][0]['illegibility']):
            print(len(jdict[i]));
            continue;
        im = cv2.imread(impath);
        gt = jdict[i][0]['transcription'];
        lang = jdict[i][0]['language'];
        db.add_data_utf(im, gt, lang);
    cnt = 0;
    db.end_this();

