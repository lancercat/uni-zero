import glob
import os;
import random
import shutil

import cv2;
import numpy as np;
import torch;
import tqdm
from neko_sdk.ocr_modules.charset.jpn_filters import with_hirakata,ukanji,seen

from neko_sdk.ocr_modules.fontkit.fntmgmt import make_meta

from neko_sdk.ocr_modules.charset.chs_cset import common_999, t1_3755
from neko_sdk.ocr_modules.trdg_driver.corpus_data_generator_driver import neko_random_string_generator;

from neko_sdk.ocr_modules.fontkit.fntmgmt import fntmgr
from osocr_tasks.lmdbcvt.cropwordcvt import crop_word_export

from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize
from neko_sdk.ocr_modules.trdg_driver.drive_core import neko_basic_string_generator_core

import torch

from osocrNG.data_utils.raw_names import raw_data_item_names as RN
from osocrNG.data_utils.neko_im_label_lmdb_holder import neko_im_label_lmdb_holder
import tqdm
def get_all_text(dbroot):
    holder=neko_im_label_lmdb_holder({neko_im_label_lmdb_holder.PARAM_root:dbroot});
    all_labels=[];
    for id in tqdm.tqdm(holder.all_valid_indexes()):
        item=holder.fetch_item(id);
        if(item is not None):
            all_labels.append(item[RN.LABEL]);
    return all_labels;


def mk_rand_by_uniform(corpus_dict,min_len=1,max_len=10,cnt=300):
    text_dict={};
    charset_dict={};
    token_freq = {};
    for k in corpus_dict:
        token_freq[k] = {};
        for s in corpus_dict[k]:
            for t in tokenize(s):
                if (t not in token_freq[k]):
                    token_freq[k][t] = 0;
                token_freq[k][t] += 1;
    for k in corpus_dict:
        charset_dict[k]=list(token_freq[k].keys());
    for k in charset_dict:
        for l in range(min_len,max_len):
            texts=[];
            for id in range(cnt):
                sample_keys_weighted_multiple = np.random.choice(charset_dict[k], size=l, replace=True)
                text="".join(sample_keys_weighted_multiple);
                texts.append(text);
            text_dict["-".join(["uniform",k,str(l)])]=texts;
    return text_dict;

def mk_rand_by_freq(corpus_dict,min_len=1,max_len=10,cnt=300):
    text_dict={};
    token_freq = {};
    for k in corpus_dict:
        token_freq[k] = {};
        for s in corpus_dict[k]:
            for t in tokenize(s):
                if (t not in token_freq[k]):
                    token_freq[k][t] = 0;
                token_freq[k][t] += 1;
    for k in corpus_dict:
        for l in range(min_len,max_len):
            texts=[];
            for id in range(cnt):
                keys = list(token_freq[k].keys());
                frequencies = list(token_freq[k].values());
                probabilities = np.array(frequencies) / sum(frequencies);
                sample_keys_weighted_multiple = np.random.choice(keys, size=l, p=probabilities, replace=True)
                text="".join(sample_keys_weighted_multiple);
                texts.append(text);
            text_dict["-".join(["freq",k,str(l)])]=texts
    return text_dict;
def mk_copy(corpus_dict,min_len=1,max_len=10,cnt=300):
    text_dict={};
    for k in corpus_dict:
        for l in range(min_len,max_len):
            texts=[];
            for t in corpus_dict[k]:
                if(len(t)==l):
                    texts.append(t);
            if(len(texts)<cnt):
                texts+=list(np.random.choice(texts,size=cnt-len(texts),replace=True));
            text_dict["-".join(["full",k,str(l)])]=texts[:cnt];
    return text_dict;

def render_k(droot,d,k):
    bgims=torch.load("/home/lasercat/ssddata/synth_lsct/bgim.pt");
    g = neko_basic_string_generator_core(bgims["bgims"]);
    g.skewing_angle = [0, 0, 0, 2];
    g.background_types = [0];
    i = 0;
    text=d[k];
    fonts=['/home/lasercat/ssddata/synth_lsct/fonts-notocommon/NotoSansCJK-Regular.ttc' for _ in text];
    dst=os.path.join(droot,k);
    shutil.rmtree(dst,ignore_errors=True);
    os.makedirs(dst);
    for c, f in zip(text,fonts):
        pil_image, gt = g.mk_clip(f, c);
        if (pil_image is None):
            continue;
        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        # cv2.imshow("baz", open_cv_image);
        # print(gt);
        # cv2.waitKey(10);

        imfn = os.path.join(dst, str(i) + ".jpg");
        txtfn = os.path.join(dst, "gt_" + str(i) + ".txt");
        cv2.imwrite(imfn, open_cv_image);
        with open(txtfn, "w+") as fp:
            fp.writelines([gt]);
        i += 1;
def make_dataset():
    a=torch.load("/run/media/lasercat/ssddata2/synth/corpus.pt");
    shutil.rmtree("/run/media/lasercat/ssddata2/controlled_synth/",ignore_errors=True);
    jptxt=a["jpn"];
    ukanji_txt=[];
    kana_txt=[];
    seen_txt=[];
    for t in jptxt:
        if(with_hirakata(t)):
            kana_txt.append(t);
        if(ukanji(t)):
            ukanji_txt.append(t);
        if(seen(t)):
            seen_txt.append(t);
    a["jpn-kana"]=kana_txt;
    a["jpn-ukanji"]= ukanji_txt;
    a["jpn-seen"]= seen_txt;

    cd={};
    cd.update(mk_copy(a,1,11));
    cd.update(mk_rand_by_freq(a,1,11));
    cd.update(mk_rand_by_uniform(a,1,11));
    for k in cd:
        render_k("/run/media/lasercat/ssddata2/controlled_synth/",cd,k);
    pass;
def export_lmdb(root,droot):
    ads=glob.glob(os.path.join(root,"*"));
    for d in ads:
        l=os.path.join(droot,os.path.basename(d));
        crop_word_export.mkfolder(d,l);
def export_combined_lmdb(root,droot):
    rng=["full","uniform","freq"];
    lang=["jpn","kr","seen","jpn-kana","jpn-ukanji","jpn-seen"];
    for r in rng:
        for la in lang:
            dst=os.path.join(droot,"-".join([r,la,"combined"]));
            ads = glob.glob(os.path.join(root, "-".join([r,la,"*"])));
            crop_word_export.mk_combined_folder(ads,dst);

if __name__ == '__main__':
    make_dataset();
    shutil.rmtree("/home/lasercat/ssddata/controlled_synth_lmdb/",ignore_errors=True);
    os.makedirs("/home/lasercat/ssddata/controlled_synth_lmdb/");
    export_lmdb("/run/media/lasercat/ssddata2/controlled_synth/","/home/lasercat/ssddata/controlled_synth_lmdb/");
    export_combined_lmdb("/run/media/lasercat/ssddata2/controlled_synth/","/home/lasercat/ssddata/controlled_synth_lmdb/")




# if __name__ == '__main__':
#     jptxt=get_all_text("/home/lasercat/ssddata/mlttrjp_hv");
#     krtxt=get_all_text("/home/lasercat/ssddata/mlttrkr_hori/");
#     lsvt_txt=get_all_text("/home/lasercat/ssddata/lsvtdb_seen_NG");
#     torch.save({
#         "jpn":jptxt,
#         "kr":krtxt,
#         "seen":lsvt_txt},"/run/media/lasercat/ssddata2/synth/corpus.pt");
#     pass;