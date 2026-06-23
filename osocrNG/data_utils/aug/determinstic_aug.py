# coding:utf-8
import random
from multiprocessing import get_context

import numpy as np

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_abstract_agent import neko_abstract_async_agent_gen1
from neko_sdk.ocr_modules.augmentation.qhbaug import qhbwarp
from osocrNG.io import neko_aligned_left_top_padding_beacon_np
from osocrNG.data_utils.raw_names import raw_data_item_names as RN


def augment(imgp, rng):
    img = np.array(imgp);
    if (imgp.height < imgp.width * 1.1 and imgp.width < imgp.height * 1.1):
        return img;
    if (imgp.height > imgp.width):
        return qhbwarp(img.transpose(1, 0, 2), 10, rng=rng).transpose(1, 0, 2);
    else:
        return qhbwarp(img, 10, rng=rng);
def process_image(cls,args):
    imgp,rng,w,h,bw,bh=args;
    return neko_aligned_left_top_padding_beacon_np(
        cls.augment(imgp,rng),
        None,h,w,bh,bw);

class determinstic_augmenter:
    @ classmethod
    def augment(cls,imgp,rng):
        img=np.array(imgp);
        if(imgp.height<imgp.width*1.1 and imgp.width<imgp.height*1.1):
            return img;
        if(imgp.height>imgp.width):
            return qhbwarp(img.transpose(1,0,2),10,rng=rng).transpose(1,0,2);
        else:
            return qhbwarp(img,10,rng=rng);
    @classmethod
    def process_image(cls,args):
        imgp,rng,w,h,bh,bw=args;
        return neko_aligned_left_top_padding_beacon_np(
            cls.augment(imgp,rng),
            None,h,w,bh,bw);


    def __init__(this,param):
        this.rng=random.Random(neko_get_arg("seed",param,9));
        this.width=param["width"];
        this.height = param["height"];
        this.bh = neko_get_arg("beacon_w", param, 48);
        this.bw = neko_get_arg("beacon_h", param, 48);

    def process(this,images,thread_pool=None):
        rngs=[random.Random(this.rng.randint(0,0xFFFFFFFFFFFCA71)) for _ in range(len(images))];
        # rngs=[None for _ in range(len(images))];

        hs=[this.height for _ in range(len(images))];
        ws = [this.width for _ in range(len(images))];

        bhs = [this.bh for _ in range(len(images))];
        bws = [this.bw for _ in range(len(images))];

        if(thread_pool is None):
            l=[this.process_image(i) for i in list(zip(images, rngs, ws, hs, bws, bhs))]
        else:
            l=list(thread_pool.map(determinstic_augmenter.process_image,
                                        list(zip(images,rngs,ws,hs,bws,bhs)))
                   );
        # return ;
        return l;


class augment_and_padding_agent(neko_abstract_async_agent_gen1):
    def setup(this,param):
        this.width=param["width"];
        this.height=param["height"];
        augpara=neko_get_arg("augmenter_para",param,{"seed":9});
        this.augmenter_workers=neko_get_arg("augmenter_workers",param,9);
        augpara["width"]=this.width;
        augpara["height"]=this.height;
        augpara["beacon_w"]=param["beacon_w"];
        augpara["beacon_h"] = param["beacon_h"];

        this.augmenter=determinstic_augmenter(augpara);
        # this.augmenter=param["augmenter"];
        this.batch_size=neko_get_arg("batch_size",param,48);

    def action_loop(this):
        if(this.augmenter_workers==0):
            thread_pool=None;
        else:
            thread_pool= get_context("spawn").Pool(this.augmenter_workers);
        while this.status==this.STATUS_running:
            data=[];
            for i in range(this.batch_size):
                d=this.environment.queue_dict["raw_data"].get();
                # print("fetching raw");
                if(d == "NEP_flush_NEP"):
                    if(len(data)):
                        break;
                    else:
                        continue;
                data.append(d);
            il=[i[RN.IMAGE]  for i in data ];
            il=this.augmenter.process(il,thread_pool);
            # Bmask will only be added upon collating. there is no point to add a full 1 mask.
            ddict={
                RN.IMAGE:[],
                RN.BEACON:[], # Beacon is made here bcs you may want to do different stuff
                RN.LABEL:[],
                RN.SIZE:[],
                RN.UID:[], #
            }
            for i in range(len(il)):
                ddict[RN.IMAGE].append(il[i][0]);
                ddict[RN.BEACON].append(il[i][2]);
                ddict[RN.SIZE].append(il[i][3]);
                ddict[RN.LABEL].append(data[i][RN.LABEL]);
                ddict[RN.UID].append(data[i][RN.UID])
            # print("putting_to_collate");
            this.environment.queue_dict["aligned_data"].put(ddict);




class statlmdb:
    def statistic(this):
        from osocrNG.charset.jpn_cset import hira,kata;
        from osocrNG.charset.chs_cset import t1_3755;
        from osocrNG.charset.etc_cset import latin54, digits;

        print(this.nSamples);
        total_dict = {
            "total": 0,
            "Kana":0,
            "digit": 0,
            "t1_chinese": 0,
            "latin54": 0,
            "other":0,
        }
        total_l_dict = {
            "total": 0,
            "Kana":0,
            "digit": 0,
            "t1_chinese": 0,
            "latin54": 0,
            "other": 0,
        }
        for i in range(0, len(this.roots)):
            print("path", this.root_paths[i]);
            print("length", this.lengths[i]);
            char_dict = {
                "total": 0,
                "Kana": 0,
                "digit": 0,
                "t1_chinese": 0,
                "latin54": 0,
                "other":0,
            }
            line_dict = {
            "total": 0,
            "Kana":0,
            "digit": 0,
            "t1_chinese": 0,
            "latin54": 0,
            "other":0,
            }
            if (this.lengths[i] > 160626):
                continue;
            for j in range(this.lengths[i]):
                with this.envs[i].begin(write=False) as txn:
                    label_key = 'label-%09d' % j;
                    try:
                        label = str(txn.get(label_key.encode()).decode('utf-8'));
                    except:
                        print("corrupt", i, j);
                        continue;
                    has_t1ch=0;
                    has_lat = 0;
                    has_dig=0;
                    has_kana=0;
                    has_other=0;
                    for c in label:
                        char_dict["total"] += 1;
                        total_dict["total"] += 1;
                        if (c in t1_3755):
                            char_dict["t1_chinese"] += 1;
                            total_dict["t1_chinese"] += 1;
                            has_t1ch = 1
                        elif (c in latin54):
                            char_dict["latin54"] += 1;
                            total_dict["latin54"] += 1;
                            has_lat=1
                        elif (c in digits):
                            char_dict["digit"] += 1;
                            total_dict["digit"] += 1;
                            has_dig=1;
                        elif (c in hira or c in kata):
                            char_dict["Kana"] += 1;
                            total_dict["Kana"] += 1;
                            has_kana=1;
                        else:
                            char_dict["other"] += 1;
                            total_dict["other"] += 1;
                            has_other=1;
                            print(c);
                line_dict["total"] += 1;
                line_dict["digit"]+=has_dig;
                line_dict["t1_chinese"] += has_t1ch;
                line_dict["latin54"] += has_lat;
                line_dict["other"] += has_other;
                line_dict["Kana"] += has_kana;

                total_l_dict["total"] += 1;
                total_l_dict["digit"] += has_dig;
                total_l_dict["t1_chinese"] += has_t1ch;
                total_l_dict["latin54"] += has_lat;
                total_l_dict["other"] += has_other;
                total_l_dict["Kana"] += has_kana;

            print("chcnt", char_dict);
            print("lncnt", line_dict);


        print("totlncnt", total_l_dict);
        print("totcnt", total_dict);