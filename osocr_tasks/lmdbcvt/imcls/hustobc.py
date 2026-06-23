import json
import os
import glob
import shutil

from kornia.color import lab

from export_cache import export_split
from plotly.graph_objs.layout.newshape import label


def cache_hustobc_split(root_dir, target_root, split):
    with open(os.path.join(root_dir,"code","Validation_label.json"),"r") as fp:
        ldict=json.load(fp);
    with open(os.path.join(root_dir,"code", "Validation_"+split+".json"), "r") as fp:
        fdict=json.load(fp);
    target_split = os.path.join(target_root, split);

    shutil.rmtree(target_split, ignore_errors=True)
    os.makedirs(target_split, exist_ok=True);
    cntr={};
    for f in fdict:
        img=os.path.join(root_dir,f["path"][3:])
        label=str(f["label"]);
        variant=label+"_0";
        dpath=os.path.join(target_split,label,variant)
        os.makedirs(dpath,exist_ok=True);
        if (label not in cntr):
            cntr[label] = 0;
        insid=str(cntr[label]);
        cntr[label]+=1;
        dimg="img_"+os.path.basename(img).replace(".png","").replace("_","-").replace(".","-").replace("#","-")+"_0_"+insid+".png";
        shutil.copy(img,os.path.join(dpath,dimg));
        pass;


    pass;


if __name__ == '__main__':
    # cache_hustobc_split("/run/media/lasercat/writebuffer/hustobc/","/run/media/lasercat/writebuffer/hustobc/cache/","train");
    # cache_hustobc_split("/run/media/lasercat/writebuffer/hustobc/","/run/media/lasercat/writebuffer/hustobc/cache/","val");
    # cache_hustobc_split("/run/media/lasercat/writebuffer/hustobc/", "/run/media/lasercat/writebuffer/hustobc/cache/",
    #                     "test");
    trd=export_split("/run/media/lasercat/writebuffer/hustobc/cache/", "train-igen-0-299","hust-obc-zero");
    # export_split("/run/media/lasercat/writebuffer/hustobc/cache/", "val-igen-0-299","hust-obc-zero",trd);
    # export_split("/run/media/lasercat/writebuffer/hustobc/cache/", "test-igen-0-299","hust-obc-zero",trd);
    # export_split("/run/media/lasercat/writebuffer/hustobc/cache/", "val-0-299","hust-obc-zero",trd);
    # export_split("/run/media/lasercat/writebuffer/hustobc/cache/", "test-0-299","hust-obc-zero",trd);
