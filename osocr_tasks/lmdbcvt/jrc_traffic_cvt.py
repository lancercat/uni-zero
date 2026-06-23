import json
import os

import cv2

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper

dst="/mnt/shared/traffic/"

root="/run/media/lasercat/data/datasets/fs/traffic";
anno=os.path.join(root,"annotations.json");
with open(anno,"r") as fp:
    anno=json.load(fp);
dbdict={
    "train":im_lmdb_wrapper(os.path.join(dst,"train")),
    "test": im_lmdb_wrapper(os.path.join(dst, "test")),
    "other": im_lmdb_wrapper(os.path.join(dst, "other")),
}
dblset={
    "train":{},
    "test": {},
    "other": {},
}

for img in anno["imgs"]:
    ia=anno["imgs"][img];
    iname=os.path.join(root,ia["path"])
    op=ia["path"].split("/")[0];
    if(len(ia["objects"])==0):
        continue;
    im=cv2.imread(iname);
    cnt=0;
    for obj in (ia["objects"]):
        label=obj["category"];
        ba=obj["bbox"]
        r=int(ba["xmax"]);
        l = int(ba["xmin"]);
        if(l<0):
            l=0;
        t = int(ba["ymin"]);
        if(t<0):
            t=0;
        b = int(ba["ymax"]);
        clip=im[t:b,l:r];
        dname=img+"_"+str(cnt);
        dbdict[op].adddata_kv({"image":clip},{"label":label,"name":dname},{});
        d=dblset[op]
        if(label not in d.keys()):
            d[label]=0;
        d[label]+=1;
        cnt+=1;
        # cv2.imshow("debug",clip);
        # print(label);
        # cv2.waitKey(10);
        pass;
print(dblset);
pass;
