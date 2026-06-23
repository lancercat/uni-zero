import glob
import json
import os
import shutil

import pylcs

from neko_sdk.visualization.compile_ocr import getres


# A function to bootstrap a result folder.
def accrfolder(root,filter,dst,thresh,arcntr,case_sensitive=False,ignore=None,unk_token="⑨"):
    arcntr.clear();
    files=glob.glob(os.path.join(root,"*.txt"));
    tot =0;
    corr=0;
    tned=0;
    tlen=0;
    for f in files:
        gt,pr=getres(f);
        if(ignore is not None):
            ignf = False;
            for ii in ignore:
                if gt.find(ii)!=-1:
                    ignf=True;
                    break;
            if(ignf):
                continue;
        if(not case_sensitive):
            gt=gt.lower();
            pr=pr.lower();
        tlen+=len(gt);
        if(not filter(gt)):
            continue;
        arcntr.add_iter([pr],[gt],[gt]);
        infoname=os.path.join(f.replace("txt","json"));
        infodict={"gt":gt,"pr":pr,"unk":unk_token};
        gt_no_unk=gt.replace(unk_token,"");
        pred_no_unk=pr.replace(unk_token,"");

        ned_wunk = 1 - pylcs.edit_distance(gt, pr) / len(gt);
        ned_no_unk = 1 - pylcs.edit_distance(gt, pr) / len(gt);
        if(ned_wunk>0.999 and gt!=pr):
            ned_wunk=0; # a pylcs bug.
        infodict["ned_wunk"]=ned_wunk;
        infodict["ned_no_unk"] = ned_no_unk;
        infodict["correct"]=(gt==pr);
        infodict["correct_no_unk"]=(gt_no_unk==pred_no_unk);
        with open(infoname,"w") as fp:
            json.dump(infodict,fp);
        for t in thresh:
            dfolder = os.path.join(dst, str(t));
            os.makedirs(dfolder,exist_ok=True);
        for t in thresh:
            try:
                if(ned_wunk*10+9e-9>=t):
                    dfolder=os.path.join(dst,str(t));
                    shutil.copy(f,dfolder);
                    shutil.copy(f.replace(".txt",".jpg"),dfolder);
                    break
                    pass;
            except:
                pass;
        tned+=ned_wunk
        if (gt==pr):
            corr+=1;
        else:
            # print(gt,pr);
            pass;
        tot+=1;
    return arcntr.show();

def detailed(root,workdir,prefix,ks,threshs,filters,arcntr,texcomp,ignore=None):
    rec_dict={}
    for k in ks:
        dst=os.path.join(workdir,k)
        try:
            shutil.rmtree(dst);
        except:
            pass;
        os.makedirs(os.path.join(workdir,k),exist_ok=True);
        for t in threshs:
            os.makedirs(os.path.join(workdir, k,str(t)), exist_ok=True);
        if(k not in filters):
            continue;
        rec_dict[k]=accrfolder(root,filters[k],dst,threshs,arcntr,ignore=ignore);
    return rec_dict;

