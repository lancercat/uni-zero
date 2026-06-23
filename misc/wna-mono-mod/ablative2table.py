import os

import numpy as np
import torch

from ablative import regdict,texdict
def extract_accrs(d):
    all=[];
    for k in d["JPNHV-JPNHV-GZSL"]:
        all.append(d["JPNHV-JPNHV-GZSL"][k]["case_inv_acr_time"]["ACR"]);
    return all;

def mktex_one(num,tag,isep="-",tsep="o"):
    return "\\newcommand{\\"+tag.replace(isep,tsep)+"}{"+"{:.2f}".format(num)+"}\n";
def mktex(mean,std,mname,mflag="AVG",sflag="STD",isep="-",tsep="o"):
    return [
        mktex_one(mean*100,mname+isep+mflag,isep,tsep),
        mktex_one(std*100, mname + isep + sflag, isep, tsep),
    ];

if __name__ == '__main__':
    root=os.getcwd();
    at=[];
    add={};
    for k in regdict:
        allacr = [];
        add[k]={};
        for run in regdict[k]:
            rdic=torch.load(os.path.join(root,"object310-rel",run,"ablruns.pt"),weights_only=False);
            rr=extract_accrs(rdic);
            allacr+=rr;
            add[k][run]=rr;
        at+=mktex(np.mean(allacr),np.std(allacr),texdict[k]);
    for l in at:
        print(l[:-1]);
    print("------")










