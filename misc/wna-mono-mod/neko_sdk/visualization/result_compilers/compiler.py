import os.path

import cv2
import numpy as np

from neko_sdk.visualization.compile_ocr import Kategory_remix
from neko_sdk.visualization.result_compilers.bootstrap import accrfolder
from neko_sdk.visualization.result_compilers.make_tex import maketex;


def compile_core(src,dst,branch,codename,threshs,dsname,protocol,splitkey,counter,filter,case_sensitive=False,dwgood=1280,dwbad=640):
    dic = {}
    dstd = dst;#os.path.join(dst, "details");
    os.makedirs(dstd, exist_ok=True);
    rec = accrfolder(root=src, filter=filter, dst=dstd, thresh=threshs, arcntr=counter(protocol+"-"+splitkey,case_sensitive), case_sensitive=case_sensitive);
    cmds_ = maketex(codename, protocol+"-"+splitkey.replace(" ",""), dsname, branch, rec)
    dic["cmds"] = cmds_;
    return dic;


def compile_alot(root,dss,ds_codenames,modnames,workingdir,devs,branch,methods,codenames,threshs,protocols,counters,filterss,case_sensitive=False,dwgood=1280,dwbad=640):
    rdic = {};
    cmds = [];
    ims=[];
    for mid in range(len(methods)):
        rdic[codenames[mid]]={}
        for p, c ,filters in zip(protocols, counters,filterss):
            rdic[codenames[mid]][p] = {}
            for did in range(len(dss)):
                rdic[codenames[mid]][p][ds_codenames[did]] = {}
                for fk in filters:
                # "318prirC/open_basemodel_protorecIN3_g3ks/jtrmodels/OSTR/rec3_chs_prototyper/JAP_lang/"
                    src = os.path.join(root,devs[mid], methods[mid], p,modnames[mid],dss[did]);
                    dst = os.path.join(workingdir,methods[mid], p,dss[did],fk);
                    mdic=compile_core(src,dst, branch, codenames[mid], threshs, ds_codenames[did], p, fk,
                                 c, filters[fk], case_sensitive, dwgood, dwbad);
                    cmds += mdic["cmds"];
                    rdic[codenames[mid]][p][ds_codenames[did]][fk] = mdic;
                dst=os.path.join(workingdir, methods[mid], p, dss[did])
                imgood = Kategory_remix(root=dst, K=list(filters.keys()), P=[str(i) for i in threshs][:1], dw=dwgood,extra_terms=0);
                imbad = Kategory_remix(root=dst, K=list(filters.keys()), P=[str(i) for i in threshs][1:], dw=dwbad,extra_terms=0)
                im = np.concatenate([imgood, imbad], axis=1);
                rdic[codenames[mid]][p][ds_codenames[did]]["im"] = im;
                cv2.imwrite(os.path.join(dst, "goodnbad.png"), im);
                ims.append(im);

    cv2.imwrite(os.path.join(workingdir,"goodnbadall.png"),np.concatenate(ims));
    a="";
    for i in cmds:
        a+=i+"\n";
    print(a);
    return rdic, cmds
