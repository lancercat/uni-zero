import glob
import os
import random

import cv2
import numpy as np


def getres(file):
    with open(file,"r") as fp:
        [gt,pr]=[i.strip() for i in fp ][:2];
        return gt, pr;



#
def remix(root,pkeys,dwidth=1024,mar=128,flag=44,rows=1,extra_terms=1):
    idict={}
    for k in pkeys:
        images=[];
        images_=glob.glob(os.path.join(root,k,"*.jpg"));
        for i in images_:
            if(i.find("att")!=-1):
                continue;
            else:
                images.append(i);
        random.shuffle(images);
        idict[k]=images;
    a=[]
    for r in range(rows):
        cw = 0;
        imgs = [];
        i=0;
        while cw<dwidth and i<100000:
            rk=random.sample(pkeys,1)[0];
            if(len(idict[rk])==0):
                i += 1
                continue;
            # a third term may appear beyond visual context
            im=cv2.imread(idict[rk][0])[:128+32*extra_terms,flag:];
            idict[rk]=idict[rk][1:];
            if(im.shape[1]+cw<dwidth+mar):
                imgs.append(im);
                cw+=im.shape[1];
            i+=1

        im=cv2.resize(np.concatenate(imgs,1),(dwidth,128));
        a.append(im);
    return np.concatenate(a,axis=0);

def Kategory_remix(root,K,P,dw=1024,extra_terms=1,rows=1):
    rs=[]
    for k in K:
        gr=remix(os.path.join(root,k),P,dwidth=dw,extra_terms=extra_terms,rows=rows);
        # br=remix(os.path.join(root,k),B,dwidth=dw);
        rs.append(gr);
        # gr=remix(os.path.join(root,k),G,dwidth=dw,extra_terms=extra_terms);
        # rs.append(gr);
        # rs.append(br);
    return np.concatenate(rs);



def test(file):
    with open(file,"r") as fp:
        gt,pred,_=[i.strip() for i in fp];
    return gt==pred;

def get_acr_stat(root,d,S,id):
    "500/closeset_benchmarks/HWDB_unseen/"
    txts=[os.path.join(root,s,"closeset_benchmarks",d,str(id)+".txt") for s in S];
    imgs = [os.path.join(root, s, "closeset_benchmarks", d, str(id) + ".jpg") for s in S];
    stats=[int(test(s)) for s in txts];
    for i in range(1, len(imgs)):
        if(stats[i]<stats[i-1]):
            return None;
    ims=[cv2.imread(img)[:,48:] for img in imgs];
    for i in range(1,len(ims)):
        ims[i]=ims[i][96:]
    return np.concatenate(ims,axis=0)

def get_acr_stat2(root,d,t,S,id):
    "1000/closeset_benchmarks/base_ctw_prototyper/"
    "500/closeset_benchmarks/HWDB_unseen/"
    txts=[os.path.join(root,"jtrmodels"+s,"closeset_benchmarks",t,d,str(id)+".txt") for s in S];
    imgs = [os.path.join(root, "jtrmodels"+s, "closeset_benchmarks",t, d, str(id) + ".jpg") for s in S];
    stats=[int(test(s)) for s in txts];
    for i in range(1, len(imgs)):
        if(stats[i]<stats[i-1]):
            return None;
    ims=[cv2.imread(img)[:128,48:] for img in imgs];
    for i in range(1,len(ims)):
        ims[i]=ims[i][96:]
    return np.concatenate(ims,axis=0)
