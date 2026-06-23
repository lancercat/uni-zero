
import numpy as np
import cv2
import pylcs
import regex
import torch


def render_imgschscore(dict,im_file,t,dstfile):
    im = cv2.imread(im_file);
    im = cv2.resize(im, (128, 64))
    with open(t, "r") as gtf:
        gt, ch = [l.strip() for l in gtf];
        if (ch != ""):
            predp = dict["protos"][dict["label_dict"][ch]][0][0].numpy()
            predp = cv2.resize(predp, (32, 32)).astype(np.uint8)
        else:
            predp = np.zeros([32, 32]).astype(np.uint8);

        gtp = dict["protos"][dict["label_dict"][gt]][0][0].numpy()
        gtp = cv2.resize(gtp, (32, 32)).astype(np.uint8);
        im[:32, -32:, :] = gtp.reshape(32, 32, 1);
        im[32:, -32:, :] = 0
        if (gt == ch):
            im[32:, -32:, 1] = predp;
        else:
            im[32:, -32:, 2] = predp;
        cv2.imwrite(dstfile, im);
        pass;

def render_imgschs(dict,im_file,txtfiles):
    for t in txtfiles:
        dim=t.replace("txt","jpg");
        render_imgschscore(dict,im_file,t,dim)

def color(dict,ch,b,g,r):
    if(ch not in dict["label_dict"]):
        default = np.zeros([32, 32, 3], np.uint8) + 255;
        if(ch.upper()in dict["label_dict"]):
            return color(dict,ch.upper(),b,g,r);
        elif ch=="⑨" or ch=='[UNK]':
            default[:,:,0]=249;
            default[:, :, 1] = 139;
            default[:, :, 2] = 111;
        else:
            default[:,:,0]=249;
            default[:, :, 1] = 139;
            default[:, :, 2] = 111
            # print(ch);
        return default;
    if ch == "[s]" or ch == '[UNK]':
        default = np.zeros([32, 32, 3], np.uint8) + 255;
        default[:, :, 0] = 249;
        default[:, :, 1] = 139;
        default[:, :, 2] = 111;
        return default;
    chid=dict["label_dict"][ch];
    proto=dict["protos"][chid][0][0].numpy();
    ret=np.zeros([proto.shape[0],proto.shape[1],3]);
    b/=255.
    g/=255.
    r/=255.
    ret[:,:,0]=proto*b;
    ret[:,:,1]=proto*g;
    ret[:,:,2]=proto*r;
    return ret.astype(np.uint8);
def render_pred(dict,gt_word,pred_word):
    pred_patchs=[];
    flag=True;
    if(gt_word is not None):
        corids=pylcs.lcs_str2id(gt_word.lower(), pred_word.lower())
    else:
        flag=False;
        corids=set(range(len(pred_word)))

    chs=list(regex.findall(r'\X', pred_word, regex.U))
    if (len(pred_word) != len( chs)):
        flag = False;
        corids = set(range(len(pred_word)))
    for i in range(len(chs)):
        if (i in corids):
            if(flag):
                proto = color(dict, chs[i], 0, 255, 0)
            else:
                proto = color(dict, chs[i], 255, 255, 255)
        else:
            proto = color(dict, chs[i], 0, 0, 255);
        pred_patchs.append(proto);
    return pred_patchs;
def render_gt(dict,seenchs,gt_word):

    gtpatchs = [];
    chs=list(regex.findall(r'\X', gt_word, regex.U))
    for i in range(len(chs)):
        if (chs[i] in seenchs):
            proto = color(dict, chs[i], 255, 255, 255)
        else:
            proto = color(dict, chs[i], 0, 255, 255);
        gtpatchs.append(proto);
    return gtpatchs;

def alignment(patch_lists):
    mlen = max([len(l) for l in patch_lists]);
    for i in range(len(patch_lists)):
        for j in range(len(patch_lists[i]),mlen):
            patch_lists[i].append(np.zeros([32,32,3]));
    return patch_lists;

def resize(im,l,catdim,ldim,forcedcatdim,minval=16):
    if(forcedcatdim<0):
        ratio = l / im.shape[ldim];
        otherside=max(int(ratio*im.shape[catdim]),minval)
    else:
        otherside=forcedcatdim;
    if(catdim==0):
        dshape=[l,otherside];
    elif(catdim==1):
        dshape=[otherside,l];
    return cv2.resize(im,dshape);

def cat_all(raw_im,patch_lists,aux_img_list,catdim,pcatdim,forced_otherside,minval):
    apl=[];
    img_list=[raw_im]+aux_img_list;
    for pl in patch_lists:
        apl.append(np.concatenate(pl,pcatdim));
    l=apl[0].shape[pcatdim];
    niml=[resize(im,l,catdim,pcatdim,forced_otherside,minval) for im in img_list]
    fiml=[niml[0]]+apl+niml[1:];
    return np.concatenate(fiml,axis=catdim);


def flair_it_core(rim,flairs,extenddim=0,compdim=1,flair_sz=32):
    rf=np.concatenate([resize(f,flair_sz,compdim,extenddim,flair_sz,16) for f in flairs],axis=compdim);
    rf=resize(rf,rim.shape[compdim],extenddim,compdim,flair_sz,16);
    return np.concatenate([rf,rim],axis=extenddim);

def remix(raw_im,aux_img_list,patch_lists,forced_otherside=-1,mival=16):
    patch_lists=alignment(patch_lists);
    if(raw_im.shape[0]>raw_im.shape[1]):
        return cat_all(raw_im,patch_lists,aux_img_list,1,0,forced_otherside=forced_otherside,minval=mival);
    else:
        return cat_all(raw_im, patch_lists, aux_img_list, 0, 1, forced_otherside=forced_otherside, minval=mival);


def flair_it(raw_im,rim,flairs,flair_sz=32):
    if (raw_im.shape[0] > raw_im.shape[1]):
        return flair_it_core(rim,flairs,1,0,flair_sz);
    else:
        return flair_it_core(rim,flairs,0,1,flair_sz);