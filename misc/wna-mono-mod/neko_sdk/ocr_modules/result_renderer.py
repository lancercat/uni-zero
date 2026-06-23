import numpy as np
import numpy as np
import pylcs;
import regex
import torch
import cv2
from neko_sdk.ocr_modules.element_renderer import color,render_pred,render_gt

def render_card(lim,patchs,length):
    res = np.zeros((32, length * 32 + 48, 3));
    res[0:32, 0:48,:] = lim.reshape((32,48,-1));
    for i in range(len(patchs)):
        res[0:32, 48 + i * 32:80 + i * 32] = patchs[i];
    return res;

def render_chars(lim,dict,chs,colors,length):
    patchs=[];
    for i in range(len(chs)):
        patchs.append(color(dict,chs[i],*(colors[i])));
    return render_card(lim,patchs,length);

def dye(ids,srcdst,color):
    for i in ids:
        srcdst[i]=color;
    return srcdst;
def get_unpadded(im,length,kar=False,h=64,padvalue=0,transposed=False):
    if(transposed):
        im = im.transpose([1, 0, 2]);
    if(padvalue<0):
        try:
            padl = np.min(np.nonzero(np.hstack((im-padvalue).sum(0).sum(1))));
            padr = np.max(np.nonzero(np.hstack((im-padvalue).sum(0).sum(1))));

            if(padl==0 and padr==im.shape[1]-1):
                imc=im;
            else:
                imc = im[:, padl:padr, :]
            if(kar):
                h=int(imc.shape[0]*(length/imc.shape[1]))
            im1 = cv2.resize(imc, (length, h))
        except:
            im1 = cv2.resize(im, (length , h))
    else:
        return cv2.resize(im, (length , h));

    return im1;

def render_diff(dict,top,bottom,gt,im,sc=(255,255,255),dc=(0,0,255),padvalue=0):


    same_id_top=pylcs.lcs_str2id(bottom,top);
    same_id_bottom=pylcs.lcs_str2id(top,bottom);
    length=max(len(bottom),len(top),len(gt));

    tc=[dc for i in range(len(top))];
    bc = [dc for i in range(len(bottom))];
    gc=[sc for i in range(len(gt))];
    dye(same_id_top,tc,sc);
    dye(same_id_bottom, bc, sc);
    ti=render_chars(np.zeros([32,48],np.uint8),dict,top,tc,length);
    bi = render_chars(np.zeros([32, 48], np.uint8), dict, bottom, bc, length);
    gi = render_chars(np.zeros([32, 48], np.uint8), dict, gt, gc, length);
    im=get_unpadded(im,length*32,padvalue=padvalue);
    dst=np.zeros([64+32*3,length*32+48,3]);
    dst[:64,48:]=im;
    dst[64:]=np.concatenate([gi,ti,bi],axis=0);
    return dst;



def render_words(dict,seenchs,im,gt_word,pred_words,flag=0,padvalue=0,transposed=False,flair=None):
    if(flair is None):
        try:
            Nep
            PRED = 255 - cv2.resize(cv2.imread("pr.png"), (48, 32));
        except:
            PRED = None;
    else:
        PRED=cv2.resize(flair,(48,32)); # abuse this section for some routing infor etc.
    try:
        Nep
        GT=255-cv2.resize(cv2.imread("gt.png"), (48, 32));
    except:
        GT=np.zeros([32,48,3],np.uint8);
        GT[:,:,:]=255;
    patchs = [
    ]
    try:
        if(gt_word is not None):
            patchs.append([GT]+render_gt(dict,seenchs,gt_word))
        for word in pred_words:
            if (PRED is None):
                PRED = np.zeros([32, 48, 3], np.uint8);
                if (gt_word == word):
                    PRED[:, :, 1] = 255;
                else:
                    PRED[:, :, 2] = 255;
            patchs.append([PRED]+render_pred(dict,gt_word,word));
        # remove flag block
        ml=max([len(i)-1 for i in patchs]);
        taw=ml*32+48

        res=np.zeros((32*len(patchs),taw,3));
        for r in range(len(patchs)):
            off=0;
            for c in range(len(patchs[r])-flag):
                p=patchs[r][c];
                pw=p.shape[1];
                if(transposed and p.shape[0]==p.shape[1]):
                    p=p.transpose([1,0,2]);
                res[r*32:r*32+32,off:off+pw]=p;
                off+=pw;
        if(gt_word is not None):
            lgtp=len(regex.findall(r'\X', gt_word, regex.U))-flag;
        else:
            lgtp=len(regex.findall(r'\X', pred_words[0], regex.U))-flag;
        # because the padding is guessed on the x axis, if its not, we transpose it.

        im1=get_unpadded(im,lgtp*32,padvalue=padvalue,transposed=transposed);
        im11=np.zeros((64,ml*32+48,3));
        im11[:,48:lgtp*32+48,:]=im1;

        fin=np.concatenate([im11.astype(np.uint8),res.astype(np.uint8)],0);
        if (transposed):
            fin = fin.transpose([1, 0, 2]);
    except:
        print("error during visualization")
        fin=np.zeros([32,32,3]);
    if(len(pred_words)):
        if(gt_word is not None):
            return fin,\
                   [1-(pylcs.edit_distance(pred_word,gt_word))/max(1,len(gt_word))
                    for pred_word in pred_words];
        else:
            return fin,[0 for pred_word in pred_words];
    else:
        return fin,[0]
def render_word(dict,seenchs,im,gt_word,pred_word,flag=0,padvalue=0,transposed=False):
    if(gt_word is not None):
        fin,neds=render_words(dict,seenchs,im,gt_word,[pred_word],flag,padvalue,transposed);
    else:
        fin,neds=render_words(dict,seenchs,im,gt_word,[pred_word],flag,padvalue,transposed);
    return fin,neds[0];

if __name__ == '__main__':
    from neko_sdk.ocr_modules.charset.chs_cset import t1_3755;
    from neko_sdk.ocr_modules.charset.etc_cset import latin62;
    cs=t1_3755.union(latin62);
    for i in range(2000):
        img=cv2.imread("/home/lasercat/ssddata/pamidump/kr/"+str(i)+"before_img.jpg");
        with open("/home/lasercat/ssddata/pamidump/kr/"+str(i)+"_res.txt","r") as ifp:
            [gt,pr,_]=[i.strip() for i in ifp];
        dict=torch.load("/home/lasercat/ssddata/dicts/dabkrmlt.pt")
        red,ned=render_word(dict,cs,img,gt.lower(),pr.lower());
        cv2.imshow("red",red);
        cv2.waitKey(0);