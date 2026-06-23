import cv2
import numpy as np
def cvtim(ims,mean,var):
    ims = ims * var + mean;
    ims = ims.transpose([0, 2, 3, 1]);
    ims = ims.astype(np.uint8);
    iml = np.split(ims, ims.shape[0], axis=0);
    iml = [i[0] for i in iml];
    return iml;


def show_npims( ims, name,mean=0,var=127,timeout=0):
    iml=cvtim(ims,mean,var);
    if(iml[0].shape[1]>iml[0].shape[0]):
        ims = np.concatenate(iml);
    else:
        ims=np.concatenate(iml,axis=1);
    # for i in range(len(ims)):
    cv2.imshow(name, ims);
    cv2.waitKey(timeout);
def show_tims( tim, name,mean=0, var=127,timeout=0):
    show_npims((tim).detach().cpu().numpy(),name,var,mean)
def show_list_ims( ims, name,mean=0,var=127,timeout=0):
    ims=[im * var + mean for im in ims];
    for i in range(len(ims)):
        cv2.imshow(name+str(i), ims[i]);
    cv2.waitKey(timeout);