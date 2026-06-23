# coding:utf-8
import cv2
import numpy as np

from neko_sdk.ocr_modules.augmentation.qhbaug import qhbwarp


def clahe( bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_RGB2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15, 15))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return bgr;


def keepratio_resize(img, h, w,rgb=False):
    cur_ratio = img.shape[1] / float(img.shape[0])
    target_ratio = w / float(h)
    mask_height = h
    mask_width = w
    img = np.array(img)
    if rgb==False and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif rgb==True and len(img.shape)==1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if cur_ratio > target_ratio:
        cur_target_height = h
        # print("if", cur_ratio, self.target_ratio)
        cur_target_width = w
    else:
        cur_target_height = h
        # print("else",cur_ratio,self.target_ratio)
        cur_target_width = int(h * cur_ratio)
    img = cv2.resize(img, (cur_target_width, cur_target_height))
    start_x = int((mask_height - img.shape[0]) / 2)
    start_y = int((mask_width - img.shape[1]) / 2)
    if(rgb):
        mask = np.zeros([mask_height, mask_width,3]).astype(np.uint8);
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
    else:
        mask = np.zeros([mask_height, mask_width,1]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1],0] = img

    img = mask
    bmask=None;
    return img,bmask

def neko_DAN_padding(img,ismask,img_height,img_width,target_ratio,qhb_aug,gray):
    cur_ratio = img.size[0] / float(img.size[1])

    mask_height = img_height
    mask_width = img_width
    img = np.array(img)
    if(gray):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif (len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR);
    if (qhb_aug):
        try:
            img = qhbwarp(img, 10);
        except:
            pass;

    if cur_ratio > target_ratio:
        cur_target_height = img_height
        # print("if", cur_ratio, self.target_ratio)
        cur_target_width = img_width
    else:
        cur_target_height = img_height
        # print("else",cur_ratio,self.target_ratio)
        cur_target_width = int(img_height * cur_ratio)
    img = cv2.resize(img, (cur_target_width, cur_target_height))
    start_x = int((mask_height - img.shape[0]) / 2)
    start_y = int((mask_width - img.shape[1]) / 2)
    if(gray):
        mask = np.zeros([mask_height, mask_width]).astype(np.uint8)
    else:
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
    mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
    bmask = np.zeros([mask_height, mask_width]).astype(np.float32)
    bmask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = 1
    img = mask
    return img,bmask


def neko_aligned_left_padding(img,ismask,img_height,img_width,target_ratio,qhb_aug,gray):
    img = np.array(img)

    cur_ratio = img.shape[1] / float(img.shape[0])

    mask_height = img_height
    mask_width = img_width
    if(gray):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif (len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR);
    if (qhb_aug):
        try:
            img = qhbwarp(img, 10);
        except:
            pass;

    if cur_ratio > target_ratio:
        cur_target_height = img_height
        # print("if", cur_ratio, self.target_ratio)
        cur_target_width = img_width
    else:
        cur_target_height = img_height
        # print("else",cur_ratio,self.target_ratio)
        cur_target_width = int(img_height * cur_ratio)
    img = cv2.resize(img, (cur_target_width, cur_target_height))
    start_x =0
    start_y = 0
    if(gray):
        mask = np.zeros([mask_height, mask_width,1]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1],0] = img
    else:
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
        mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
    bmask = np.zeros([mask_height, mask_width]).astype(np.float32)
    bmask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = 1
    img = mask
    return img,bmask

def neko_aligned_left_padding_beacon(img,ismask,img_height,img_width,beacon_height,beacon_width,target_ratio,qhb_aug,gray):
    cur_ratio = img.size[0] / float(img.size[1])

    mask_height = img_height
    mask_width = img_width
    img = np.array(img)
    if(gray):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif (len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR);
    if (qhb_aug):
        try:
            img = qhbwarp(img, 10);
        except:
            pass;

    if cur_ratio > target_ratio:
        cur_target_height = img_height
        # print("if", cur_ratio, self.target_ratio)
        cur_target_width = img_width
    else:
        cur_target_height = img_height
        # print("else",cur_ratio,self.target_ratio)
        cur_target_width = int(img_height * cur_ratio)
    img = cv2.resize(img, (cur_target_width, cur_target_height))
    beacon=cv2.resize(img,(beacon_width,beacon_height));
    start_x =0
    start_y = 0
    if(gray):
        mask = np.zeros([mask_height, mask_width,1]).astype(np.uint8)
    else:
        mask = np.zeros([mask_height, mask_width, 3]).astype(np.uint8)
    mask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = img
    bmask = np.zeros([mask_height, mask_width]).astype(np.float32)
    bmask[start_x: start_x + img.shape[0], start_y: start_y + img.shape[1]] = 1
    img = mask
    return img,bmask,beacon



def fill(img,img_height,img_width):
    tarimg = cv2.resize(img, (img_width, img_height));
    bmask = np.zeros([img_height, img_width,1]).astype(np.float32)+1;
    return tarimg,bmask,(img_width, img_height);

def pad_core(img,cur_target_height,cur_target_width,img_height,img_width,base):
    img = cv2.resize(img, (cur_target_width, cur_target_height))
    tarimg = (np.zeros([img_height, img_width, img.shape[-1]]) + base).astype(np.uint8)
    tarimg[: img.shape[0], :img.shape[1]] = img
    bmask = np.zeros([img_height, img_width,1]).astype(np.float32);
    bmask[: img.shape[0], :img.shape[1]] = 1;
    return tarimg, bmask,(cur_target_width, cur_target_height);

def pad_al_h_core(img,cur_ratio,img_height,img_width,base):
    cur_target_height = img_height
    # print("else",cur_ratio,self.target_ratio)
    cur_target_width = int(img_height * cur_ratio)
    if (cur_target_width > img_width):
        cur_target_width = img_width;
    return pad_core(img,cur_target_height,cur_target_width,img_height,img_width,base);


def pad_al_v_core(img,cur_ratio,img_height,img_width,base):
    cur_target_width = img_width;
    # print("else",cur_ratio,self.target_ratio)
    cur_target_height = int(img_width / cur_ratio);
    if(cur_target_height>img_height):
        cur_target_height=img_height;
    return pad_core(img,cur_target_height,cur_target_width,img_height,img_width,base);
def pad_al_h(img,img_height,img_width):
    cur_ratio = img.size[0] / float(img.size[1]);
    return pad_al_h_core(img,cur_ratio,img_height,img_width,0);


def pad_al_v(img,img_height,img_width):
    cur_ratio = img.size[0] / float(img.size[1]);
    return pad_al_v_core(img,cur_ratio,img_height,img_width,0);


def pad_al_h_np(img,img_height,img_width,base=127.5):
    cur_ratio = img.shape[1] / float(img.shape[0]);
    return pad_al_h_core(img,cur_ratio,img_height,img_width,base);


def pad_al_v_np(img,img_height,img_width,base=127.5):
    cur_ratio = img.shape[1] / float(img.shape[0]);
    return pad_al_v_core(img,cur_ratio,img_height,img_width,base);


def neko_aligned_left_top_padding_beacon_np(img,ismask,img_height,img_width,beacon_height,beacon_width):
    target_ratio = img_width / img_height;
    cur_ratio = img.shape[1] / float(img.shape[0]);
    beacon=cv2.resize(img,(beacon_width,beacon_height));
    if (len(img.shape) == 2):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR);
    if(img_width>img_height):
        if cur_ratio < target_ratio:
            img,bmask,sz=pad_al_h_np(img,img_height,img_width);
        else:
            img,bmask,sz=fill(img,img_height,img_width);
    else:
        if cur_ratio > target_ratio:
            img, bmask,sz = pad_al_v_np(img, img_height, img_width);
        else:
            img, bmask,sz = fill(img, img_height, img_width);
    return img,bmask,beacon,sz

