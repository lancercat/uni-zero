import cv2
import numpy as np

from neko_sdk.ocr_modules.augmentation.qhbaug import WarpMLS


def jitter(img,s):
    """
    jitter
    """
    w, h, _ = img.shape
    if s>0:
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img

def cvtColor(img,delta):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img


def add_gasuss_noise(image,noise):
    """
    Gasuss noise
    """
    out = image + 0.5 * noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out

def get_crop(image,ratio,top_crop):
    """
    random crop
    """
    h, w, _ = image.shape
    crop_img = image.copy()
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img
def apply_tia(src,src_pts,dst_pts,img_w,img_h):
    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()
    return dst;
def reverse(img):
    return 255-img;

class neko_qhbwrapR_executer:
    def __init__(this):
        this.actions={
            "tia": apply_tia,
            "jitter":jitter,
            "cvtcolor": cvtColor,
            "blur":blur,
            "gaussian_noise":add_gasuss_noise,
            "crop":get_crop,
            "reverse":reverse,
        }
    def execute_cfg(this,raw,cfg):
        img=raw.copy();
        img=this.actions[cfg["act"]](img,*cfg["para"]);
        return img;
    def execute_cfgs(this,raw,cfgs):
        img=raw.copy();
        for c in cfgs:
            img=this.actions[c["act"]](img,*c["para"]);
        return img;