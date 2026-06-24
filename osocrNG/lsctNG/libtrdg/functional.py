import cv2
import numpy

from neko_sdk.thirdparty.trdg import  background_generator
import random
import numpy as np


def gen_bg(background_type, background_height, background_width, background_img):
    if background_type == 0:
        background_img = background_generator.gaussian_noise(
            background_height, background_width
        )
    elif background_type == 1:
        background_img = background_generator.plain_white(
            background_height, background_width
        )
    elif background_type == 2:
        background_img = background_generator.quasicrystal(
            background_height, background_width
        )
    else:
        try:
            background_img = background_generator.image_pic(
                background_height, background_width, background_img
            ).convert("RGB");
        except:
            # print(image_dir);
            background_img = background_generator.quasicrystal(
                background_height, background_width
            )

    return background_img;
def gen_bg_numpy(background_type, background_height, background_width, background_img):
    if background_type < 3:
        return numpy.array(
            gen_bg(background_type,background_height, background_width, None)
        )
    else:
        if(background_img.shape[0]==background_height and background_img.shape[1]==background_width):
            return background_img
        else:
            return cv2.resize(background_img,( background_width, background_height));


def blend(m,b,c):
    tm = m / 255;
    cm = tm * c;
    mm = np.array(b)[:, :, ::-1] * (1 - tm) + cm;
    return mm.astype(np.uint8);
def blend_np(m, b, c):
    """Optimized version of blend_np."""
    tm = m.astype(np.float32) / 255.0  # Convert to float32 once
    cm = tm * c
    mm = b * (1 - tm) + cm
    return mm.astype(np.uint8)
# def blend_np(m,b,c):
#     tm = m / 255;
#     cm = tm * c;
#     mm = b * (1 - tm) + cm;
#     return mm.astype(np.uint8);

def gen_bg_stub(param):
    background_type, background_height, background_width, background_img=param;
    return np.array(gen_bg(background_type, background_height, background_width, background_img))[:,:,::-1];


def gen_bg_in_parallel(params,pool):
    return pool.map(gen_bg_stub,params);


def mix_with_gen_bg(mask,color,background_type, background_img):
    bg=gen_bg_numpy(background_type,mask.shape[0], mask.shape[1],background_img);
    return blend_np(mask,bg,color);

def mix_with_gen_bg_stub(params):
    mask, color, background_type, background_img=params;
    return mix_with_gen_bg(mask, color, background_type, background_img);
def mix_with_gen_bg_serial(masks, colors, background_types, background_imgs):
    ars=[];
    for i in range(len(masks)):
        if(masks[i] is not None):
            ars.append(mix_with_gen_bg(masks[i],colors[i],background_types[i],background_imgs[i]));
        else:
            ars.append(None);
    return ars;

def mix_with_gen_bg_in_parallel(masks, colors, background_types, background_imgs,pool):
    return pool.map(mix_with_gen_bg_stub, zip(masks, colors, background_types, background_imgs))

if __name__ == '__main__':
    import torch
    bug=torch.load("/home/lasercat/tmp/bug.pt")
    for i in bug:
        mix_with_gen_bg_stub(i);