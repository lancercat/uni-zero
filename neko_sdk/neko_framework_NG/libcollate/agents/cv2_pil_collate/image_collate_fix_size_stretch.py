import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
import cv2
import numpy as np
from neko_sdk.cfgtool.argsparse import neko_get_arg
from multiprocess.pool import Pool


# collate the image by scaling them down to max_edge.
# not really for the word samples, where we need to specify the minimum size.

# but for character recognition tasks, its good enough.

def collate_fixed_size_strech_core(cv2_image,target_w,target_h,pad_w,pad_h,pad_val):
    # Convert RGB to BGR (which is the default color format used by OpenCV)
    original_h, original_w, original_c = cv2_image.shape
    resized_image = cv2.resize(cv2_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR);
    if(pad_val>=0):
        # Create a black background with the target size
        background = np.zeros((target_h+pad_h+pad_h, target_w+pad_w+pad_w, original_c), dtype=np.uint8)+int(pad_val);
    else:
        background = np.zeros((target_h+pad_h+pad_h, target_w+pad_w+pad_w, original_c), dtype=np.uint8)+np.mean(resized_image,axis=(0,1)).astype(np.uint8);

    # Calculate the position to paste the resized image
    offset_x = pad_w
    offset_y = pad_h
    background[
        offset_y:offset_y +target_h,
        offset_x:offset_x+ target_w] = resized_image;
    background=background/128.0;
    background-=1;
    return np.transpose(background,[2,0,1]);

def collate_fixed_size_strech_pil(params):
    image_pil,target_w,target_h,pad_w,pad_h,pad_val=params;

    pil_image = image_pil.convert('RGB')

    # Convert PIL Image to numpy array
    numpy_image = np.array(pil_image)

    # Convert RGB to BGR (which is the default color format used by OpenCV)
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR);
    return collate_fixed_size_strech_core(cv2_image, target_w, target_h, pad_w, pad_h, pad_val);
def collate_fixed_size_strech_cv2(params):
    cv2_image,target_w,target_h,pad_w,pad_h,pad_val=params;
    return collate_fixed_size_strech_core(cv2_image, target_w, target_h, pad_w, pad_h, pad_val);

def collate_fixed_size_strech_in_parallel_cv2(images,target_w,target_h,pad_w,pad_h,pad_val,pool:Pool):
    ads=[(i,target_w,target_h,pad_w,pad_h,pad_val) for i in images];
    # if (type(images[0]) != type(np.array(0))):
    #     fatal("???");
    return pool.map(collate_fixed_size_strech_cv2,ads);
def collate_fixed_size_strech_in_parallel_pil(images,target_w,target_h,pad_w,pad_h,pad_val,pool:Pool):
    ads=[(i,target_w,target_h,pad_w,pad_h,pad_val) for i in images];

    return pool.map(collate_fixed_size_strech_pil,ads);
def collate_fixed_size_strech_serial_cv2(images,target_w,target_h,pad_w,pad_h,pad_val):
    ads=[(i,target_w,target_h,pad_w,pad_h,pad_val) for i in images];
    # if (type(images[0]) != type(np.array(0))):
    #     fatal("???");
    return [collate_fixed_size_strech_cv2(ad) for ad in ads];
def collate_fixed_size_strech_serial_pil(images,target_w,target_h,pad_w,pad_h,pad_val):
    ads=[(i,target_w,target_h,pad_w,pad_h,pad_val) for i in images];
    return [collate_fixed_size_strech_pil(ad) for ad in ads];


class neko_image_collate_fixed_size_stretch_cv2(ama):
    INPUT_raw_images="raw_images";
    OUTPUT_tensor_images="tensor_images";

    PARAM_template_size_hw="template_size";
    PARAM_padding_size_hw="padding_size";

    PARAM_padding_val="padding_val";
    DEFAULT_PAD_VAL=-1;# use mean padding

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.raw_images = this.register_input(this.INPUT_raw_images, iocvt_dict);
        this.tensor_images = this.register_input(this.OUTPUT_tensor_images, iocvt_dict);

        pass;

    def set_etc(this, params):
        this.template_size_hw = neko_get_arg(this.PARAM_template_size_hw, params);
        this.padding_size_h,this.padding_size_w = neko_get_arg(this.PARAM_padding_size_hw, params,(1,1)); # just pad 1 pixel if not otherwise specified.
        this.padding_val=neko_get_arg(this.PARAM_padding_val,params,this.DEFAULT_PAD_VAL);
        this.core_w=this.template_size_hw[1] - this.padding_size_w-this.padding_size_w;
        this.core_h = this.template_size_hw[0]-this.padding_size_h-this.padding_size_h;
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        raw_images = workspace.get(this.raw_images);
        all_ims=collate_fixed_size_strech_serial_cv2(raw_images, this.core_w, this.core_h, this.padding_size_w, this.padding_size_h,
                                         this.padding_val);
        tim=torch.tensor(np.array(all_ims), dtype=torch.float);
        if(tim.shape[1]==1):
            tim=tim.repeat([1,3,1,1]);
        workspace.add(this.tensor_images,tim);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   raw_images,
                   tensor_images,
                    template_size_hw,padding_size_hw,padding_val
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_raw_images: raw_images, cls.OUTPUT_tensor_images: tensor_images},
                           cls.PARAM_padding_size_hw: padding_size_hw, cls.PARAM_template_size_hw: template_size_hw,
                           "modcvt_dict": {}}}
class neko_image_collate_fixed_size_stretch_pil( neko_image_collate_fixed_size_stretch_cv2):
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        raw_images = workspace.get(this.raw_images);
        all_ims=collate_fixed_size_strech_serial_pil(raw_images, this.core_w, this.core_h, this.padding_size_w, this.padding_size_h,
                                         this.padding_val);
        workspace.add(this.tensor_images,torch.tensor(np.array(all_ims), dtype=torch.float));
        return workspace, environment;
import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
import cv2
import numpy as np
from neko_sdk.cfgtool.argsparse import neko_get_arg
from multiprocess.pool import Pool


# collate the image by scaling them down to max_edge.
# not really for the word samples, where we need to specify the minimum size.

# but for character recognition tasks, its good enough.
# removed mvn from collate--- do it elsewhere with an agent to prevent unnecessary issues
def collate_fixed_size_strech_core(cv2_image,target_w,target_h,pad_w,pad_h,pad_val):
    # Convert RGB to BGR (which is the default color format used by OpenCV)
    original_h, original_w, original_c = cv2_image.shape
    resized_image = cv2.resize(cv2_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR);
    if(pad_val>=0):
        # Create a black background with the target size
        background = np.zeros((target_h+pad_h+pad_h, target_w+pad_w+pad_w, original_c), dtype=np.uint8)+int(pad_val);
    else:
        background = np.zeros((target_h+pad_h+pad_h, target_w+pad_w+pad_w, original_c), dtype=np.uint8)+np.mean(resized_image,axis=(0,1)).astype(np.uint8);

    # Calculate the position to paste the resized image
    offset_x = pad_w
    offset_y = pad_h
    background[
        offset_y:offset_y +target_h,
        offset_x:offset_x+ target_w] = resized_image;
    return np.transpose(background,[2,0,1]);

def collate_fixed_size_strech_pil(params):
    image_pil,target_w,target_h,pad_w,pad_h,pad_val=params;

    pil_image = image_pil.convert('RGB')

    # Convert PIL Image to numpy array
    numpy_image = np.array(pil_image)

    # Convert RGB to BGR (which is the default color format used by OpenCV)
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR);
    return collate_fixed_size_strech_core(cv2_image, target_w, target_h, pad_w, pad_h, pad_val);
def collate_fixed_size_strech_cv2(params):
    cv2_image,target_w,target_h,pad_w,pad_h,pad_val=params;
    return collate_fixed_size_strech_core(cv2_image, target_w, target_h, pad_w, pad_h, pad_val);

def collate_fixed_size_strech_in_parallel_cv2(images,target_w,target_h,pad_w,pad_h,pad_val,pool:Pool):
    ads=[(i,target_w,target_h,pad_w,pad_h,pad_val) for i in images];
    # if (type(images[0]) != type(np.array(0))):
    #     fatal("???");
    return pool.map(collate_fixed_size_strech_cv2,ads);
def collate_fixed_size_strech_in_parallel_pil(images,target_w,target_h,pad_w,pad_h,pad_val,pool:Pool):
    ads=[(i,target_w,target_h,pad_w,pad_h,pad_val) for i in images];

    return pool.map(collate_fixed_size_strech_pil,ads);



class neko_image_collate_fixed_size_stretch_cv2_para(ama):
    INPUT_raw_images="raw_images";
    OUTPUT_tensor_images="tensor_images";

    PARAM_template_size_hw="template_size";
    PARAM_padding_size_hw="padding_size";

    PARAM_padding_val="padding_val";
    DEFAULT_PAD_VAL=-1;# use mean padding

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.raw_images = this.register_input(this.INPUT_raw_images, iocvt_dict);
        this.tensor_images = this.register_input(this.OUTPUT_tensor_images, iocvt_dict);

        pass;

    def set_etc(this, params):
        this.template_size_hw = neko_get_arg(this.PARAM_template_size_hw, params);
        this.padding_size_h,this.padding_size_w = neko_get_arg(this.PARAM_padding_size_hw, params,(1,1)); # just pad 1 pixel if not otherwise specified.
        this.padding_val=neko_get_arg(this.PARAM_padding_val,params,this.DEFAULT_PAD_VAL);
        this.core_w=this.template_size_hw[1] - this.padding_size_w-this.padding_size_w;
        this.core_h = this.template_size_hw[0]-this.padding_size_h-this.padding_size_h;

        this.pool=Pool(neko_get_arg("workers",params,12));

        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        raw_images = workspace.get(this.raw_images);
        all_ims=collate_fixed_size_strech_in_parallel_cv2(raw_images, this.core_w, this.core_h, this.padding_size_w, this.padding_size_h,
                                         this.padding_val, this.pool);
        tim=torch.tensor(np.array(all_ims), dtype=torch.float);
        if(tim.shape[1]==1):
            tim=tim.repeat([1,3,1,1]);
        workspace.add(this.tensor_images,tim);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   raw_images,
                   tensor_images,
                    template_size_hw,padding_size_hw,padding_val
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_raw_images: raw_images, cls.OUTPUT_tensor_images: tensor_images},
                           cls.PARAM_padding_size_hw: padding_size_hw, cls.PARAM_template_size_hw: template_size_hw,
                           "modcvt_dict": {}}}
class neko_image_collate_fixed_size_stretch_pil_para( neko_image_collate_fixed_size_stretch_cv2):
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        raw_images = workspace.get(this.raw_images);
        all_ims=collate_fixed_size_strech_in_parallel_pil(raw_images, this.core_w, this.core_h, this.padding_size_w, this.padding_size_h,
                                         this.padding_val, this.pool);
        workspace.add(this.tensor_images,torch.tensor(np.array(all_ims), dtype=torch.float));
        return workspace, environment;
