from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
import torch.nn.functional as trnf
import torch
class neko_image_list_grid_sampler_collate(ama):
    INPUT_tensor_image_list="raw_images";
    INPUT_grids="grids";
    OUTPUT_tensor_images="tensor_images";
    PARAM_mode="mode"
    pass;

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.tensor_image_list = this.register_input(this.INPUT_tensor_image_list, iocvt_dict);
        this.grids = this.register_input(this.INPUT_grids, iocvt_dict);
        this.tensor_images = this.register_output(this.OUTPUT_tensor_images, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.mode=neko_get_arg(this.PARAM_mode,params);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        tensor_image_list = workspace.get(this.tensor_image_list);
        grids=workspace.get(this.grids).permute(0,2,3,1); # grids should be interally nchw--- bcs we may use non affine grids
        B, GH, GW, _ = grids.shape;
        device=grids.device;
        C = tensor_image_list[0].shape[1] ; # Channels
        output = torch.empty((B, C, GH, GW), device=device)
        for i in range(B):
            # Move to device asynchronously (works best if CPU tensors are pinned)
            img = tensor_image_list[i].to(device, non_blocking=True)

            # grid_sample is a non-blocking kernel launch.
            # By assigning to a slice of a pre-allocated tensor, we avoid the sync of cat()
            output[i:i + 1] = trnf.grid_sample(
                img,
                grids[i:i + 1],
                padding_mode='zeros',
                align_corners=False,
                mode=this.mode
            )
        # tarlst=[trnf.grid_sample(tensor_image_list[i].to(grids.device, non_blocking=True),grids[i].unsqueeze(0)) for i in range(len(tensor_image_list))];
        # output=torch.cat(tarlst,0);
        workspace.add(this.tensor_images,output);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   tensor_image_list,
                   tensor_grids,
                   tensor_images,
                   mode="bilinear"
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_tensor_image_list: tensor_image_list,cls.INPUT_grids:tensor_grids, cls.OUTPUT_tensor_images: tensor_images},
            "modcvt_dict": {},
            cls.PARAM_mode: mode},
            };


class neko_tensorblk_list_grid_sampler_collate(neko_image_list_grid_sampler_collate):

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        tensor_image_list = workspace.get(this.tensor_image_list);
        grids=workspace.get(this.grids).permute(0,2,3,1); # grids should be interally nchw--- bcs we may use non affine grids
        B, GH, GW, _ = grids.shape;
        device=grids.device;
        out=[];
        for i in range(B):
            # Move to device asynchronously (works best if CPU tensors are pinned)
            img = tensor_image_list[i].to(device, non_blocking=True)
            # grid_sample is a non-blocking kernel launch.
            # By assigning to a slice of a pre-allocated tensor, we avoid the sync of cat()
            out.append(trnf.grid_sample(
                img,
                grids[i:i + 1],
                padding_mode='zeros',
                align_corners=False,
                mode=this.mode
            )); # we have  a block of tensors for semantic segmentations, and more if instance seg
        # tarlst=[trnf.grid_sample(tensor_image_list[i].to(grids.device, non_blocking=True),grids[i].unsqueeze(0)) for i in range(len(tensor_image_list))];
        # output=torch.cat(tarlst,0);
        workspace.add(this.tensor_images,out);
        return workspace, environment;



