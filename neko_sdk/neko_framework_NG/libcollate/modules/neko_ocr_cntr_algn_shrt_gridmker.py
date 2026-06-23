import os.path

from torch import nn
from neko_sdk.cfgtool.argsparse import neko_get_arg
import torch

import torch.nn.functional as trnf

# center align shot samples, but strecth long to fit
class neko_ocr_cntr_algn_shrt(nn.Module):
    PARAM_template_w="template_w";
    PARAM_template_h="template_h";

    def mkgrid(this, w, h):
        xs = torch.linspace(-1, 1, w, dtype=torch.float32)
        ys = torch.linspace(-1, 1, h, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0)
        return grid


    def __init__(this,params):
        super().__init__();
        this.template_w=neko_get_arg(this.PARAM_template_w,params);
        this.template_h=neko_get_arg(this.PARAM_template_h,params);
        this.template_ratio=this.template_w/this.template_h;
        this.grid=nn.Parameter(
            this.mkgrid(this.template_w,this.template_h),
            requires_grad=False
        );
    def forward(this,raw_src,thumb_map,thumb_ten):
        # just return the raw grids
        nims=len(raw_src);

        ags=this.grid.repeat(nims,1,1,1);
        sfs=[];

        for i in range(nims):
            w,h=raw_src[i].shape[-1],raw_src[i].shape[-2];
            srat=w/h;
            if (srat<this.template_ratio):
                sfs.append(this.template_ratio/srat);
            else:
                sfs.append(1);
        ags[:,:,:,0]*=(torch.tensor(sfs, device=ags.device,dtype=ags.dtype).view(nims,1,1));
        return ags.permute(0,3,1,2);
if __name__ == '__main__':
    m=neko_ocr_cntr_algn_shrt({neko_ocr_cntr_algn_shrt.PARAM_template_w:32,neko_ocr_cntr_algn_shrt.PARAM_template_h:32});
    # m.cuda();
    m([torch.rand(1,3,39,71) for _ in range(5)],None,None);
    pass;

