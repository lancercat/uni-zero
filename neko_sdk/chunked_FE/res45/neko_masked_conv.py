from torch import Tensor
from torch import nn
from torch.nn import functional as trnf


class neko_masked_conv(nn.Conv2d):
    
    def adjustmask(this,input,mask):
        return trnf.interpolate(mask,input.shape[-2:],mode="bilinear");
    def forward(this, input: Tensor,mask: Tensor):
        data=super().forward(input);
        if(mask is None):
            return data, None
        if(data.shape[-1]!=mask.shape[-1]):
            mask=this.adjustmask(data,mask);
        return data*mask, mask;

class neko_masked_drop(nn.Dropout):
    def forward(self, input: Tensor,mask: Tensor):
        data=super().forward(input); # you drop zero, and you get zero, noneed re-applying
        return data, mask;
