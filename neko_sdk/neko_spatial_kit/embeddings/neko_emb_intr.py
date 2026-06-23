
import torch;
from torch import nn;
from torch.nn import functional as trnf
import math
from neko_sdk.cfgtool.argsparse import neko_get_arg


class neko_add_embint_se(nn.Module):
    def __init__(this,w=16,h=16,c=32):
        super(neko_add_embint_se, this).__init__()
        this.param=torch.nn.Parameter(torch.rand(1,c,h,w)*2-1)
    def forward(this,x):
        N,C,H,W=x.shape;
        return torch.cat([x,trnf.interpolate(this.param,[H,W],mode="bilinear").repeat(N,1,1,1)],dim=1);


class neko_add_embint_se_NG(nn.Module):
    PARAM_emb_w="w";
    PARAM_emb_h="h";
    PARAM_emb_ch="ch";
    def init_weight(this):
        return torch.rand(1,this.c,this.h,this.w)*2-1;
    def __init__(this,params):
        super(neko_add_embint_se_NG, this).__init__()
        this.w=neko_get_arg(this.PARAM_emb_w,params,16);
        this.h = neko_get_arg(this.PARAM_emb_h, params, 16);
        this.c = neko_get_arg(this.PARAM_emb_ch, params, 32);
        this.param=torch.nn.Parameter(this.init_weight());
    def forward(this,x):
        N,C,H,W=x.shape;
        return torch.cat([x,trnf.interpolate(this.param,[H,W],mode="bilinear").repeat(N,1,1,1)],dim=1);

class neko_add_embint_se_HD_nocond(nn.Module):
    def __init__(this,w=128,h=128,c=32):
        super(neko_add_embint_se_HD_nocond, this).__init__()
        this.param=torch.nn.Parameter(torch.rand(1,c,h,w)*2-1)
    def forward(this):
        return this.param;


class neko_add_embint_se_cosine_init_NG(neko_add_embint_se_NG):

    def init_weight(this):

        # --- Cosine Initialization Modification ---

        # 1. Create coordinate grids for H and W
        # Grids scaled from -1 to 1
        x_coords = torch.linspace(-1, 1, this.w).view(1, 1, 1, this.w).repeat(1, 1, this.h, 1)
        y_coords = torch.linspace(-1, 1, this.h).view(1, 1, this.h, 1).repeat(1, 1, 1, this.w)

        # 2. Determine the frequencies (omega) for each channel

        # DECISION: Set maximum frequency based on max(W, H)
        # This ensures the highest frequency channel has sufficient periods
        # across the largest dimension. A scaling factor (e.g., 2*pi) is common.
        base_factor = 2 * math.pi  # Standard factor for full cycle (-1 to 1)

        # Max frequency is proportional to the size of the largest dimension
        # A simple linear scaling: Max_Freq = base_factor * max(W, H)
        # We also scale by channel to ensure a sensible progression across C

        max_dim = max(this.w, this.h)
        # A conservative choice is max_dim. A more aggressive one is max_dim * C
        # Let's use max_dim * base_factor to relate directly to pixels/coordinates
        max_freq = max_dim * base_factor

        # Frequencies range linearly from a small epsilon (or 1) up to max_freq
        base_freq = 1.0  # Start with a small frequency
        frequencies = torch.linspace(base_freq, max_freq, this.c).view(this.c, 1, 1, 1)

        # 3. Compute the cosine initialization
        # The phase combines separate spatial frequencies (x and y)
        # using a simple summation (creates a diagonal wave pattern)
        # Note: repeat() is only used to broadcast/copy the 1x1xHxW coords to the Cx1xHxW phase
        phase = frequencies * (x_coords.repeat(this.c, 1, 1, 1) + y_coords.repeat(this.c, 1, 1, 1))

        # Compute the cosine: shape (c, 1, h, w) -> (c, h, w)
        cosine_init = torch.cos(phase).squeeze(1)  # Squeeze the dummy spatial channel dim

        # Reshape to (1, c, h, w) for the Parameter
        return cosine_init.unsqueeze(0);

        # --- End of Modification ---
