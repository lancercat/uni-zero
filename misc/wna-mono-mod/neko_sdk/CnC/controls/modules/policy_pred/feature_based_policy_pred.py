import torch.nn
from torch import nn
from torch.nn import functional as trnf
from neko_sdk.cfgtool.argsparse import neko_get_arg

# The basic router that works on one-step-one-expert level. Very basic.
# pooling is not handled here, add some middle-ware (bcs you may want to mixin some other features)
class neko_feature_based_static_router(torch.nn.Module):
    PARAM_indim="indim";
    PARAM_expert_names="expert_names";
    PARAM_stablize_with_norm="stablize_with_norm"
    def __init__(this,params):
        super().__init__();
        this.expert_names=params[this.PARAM_expert_names];
        this.num_experts=len(this.expert_names);
        if(neko_get_arg(this.PARAM_stablize_with_norm,params,True)):
            this.layer=nn.Sequential(
                nn.Linear(params[this.PARAM_indim],this.num_experts,bias=False),
                nn.BatchNorm1d(this.num_experts,affine=False)
            );
        else:
            this.layer = nn.Sequential(
                nn.Linear(params[this.PARAM_indim], this.num_experts, bias=False),
            );
    def forward(this,feature):
        return this.layer(feature.reshape([feature.shape[0],-1])),this.expert_names;


# I bet the model would just figure the choosing it self
# if the experts are really specialized.
# If the experts are close enough for the training data,
# the router would choose the smaller model
