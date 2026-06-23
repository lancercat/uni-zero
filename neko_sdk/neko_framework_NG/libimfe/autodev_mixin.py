import torch
class neko_autodev_mixin(torch.nn.Module):
    def __init__(this):
        super().__init__();
        this.dev_ind=torch.nn.Parameter(torch.tensor(9,dtype=torch.float),requires_grad=False);
    def todev_tensor(this,what):
        return what.to(this.dev_ind.device);
