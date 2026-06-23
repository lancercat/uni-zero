import torch
from torch_scatter import scatter_max, scatter_min,scatter_sum;
def id_cvt(pred,label):
    return pred;
def scatter_cvt_d(pred,label,dim=-1):
    dev=pred.device;
    label=label.long().to(dev);
    pred=pred.cpu();
    label=label.cpu();
    return scatter_max(pred,label,dim)[0].cuda();
def scatter_cvt_min(pred, label, dim=-1):
    # The old one seems causing locking problem when launched in parallel. Will see after updating to driver 465
    # return scatter_cvt_d(pred,label,dim)
    dev = pred.device;
    label = label.long().to(dev);
    return scatter_min(pred,label,dim)[0];

def scatter_cvt(pred, label, dim=-1):
    # If you are using AMD gpus....
    # return scatter_cvt_d(pred,label,dim)
    dev = pred.device;
    label = label.long().to(dev);
    return scatter_max(pred,label,dim)[0];

# some time we want similarity constraints....
def flatten_range(cnt):
    ter=torch.cumsum(cnt,dim=-1);
    sta=ter-cnt;
    return sta,ter;

# we will cudafy this, if it works.
def sample_proto_matching(slabel,plabel,device="cpu"):
    plcnt = scatter_sum(torch.ones_like(plabel), plabel);
    plsta,plter=flatten_range(plcnt);
    scnt=plcnt[slabel];
    ssta, ster = flatten_range(scnt);
    phandle=torch.zeros(ster[-1].item(),dtype=torch.int64);
    shandle = torch.zeros(ster[-1].item(),dtype=torch.int64);
    arrange = torch.arange(plter[-1].item(),dtype=torch.int64);
    for i in range(slabel.shape[0]):
        shandle[ssta[i].item():ster[i].item()]=i;
        lab=slabel[i].item();
        pidfrom=plsta[lab];
        pidto=plter[lab];
        phandle[ssta[i].item():ster[i].item()]=arrange[pidfrom:pidto];
    return shandle.to(device), phandle.to(device);




def scatter_cvt2(pred, label, dim=-1):
    # return scatter_cvt_d(pred,label,dim)
    dev = pred.device;
    return scatter_max(pred,label,dim)[0];
if __name__ == '__main__':
    pass;
    # def random_data
