import cv2
import torch
from torch import nn
from torch.nn import functional as trnf
from torch_scatter import scatter_mean,scatter_sum
from neko_sdk.cfgtool.argsparse import neko_get_arg


class abstract_cent(nn.Module):
    PARAM_ignore_index = "ignore_index";
    PARAM_reduction = "reduction";
    DFT_reduction = "mean";
    DFT_ignore_index = -1;
    def __init__(this, param):
        super(abstract_cent, this).__init__();
        this.setuploss(param);

    def setuploss(this, param):
        this.criterion_CE = nn.CrossEntropyLoss();
        # this.aceloss=
        this.ignore_index = neko_get_arg(this.PARAM_ignore_index,param,this.DFT_ignore_index);
        this.reduction=neko_get_arg(this.PARAM_reduction,param,this.DFT_reduction);

class osclsNG(abstract_cent):
    def forward(this, outcls, label_flatten):
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(outcls.device).float();
        # w[-1] = 0;
        clsloss = trnf.cross_entropy(outcls, label_flatten, w,ignore_index=this.ignore_index,reduction=this.reduction);
        return clsloss;

class oslenlossNG(abstract_cent):
    def forward(this, lencls, gtlen):
        # w[-1] = 0.1;
        lenloss = trnf.cross_entropy(lencls,gtlen,ignore_index=this.ignore_index,reduction=this.reduction);
        return lenloss;
class osclsNG_perinstance(abstract_cent):
    DFT_reduction = "none";
    def forward(this, outcls, label_flatten,instmap):
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(outcls.device,dtype=outcls.dtype);
        # w[-1] = 0;
        clsloss = trnf.cross_entropy(outcls, label_flatten, w,ignore_index=this.ignore_index,reduction=this.reduction);
        clsloss=scatter_mean(clsloss, instmap);
        return clsloss;
class osclsNG_perinstance_less_loss_on_corr(abstract_cent):
    DFT_reduction = "none";
    PARAM_EPS="eps";
    def setuploss(this, param):
        this.criterion_CE = nn.CrossEntropyLoss();
        # this.aceloss=
        super().setuploss(param);
        this.eps = neko_get_arg(this.PARAM_EPS, param, 0.1);
    def forward(this, outcls, label_flatten,instmap):
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(outcls.device,dtype=outcls.dtype);
        # w[-1] = 0;
        clsloss = trnf.cross_entropy(outcls, label_flatten, w,ignore_index=this.ignore_index,reduction=this.reduction);
        cw=torch.ones_like(clsloss);
        cw[torch.argmax(outcls,dim=1)==label_flatten]=this.eps; # if already correct, no need to make it hyper one-hot
        instloss=scatter_mean(clsloss*cw, instmap);
        return instloss;
class osclsNG_perinstance_balanced(abstract_cent):
    DFT_reduction = "none";
    def forward(this, outcls, label_flatten,instmap):
        # do a dryrun to get grad.
        d=torch.nn.Parameter(outcls.detach(),requires_grad=True);
        c=trnf.cross_entropy(d,label_flatten);
        c.backward();
        g=torch.abs(d.grad).mean(0);
        w = torch.ones_like(torch.ones(outcls.shape[-1])).to(outcls.device,dtype=outcls.dtype);
        w[-1] = g[:-1].mean()/g[-1];

        clsloss = trnf.cross_entropy(outcls, label_flatten, w,ignore_index=this.ignore_index,reduction=this.reduction);
        clsloss=scatter_mean(clsloss, instmap);
        return clsloss;
# class osclsNG_balancedperinst(osclsNG_perinstance):
#     def forward(this, outcls, label_flatten,instmap):
#         with
#         clsloss = trnf.cross_entropy(outcls, label_flatten,ignore_index=this.ignore_index,reduction=this.reduction);
#         clsloss=scatter_mean(clsloss, instmap);
#         return clsloss;

class osclsNG_perinstance_top20(abstract_cent):
    DFT_reduction = "none";
    def manual_gather(this,outcls,label):
        all=torch.zeros_like(outcls[:,0]);
        for i in range(label.shape[0]):
            all[i]=outcls[i][label[i]];
        return all;
    def visualize(this,outcls,scale=4):
        cv2.imwrite("/run/media/lasercat/writebuffer/tmp/meow.jpg",(outcls*scale).to(torch.uint8).cpu().numpy());
        cv2.waitKey(10);
    def gradone(this,logits,labels):
        return trnf.softmax(logits,dim=1)-trnf.one_hot(labels,logits.shape[1]);
    def gradsim(this, outcls,label_flatten):
        ograd=this.gradone(outcls,label_flatten);


        tlog = outcls.gather(1, label_flatten[..., None]).reshape(-1, 1);
        tmp = outcls + 0;
        tmp.scatter_(1, label_flatten[..., None],
                     torch.zeros_like(tlog) - 9999);  # disable the logits of the right answer
        sellog,selidx= tmp.topk(k=20, dim=1);
        newlogit = torch.cat([tlog,sellog], dim=1);
        cords=torch.cat([label_flatten.unsqueeze(1),selidx],dim=1);
        rlabel = torch.zeros_like(label_flatten);  # well we shifted right answers to 0 so duh.
        rlabel[label_flatten == this.ignore_index] = this.ignore_index;
        newgrad_compact=this.gradone(newlogit,rlabel);
        newgrad=torch.zeros_like(ograd).scatter_(1,cords,newgrad_compact);

        new_grad_towards_classes=newgrad.sum(0);
        new_encourgement=new_grad_towards_classes[new_grad_towards_classes<0];
        new_discourgement = new_grad_towards_classes[new_grad_towards_classes > 0];

        old_grad_towards_classes=ograd.sum(0);
        old_encourgement=old_grad_towards_classes[old_grad_towards_classes<0];
        old_discourgement = old_grad_towards_classes[old_grad_towards_classes > 0];

        pass;



    def forward(this,outcls,label_flatten,instmap):
        w = torch.ones_like(torch.ones(21)).to(outcls.device).float();
        # w[-1] = 0;
        this.gradsim(outcls,label_flatten);
        tlog=outcls.gather(1,label_flatten[...,None]).reshape(-1,1);
        tmp=outcls+0;
        tmp.scatter_(1, label_flatten[..., None], torch.zeros_like(tlog)-9999); # disable the logits of the right answer
        newlogit=torch.cat([tlog,tmp.topk(k=20,dim=1)[0]],dim=1);
        rlabel=torch.zeros_like(label_flatten); # well we shifted right answers to 0 so duh.
        rlabel[label_flatten==this.ignore_index]=this.ignore_index;
        clsloss = trnf.cross_entropy(newlogit, rlabel, w, ignore_index=this.ignore_index,
                                     reduction=this.reduction);
        clsloss = scatter_mean(clsloss, instmap);
        return clsloss;
class oslenlossNG_perinst(oslenlossNG):
    DFT_reduction = "none";

class noop(oslenlossNG_perinst):
    def forward(this, lencls, gtlen):
        # w[-1] = 0.1;
        lenloss = trnf.cross_entropy(lencls, gtlen, ignore_index=this.ignore_index, reduction=this.reduction);
        lenloss=torch.zeros_like(lenloss,requires_grad=True);
        return lenloss;

class osocrlossNG(nn.Module):
    def __init__(this, param):
        super(osocrlossNG, this).__init__();
        this.clsloss=osclsNG(param);
        this.lenloss = oslenlossNG(param);
    def forward(this, outcls,lencls,label_flatten,gtlen_):
        # w[-1] = 0.1;
        clsloss=this.clsloss(outcls,label_flatten);
        # label_flattenk=label_flatten+0;
        # label_flattenk[label_flatten==(outcls.shape[-1]-1)]=-1;
        # clslossk = this.clsloss(outcls, label_flattenk);
        gtlen=gtlen_.clone()
        with torch.no_grad():
            gtlen[gtlen>=lencls.shape[-1]]=-1;
        lenloss = this.lenloss(lencls,gtlen);
        return lenloss+clsloss, {"cls_loss":clsloss.item(), "len_loss":lenloss.item()};


class osocrlossNG_perinst(nn.Module):
    def __init__(this, param):
        super(osocrlossNG_perinst, this).__init__();
        this.clsloss=osclsNG_perinstance(param);
        this.lenloss = oslenlossNG_perinst(param);
    def forward(this, outcls,lencls,label_flatten,gtlen_,mapping):
        # w[-1] = 0.1;
        clsloss=this.clsloss(outcls,label_flatten,mapping);


        # label_flattenk=label_flatten+0;
        # label_flattenk[label_flatten==(outcls.shape[-1]-1)]=-1;
        # clslossk = this.clsloss(outcls, label_flattenk);
        gtlen=gtlen_.clone()
        with torch.no_grad():
            gtlen[gtlen>=lencls.shape[-1]]=-1;
        lenloss = this.lenloss(lencls,gtlen);
        return lenloss+clsloss, {"cls_loss":clsloss.detach().cpu().numpy(), "len_loss":lenloss.detach().cpu().numpy()};

