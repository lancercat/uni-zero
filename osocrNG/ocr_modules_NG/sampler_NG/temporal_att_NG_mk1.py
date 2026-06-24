import torch
from torch import nn
from torch.nn import functional as trnf

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_spatial_kit.embeddings.neko_emb_intr import neko_add_embint_se

from neko_sdk.log import fatal

# Current version only supports input whose size is a power of 2, such as 32, 64, 128 etc.
# You can adapt it to any input size by changing the padding or stride.

class neko_vmsk(nn.Module):
    PARAM_num_channels="num_channels";
    def __init__(this,param):
        super().__init__();
        num_channels=neko_get_arg(this.PARAM_num_channels,param,512);
        this.core=nn.Sequential(
                nn.Conv2d(num_channels, 1, 1), nn.Sigmoid());
    def forward(this,x):
        A=this.core(x);
        V=A*x;
        R=V.sum(-1).sum(-1)/(A.sum(-1).sum(-1)+0.000000000000009);
        return R,A;

# sampling is no more taken care by the lcam
class LCAM_NG_mk1(torch.nn.Module):
    def arm_pred_stop(this,ich_stop,maxT):
        this.lenpred= nn.Linear(ich_stop,maxT+1,False);
    def arm_last_deconv(this,deconvs,num_channels, nmasks,deconv_ksize,stride):
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, nmasks,
                                                        tuple(deconv_ksize),
                                                        tuple(stride),
                                                        (int(deconv_ksize[0] / 4.), int(deconv_ksize[1] / 4.))),
                                     nn.Sigmoid()));
        return deconvs;
    def arm_semodule(this, params):
        scales=params["scales"];
        num_se_channels=params["num_se_channels"];
        this.semods=[];
        if(type(num_se_channels) is int):
            for cid in range(len(scales)):
                this.semods.append(neko_add_embint_se(scales[cid][1],scales[cid][2],num_se_channels));
                this.add_module("se_"+str(cid),this.semods[-1]);
        else:
            fatal("error");



    def make_att(this, x, t):
        return this.deconvs[-1](x)


    def arm_vmsks(this,params):
        conv_vmsks=[];
        deconv_vmsks=[];

        for i in range(0, int(params["depth"] / 2)):
            conv_vmsks.append(neko_vmsk(params));
        this.conv_vmsks=nn.Sequential(*conv_vmsks);

        for i in range(1, int(params["depth"] / 2)):
            deconv_vmsks.append(neko_vmsk(params));
        this.deconv_vmsks=nn.Sequential(*deconv_vmsks);

    def arm_stem(this,params):
        # scales, nmasks, depth, maxT, num_channels, num_se_channels
        # cascade multiscale features
        this.arm_semodule( params);
        scales=params["scales"];
        depth=params["depth"];
        num_channels=params["num_channels"];
        this.maxT=params["maxT"];
        this.n_parts=params["n_parts"];
        nmasks=this.n_parts*this.maxT;


        fpn = []
        for i in range(1, len(scales)):
            assert not (scales[i-1][1] / scales[i][1]) % 1, 'layers scale error, from {} to {}'.format(i-1, i)
            assert not (scales[i-1][2] / scales[i][2]) % 1, 'layers scale error, from {} to {}'.format(i-1, i)
            ksize = [3,3,5] # if downsampling ratio >= 3, the kernel size is 5, else 3
            r_h, r_w = int(scales[i-1][1] / scales[i][1]), int(scales[i-1][2] / scales[i][2])
            ksize_h = 1 if scales[i-1][1] == 1 else ksize[r_h-1]
            ksize_w = 1 if scales[i-1][2] == 1 else ksize[r_w-1]
            fpn.append(nn.Sequential(nn.Conv2d(scales[i-1][0], scales[i][0],
                                              (ksize_h, ksize_w),
                                              (r_h, r_w),
                                              (int((ksize_h - 1)/2), int((ksize_w - 1)/2))),
                                     nn.BatchNorm2d(scales[i][0]),
                                     nn.ReLU(True)))
        this.fpn = nn.Sequential(*fpn)
        # convolutional alignment
        # convs
        assert depth % 2 == 0, 'the depth of CAM must be a even number.'
        this.arm_vmsks(params);
        in_shape = scales[-1]
        strides = []
        conv_ksizes = []
        deconv_ksizes = []
        h, w = in_shape[1], in_shape[2]
        for i in range(0, int(depth / 2)):
            stride = [2] if 2 ** (depth/2 - i) <= h else [1]
            stride = stride + [2] if 2 ** (depth/2 - i) <= w else stride + [1]
            strides.append(stride)
            conv_ksizes.append([3, 3])
            deconv_ksizes.append([_ ** 2 for _ in stride])
        convs = [nn.Sequential(nn.Conv2d(in_shape[0], num_channels,
                                        tuple(conv_ksizes[0]),
                                        tuple(strides[0]),
                                        (int((conv_ksizes[0][0] - 1)/2), int((conv_ksizes[0][1] - 1)/2))),
                               nn.BatchNorm2d(num_channels),
                               nn.ReLU(True))]

        for i in range(1, int(depth / 2)):
            convs.append(nn.Sequential(nn.Conv2d(num_channels, num_channels,
                                                tuple(conv_ksizes[i]),
                                                tuple(strides[i]),
                                                (int((conv_ksizes[i][0] - 1)/2), int((conv_ksizes[i][1] - 1)/2))),
                                       nn.BatchNorm2d(num_channels),
                                       nn.ReLU(True)))

        this.convs = nn.Sequential(*convs);
        # deconvs
        ich_stop=num_channels;

        deconvs = []

        for i in range(1, int(depth / 2)):
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,
                                                           tuple(deconv_ksizes[int(depth/2)-i]),
                                                           tuple(strides[int(depth/2)-i]),
                                                           (int(deconv_ksizes[int(depth/2)-i][0]/4.), int(deconv_ksizes[int(depth/2)-i][1]/4.))),
                                         nn.BatchNorm2d(num_channels),
                                         nn.ReLU(True)));
            ich_stop += num_channels;
        this.arm_pred_stop(ich_stop,this.maxT);
        deconvs =this.arm_last_deconv(deconvs,num_channels, nmasks,deconv_ksizes[0],strides[0]);
        this.deconvs = nn.Sequential(*deconvs)



    def att_len_core(this,input,mask,t_override):
        x = input[0]
        for i in range(0, len(this.fpn)):
            x = this.fpn[i](x) + input[i + 1]
        conv_feats = []
        stoppers = []
        for i in range(0, len(this.convs)):
            x = this.convs[i](x)
            conv_feats.append(x)
        stoppers.append(this.conv_vmsks[-1](x)[0]);
        for i in range(0, len(this.deconvs) - 1):
            x = this.deconvs[i](x)
            f = conv_feats[len(conv_feats) - 2 - i]
            stoppers.append(this.deconv_vmsks[i](x)[0]);
            x = x[:, :, :f.shape[2], :f.shape[3]] + f
        Ts = torch.cat(stoppers, dim=-1);
        leng = this.lenpred(Ts);
        if (t_override is None):
            t_override = leng.argmax(-1)[0];
        att = this.make_att(x, t_override);
        if (mask is not None):
            if (mask.shape[-2] != x.shape[-2] or mask.shape[-1] != x.shape[-1]):
                mask_ = trnf.interpolate(mask, [x.shape[-2], x.shape[-1]], mode="bilinear");
            else:
                mask_ = mask;
            att = att * mask_;

        return att.reshape([att.shape[0],this.maxT,this.n_parts,att.shape[2],att.shape[3]]), leng;

    def forward(this, input_,mask=None,t_override=None):
        if this.detached:
            input=[this.semods[i](input_[i].detach()) for i in range(len(input_))]
        else:
            input=[this.semods[i](input_[i]) for i in range(len(input_))]
        return this.att_len_core(input,mask,t_override);


    def __init__(this, param):
        super(LCAM_NG_mk1, this).__init__();
        this.arm_stem(param)
        this.detached=neko_get_arg("detached",param,True);
        this.n_parts = neko_get_arg("n_parts",param,1);
