import torch
from torch import nn


class temporal_fe_fpn(torch.nn.Module):
    PARAM_scales="scales";
    PARAM_num_se_channels="num_se_channels";
    PARAM_depth="depth";
    PARAM_num_channels="num_channels";


    def forward(this,input):
        x = input[0]
        for i in range(0, len(this.fpn)):
            x = this.fpn[i](x) + input[i + 1]
        conv_feats = []
        stoppers = []
        for i in range(0, len(this.convs)):
            x = this.convs[i](x)
            conv_feats.append(x)
        stoppers.append(x);
        for i in range(0, len(this.deconvs) -1):
            x = this.deconvs[i](x)
            f = conv_feats[len(conv_feats) - 2 - i]
            stoppers.append(x);
            x = x[:, :, :f.shape[2], :f.shape[3]] + f
        x=this.deconvs[-1](x);
        stoppers.append(x);
        return x,stoppers


    def __init__(this, params):

        super(temporal_fe_fpn, this).__init__();

        # scales, nmasks, depth, maxT, num_channels, num_se_channels
        # cascade multiscale features
        scales=params[this.PARAM_scales];
        depth=params[this.PARAM_depth];
        num_channels=params[this.PARAM_num_channels];


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
        this.endpoint_chs=[num_channels];

        deconvs = []

        for i in range(1, int(depth / 2)+1):
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,
                                                           tuple(deconv_ksizes[int(depth/2)-i]),
                                                           tuple(strides[int(depth/2)-i]),
                                                           (int(deconv_ksizes[int(depth/2)-i][0]/4.), int(deconv_ksizes[int(depth/2)-i][1]/4.))),
                                         nn.BatchNorm2d(num_channels),
                                         nn.ReLU(True)));
            this.endpoint_chs.append( num_channels);
        this.deconvs = nn.Sequential(*deconvs)