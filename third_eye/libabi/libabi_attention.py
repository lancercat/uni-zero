import torch
import torch.nn as nn
from neko_sdk.cfgtool.argsparse import neko_get_arg

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(this, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, this).__init__()
        this.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        this.register_buffer('pe', pe)

    def forward(this, x):


        x = x + this.pe[:x.size(0), :]
        return this.dropout(x)


class ABINet_Attention(nn.Module):
    PARAM_number_channels="number_channels";
    PARAM_n_parts="n_parts";
    PARAM_maxT="maxT";
    PARAM_n_features="n_features"
    def __init__(this,params):
        max_length=neko_get_arg(this.PARAM_maxT,params);
        n_feature=neko_get_arg(this.PARAM_n_features,params,256);
        in_channels=neko_get_arg(this.PARAM_number_channels,params);

        super().__init__()

        this.max_length = max_length;
        this.n_parts=neko_get_arg(this.PARAM_n_parts,params,1);

        this.f0_embedding = nn.Embedding(max_length*this.n_parts, in_channels)
        this.w0 = nn.Linear(max_length*this.n_parts, n_feature)
        this.wv = nn.Linear(in_channels, in_channels)
        this.we = nn.Linear(in_channels, max_length*this.n_parts)

        this.active = nn.Tanh()
        this.softmax = nn.Softmax(dim=2)

    def forward(this, enc_output):
        enc_output = enc_output.permute(0, 2, 3, 1).flatten(1, 2)
        reading_order = torch.arange(this.max_length, dtype=torch.long, device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(enc_output.size(0), -1)  # (S,) -> (B, S)
        reading_order_embed = this.f0_embedding(reading_order)  # b,25,512

        t = this.w0(reading_order_embed.permute(0, 2, 1))  # b,512,256
        t = this.active(t.permute(0, 2, 1) + this.wv(enc_output))  # b,256,512

        attn = this.we(t)  # b,256,25
        attn = this.softmax(attn.permute(0, 2, 1))  # b,25*1,256
        return attn.view(attn.shape[0],this.max_length,this.n_parts,enc_output.shape[-2],enc_output.shape[-1]);



def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))


def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode == 'nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor,
                                     mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))

class decorder_asize(nn.Module):
    def __init__(this,in_c, out_c, k=3, s=1, p=1, mode='nearest'):
        super().__init__();
        this.mode=mode;
        this.align_corners = None if mode == 'nearest' else True
        this.core=nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                             nn.BatchNorm2d(out_c),
                             nn.ReLU(True))
    def forward(this,feat,size):
        rf=nn.functional.upsample(feat,size=size,
                    mode=this.mode, align_corners=this.align_corners);
        f=this.core(rf);
        return f;



class ABINet_PositionAttention(nn.Module):
    PARAM_maxT="maxT";
    PARAM_in_channels="in_channels";

    def __init__(this, params):
        super().__init__();
        max_length=neko_get_arg(this.PARAM_maxT,params);
        in_channels = neko_get_arg(this.PARAM_in_channels,params,512);
        num_channels = neko_get_arg(this.PARAM_in_channels,params,64),
        mode = 'nearest';
        this.max_length = max_length
        this.k_encoder = nn.Sequential(
            encoder_layer(in_channels, num_channels, s=(1, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2))
        )
        this.k_decoder = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),

        );
        this.fin_dec=decorder_asize(num_channels, in_channels, mode=mode);

        this.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=max_length)

    def forward(this, x):
        N, E, H, W = x.size()
        k, v = x, x  # (N, E, H, W)

        # calculate key vector
        features = []
        for i in range(0, len(this.k_encoder)+1):
            k = this.k_encoder[i](k)
            features.append(k)
        for i in range(0, len(this.k_decoder) ):
            k = this.k_decoder[i](k)
            k = k + features[len(this.k_decoder) - 2 - i]
        k = this.fin_dec(k,x.shape[-2:]);


        # calculate query vector
        # TODO q=f(q,k)
        zeros = x.new_zeros((this.max_length, N, E))  # (T, N, E)
        q = this.pos_encoder(zeros)  # (T, N, E)
        q = q.permute(1, 0, 2)  # (N, T, E)
        q = this.project(q)  # (N, T, E)

        # calculate attention
        attn_scores = torch.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        return attn_scores.view(N, -1,1, H, W)
