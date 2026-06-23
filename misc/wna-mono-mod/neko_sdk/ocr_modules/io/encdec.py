# decoding prediction into text with geometric-mean probability
# the probability is used to select the more realiable prediction when using bi-directional decoders
import regex
import torch;
from torch.nn import functional as trnf;

from neko_sdk.ocr_modules.sptokens import tUNK, tSPLIT


def split( s,token=tSPLIT):
    return regex.findall(token, s, regex.U);

def encode_core( tdict, label_batch,token=tSPLIT,device="cpu",EOS=None):
    length = [len(split(s, token)) for s in label_batch];
    max_len = max(length)
    if(EOS is not None):
        out = torch.zeros(len(label_batch), max_len + 1).long() + EOS;
    else:
        out = torch.zeros([len(label_batch), max_len], dtype=torch.long, device=device)
    for i in range(0, len(label_batch)):
        cur_encoded = torch.tensor([tdict[char] if char in tdict else tdict[tUNK]
                                    for char in split(label_batch[i], token)], device=device)
        out[i][0:len(cur_encoded)] = cur_encoded
    return out,length

def encode_fn_naive(tdict, label_batch,case_sensitive ,EOS,token=tSPLIT,device="cpu"):
    if (not case_sensitive):
        label_batch = [l.lower() for l in label_batch];
    return encode_core( tdict, label_batch,token=token,device=device,EOS=EOS);
def encode_fn_naive_noeos( tdict, label_batch,case_sensitive,token=tSPLIT,device="cpu"):
    if (not case_sensitive):
        label_batch = [l.lower() for l in label_batch];
    return encode_core( tdict, label_batch,token=token,device=device);

def decode_prob(net_out, length,tdict):
    out = []
    out_prob = []
    net_out = trnf.softmax(net_out, dim=1)
    for i in range(0, length.shape[0]):
        current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[1][:, 0].tolist()
        current_text = ''.join([tdict[_] if _ > 0 and _ <= len(tdict) else '' for _ in current_idx_list])
        current_probability = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:, 0]
        current_probability = torch.exp(torch.log(current_probability).sum() / current_probability.size()[0])
        out.append(current_text)
        out_prob.append(current_probability)
    return (out, out_prob)