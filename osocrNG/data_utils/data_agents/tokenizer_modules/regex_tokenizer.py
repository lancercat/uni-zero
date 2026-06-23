from torch import nn
from neko_sdk.NDK.tokenizer.regex_ocr_tokenize import tokenize

class neko_regex_tokenize(nn.Module):
    def __init__(this,params):
        super().__init__();
    def forward(this,plain_text_list):
        return [tokenize(s) for s in plain_text_list];
