from torch import nn
from torch.nn import functional as trnf;

'''
Decoupled Text Decoder
'''
# Lengths are gone. and by default multipart.
class neko_DTDNG_mk2(nn.Module):

    def __init__(this,param):
        super(neko_DTDNG_mk2,this).__init__();
        this.setup_modules(param);
        this.baseline=0;

    def setup_modules(this,params):
        # this.drop=dropout;
        return;

    def getC(this, feature, A, nB, nC, nH, nW, nT,nP):
        C = feature.view(nB, 1,1, nC, nH, nW) * A.view(nB, nT, nP, 1, nH, nW)
        C = C.reshape(nB, nT,nP, nC, -1).sum(-1);
        return C;


    def sample(this,feature,A):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1];
        nP=A.shape[2];

        # Normalize
        # OOF! is this the cause for the bleeding and performance impact?????
        if(A.shape[-1] != feature.shape[-1]):
            RA=trnf.interpolate(A.view(nB,nT*nP,A.shape[-2],A.shape[-1]),[feature.shape[2],feature.shape[3]],mode="bilinear").reshape(nB,nT,nP,nH,nW);
        else:
            RA=A;
        RA = RA / (RA.view(nB, nT,nP, -1).sum(-1).view(nB, nT,nP, 1, 1)+0.0001)
        # weighted sum
        C = this.getC(feature, RA, nB, nC, nH, nW, nT,nP);
        return RA, C;

    # if you want to decorate A, do it here.
    def forward(this, feature, A):
        A,C=this.sample(feature,A);
        return C;




