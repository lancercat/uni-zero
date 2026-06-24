from torch import nn
from torch.nn import functional as trnf;

'''
Decoupled Text Decoder
'''
# these dtds does not decode prediction according to prev timestamp classifications.
# this is to get less bloating code. This starts from mk5.
# to support conventional APIs, go for GPDTDs
class neko_DTDNG_mk1(nn.Module):

    def __init__(this,param):
        super(neko_DTDNG_mk1,this).__init__();
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
        pass;

    # we may need a forward_time_stamp here or may be insert a call back on the classifier. Let's see.
    def forward(this, feature, A,  text_length):
        # If we

        if(text_length is not None):
            nsteps = max(1,int(max(text_length)))
        else:
            nsteps = A.size()[1]
        A,C=this.sample(feature,A[:,:nsteps]);
        return C;


'''
Decoupled Text Decoder
'''
# these dtds does not decode prediction according to prev timestamp classifications.
# this is to get less bloating code. This starts from mk5.
# to support conventional APIs, go for GPDTDs
# dropout will be delegated outside this.

class neko_DTDNG_mk1mp(neko_DTDNG_mk1):

    def __init__(this,param):
        super(neko_DTDNG_mk1,this).__init__();
        this.setup_modules(param);
        this.baseline=0;

    # we may need a forward_time_stamp here or may be insert a call back on the classifier. Let's see.
    def forward(this, feature, A,  text_length):
        nB, nC, nH, nW = feature.size();
        A_=A[:,:1]*A[:,1:];
        A_,C=this.sample(feature,A_);
        if(this.training and text_length is not None):
            nsteps = int(text_length.max())
        else:
            nsteps = A_.size()[1]
        out_emb=this.loop(C,nsteps,nB);
        return out_emb;
