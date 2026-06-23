from torch import nn
from neko_sdk.cfgtool.argsparse import neko_get_arg
class neko_predictive_mlp(nn.Module):

    PARAM_in_dim="in_dim";
    PARAM_hidden_dim="hidden_dim";
    PARAM_out_dim = "out_dim";
    def __init__(this,params):
        super().__init__();
        this.indim = neko_get_arg(this.PARAM_in_dim, params);

        this.outdim = neko_get_arg(this.PARAM_out_dim, params);
        this.hiddendim = neko_get_arg(this.PARAM_hidden_dim, params,2*this.outdim);
        this.core=nn.Sequential(
            nn.Linear(this.indim,this.hiddendim),
            nn.BatchNorm1d(this.hiddendim,this.hiddendim),
            nn.ReLU(),
            nn.Linear(this.hiddendim, this.outdim),
        );
    def forward(this,x):
        return this.core(x);

class neko_normed_mlp(nn.Module):

    PARAM_in_dim="in_dim";
    PARAM_hidden_dim="hidden_dim";
    PARAM_out_dim = "out_dim";
    def __init__(this,params):
        super().__init__();
        this.indim = neko_get_arg(this.PARAM_in_dim, params);

        this.outdim = neko_get_arg(this.PARAM_out_dim, params);
        this.hiddendim = neko_get_arg(this.PARAM_hidden_dim, params,2*this.outdim);
        this.fc1=nn.Linear(this.indim,this.hiddendim);
        this.bn1=nn.BatchNorm1d(this.hiddendim,this.hiddendim);
        this.relu=nn.ReLU();
        this.fc2=nn.Linear(this.hiddendim, this.outdim);
        this.bn2=nn.BatchNorm1d(this.outdim, this.outdim);
    def forward_stub(this,x):
        return this.relu(this.bn2(this.fc2(this.relu(this.bn1(this.fc1(x))))));
    def forward(this,x):

        if (x.shape[0] == 1):
            training = this.training;
            this.bn1.eval();
            this.bn2.eval();
            y=this.forward_stub(x);
            this.bn1.train(training);
            this.bn2.train(training);
            return y
        else:
            return this.forward_stub(x);
