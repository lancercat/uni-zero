import numpy as np
import torch
from torch.nn import functional as trnf
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment


def dump_att_im_grp(tensorim_,TA__,cnt):
    tensorim=tensorim_.detach().cpu();
    H,W=tensorim.shape[2],tensorim.shape[3];

    TA_=TA__.detach().cpu();
    if(len(TA_.shape)==5):
        N,T,P,AH,AW=TA_.shape;
        TA=TA_.reshape(N,T*P,AH,AW)
        TA=trnf.interpolate(TA, [H, W], mode="bilinear");
    elif(len(TA_.shape)==4):
        N,T,AH,AW=TA_.shape;
        P=1;
        TA = TA_+0;
        TA=trnf.interpolate(TA, [H, W], mode="bilinear");
    else:
        TA=None
        fatal("This does not seem to be a temporal attention tensor");
    TA = TA.reshape(N, T, P, H, W);
    alst=[];
    for n in range(N):
        slst=[];
        for t in range(cnt[n]):
            glst=[];
            for p in range(P):
                tim=((tensorim[n]*127+127) * (TA[n][t][p] * 0.9 + 0.1)).permute(1,2,0);
                glst.append(tim);
            slst.append(torch.stack(glst,0).max(dim=0)[0].numpy().astype(np.uint8));
        alst.append(slst);
    return alst;


class attention_visualization_agent(neko_module_wrapping_agent):
    INPUT_attention_map="attmsk";
    INPUT_tensor_images = "tensor_im";
    INPUT_tensor_length= "tensor_len";
    OUTPUT_visualized_masks="vismsk";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.attention_map = this.register_input(this.INPUT_attention_map, iocvt_dict);
        this.tensor_images = this.register_input(this.INPUT_tensor_images, iocvt_dict);
        this.tensor_length = this.register_input(this.INPUT_tensor_length, iocvt_dict);
        this.visualized_masks = this.register_output(this.OUTPUT_visualized_masks, iocvt_dict);
        pass;


    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        if(this.attention_map not in  workspace.inter_dict):
            return workspace, environment;
        attention_map = workspace.get(this.attention_map);

        tensor_images = workspace.get(this.tensor_images);
        tensor_length = workspace.get(this.tensor_length);
        vismap=dump_att_im_grp(tensor_images,attention_map,tensor_length);
        workspace.add(this.visualized_masks,vismap);
        return workspace,environment;
def get_attention_visualization_agent(
        attention_map, tensor_images,tensor_len,
    visualized_masks
):
    engine = attention_visualization_agent;
    return {"agent": engine, "params": {
        "iocvt_dict": {engine.INPUT_attention_map: attention_map, engine.INPUT_tensor_images: tensor_images, engine.INPUT_tensor_length: tensor_len,
                       engine.OUTPUT_visualized_masks: visualized_masks}, "modcvt_dict": {}}}


if __name__ == '__main__':
    # ws = torch.load("/run/media/lasercat/nep_core/cnc_coredump/wsdump219.pt");
    # ts=ws["vert_raw_label"];
    # cnt=[len(t) for t in ts];
    attention_visualization_agent.print_default_setup_scripts();


    # dump_att_im_grp()
