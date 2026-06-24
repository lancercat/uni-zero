
from neko_sdk.cfgtool.argsparse import neko_get_arg
import numpy as np
from osocrNG.modular_agents_ocrNG.filter_agents.abstract_pre_filter_agent import neko_pre_filter_agent
class neko_pre_closest_aspect_ratio_filter_agent_np(neko_pre_filter_agent):
    PARAM_anchor_wh="ancar_wh";
    PARAM_k_best="k";
    def set_etc(this,param):
        awhs = neko_get_arg(this.PARAM_anchor_wh, param);
        this.k=neko_get_arg(this.PARAM_k_best,param,1);
        this.ars=[awh[0]/awh[1] for awh in awhs];
        this.acnt=len(this.ars);
    def ar(this,raw_im):
        return raw_im.shape[1]/raw_im.shape[0]
    def filter(this,raw_ims,basemsk,environment,workspace):
        for i in range( basemsk.shape[0]):
            ar = this.ar(raw_ims[i]);
            arr=[max(aar,ar)/min(aar,ar) for aar in this.ars];
            ord = np.argsort(arr);
            flag=this.k;
            for aid in ord:
                if (basemsk[i][aid]):
                    if(flag):
                        flag-=1;
                    else:
                        basemsk[i][aid] = 0;
        return basemsk;

def get_neko_pre_closest_aspect_ratio_filter_agent_np(raw_ims_name,restrict_mask_in_name,
restrict_mask_out_name,anchor_wh,k_best):
    engine = neko_pre_closest_aspect_ratio_filter_agent_np;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_raw_ims: raw_ims_name, engine.INPUT_restrict_mask_in: restrict_mask_in_name, engine.OUTPUT_restrict_mask_out: restrict_mask_out_name}, engine.PARAM_anchor_wh: anchor_wh, engine.PARAM_k_best: k_best, "modcvt_dict": {}}}

if __name__ == '__main__':
    print(neko_pre_closest_aspect_ratio_filter_agent_np.get_default_configuration_scripts())
