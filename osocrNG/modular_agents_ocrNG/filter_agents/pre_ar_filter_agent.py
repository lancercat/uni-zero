
from neko_sdk.cfgtool.argsparse import neko_get_arg
from osocrNG.modular_agents_ocrNG.filter_agents.abstract_pre_filter_agent import neko_pre_filter_agent
class neko_pre_aspect_ratio_filter_agent_np(neko_pre_filter_agent):
    PARAM_max_ars="maxar_wh";
    PARAM_min_ars="minar_wh";
    def set_etc(this,param):
        this.max_ars = neko_get_arg(this.PARAM_max_ars, param);
        this.min_ars = neko_get_arg(this.PARAM_min_ars, param);
        this.acnt=len(this.max_ars);
    def ar(this,raw_im):
        return raw_im.shape[1]/raw_im.shape[0]
    def filter(this,raw_ims,basemsk,environment,workspace):
        for i in range( basemsk.shape[0]):
            ar = this.ar(raw_ims[i]);
            for t in range(len(this.max_ars)):
                if (ar< this.min_ars[t] or ar > this.max_ars[t]):
                    basemsk[i][t] = 0;
        return basemsk;

class neko_pre_aspect_ratio_filter_agent_tensor(neko_pre_aspect_ratio_filter_agent_np):
    def ar(this, raw_im):
        return raw_im.shape[-2] / raw_im.shape[-1];
def get_neko_pre_aspect_ratio_filter_agent_np(raw_ims_name,restrict_mask_in_name,
restrict_mask_out_name,max_ars,min_ars):
    engine = neko_pre_aspect_ratio_filter_agent_np;
    return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_raw_ims: raw_ims_name, engine.INPUT_restrict_mask_in: restrict_mask_in_name, engine.OUTPUT_restrict_mask_out: restrict_mask_out_name}, engine.PARAM_max_ars: max_ars, engine.PARAM_min_ars: min_ars, "modcvt_dict": {}}}

if __name__ == '__main__':
    print(neko_pre_aspect_ratio_filter_agent_np.get_default_configuration_scripts())
