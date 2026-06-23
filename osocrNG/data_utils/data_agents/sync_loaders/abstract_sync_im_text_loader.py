import cv2

from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.data.neko_data_source_NG import neko_named_multi_source_holder
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from osocrNG.data_utils.neko_im_label_lmdb_holder import neko_im_label_lmdb_holder

import random


from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment

class neko_abstract_im_text_sync_fetching_agent(ama):

    PARAM_batch_size="batch_size";
    OUTPUT_images="image";
    OUTPUT_text="text";
    OUTPUT_dscps="dscps"; # data descp;
    PARAM_force_grey="force_grey";
    def imlst_as_grey(this,lst):
        return [i.convert("L").convert("RGB") for i in lst]
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        pass;

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.images = this.register_output(this.OUTPUT_images, iocvt_dict);
        this.text = this.register_output(this.OUTPUT_text, iocvt_dict);
        this.dscps=this.register_output(this.OUTPUT_dscps,iocvt_dict);
        pass;
    def set_etc(this,params):
        this.force_grey=neko_get_arg(this.PARAM_force_grey,params,False);
        this.batch_size=neko_get_arg(this.PARAM_batch_size,params,128);
class neko_abstract_im_text_sync_fetching_agent_randomized(neko_abstract_im_text_sync_fetching_agent):
    PARAM_seed="seed";
    def set_etc(this,params):
        super().set_etc(params);
        this.seed=neko_get_arg(this.PARAM_seed,params,9);
        this.rng=random.Random(this.seed);



class neko_abstract_im_text_sync_random_multidb(neko_abstract_im_text_sync_fetching_agent_randomized):
    PARAM_names = "names";
    PARAM_roots = "roots";
    def set_holder(this):
        sourced = {};
        for n, r in zip(this.names, this.roots):
            sourced[n] = neko_im_label_lmdb_holder({
                neko_im_label_lmdb_holder.PARAM_root: r,
                neko_im_label_lmdb_holder.PARAM_vert_to_hori: False
            });
        this.holder = neko_named_multi_source_holder({
            neko_named_multi_source_holder.PARAM_sources: this.names,
            neko_named_multi_source_holder.PARAM_sourced: sourced,
        })

    def set_etc(this,params):
        super().set_etc(params);
        this.names = neko_get_arg(this.PARAM_names, params);
        this.roots = neko_get_arg(this.PARAM_roots, params);
        this.set_holder();

