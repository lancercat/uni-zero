
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.data.neko_data_source_NG import neko_named_multi_source_holder
from osocrNG.data_utils.indexer.multi_lmdb_indexer_mk2 import neko_multi_lmdb_enumerator_rand_seed_mk2 as idxmk2
from osocrNG.data_utils.neko_im_label_lmdb_holder import neko_im_label_lmdb_holder
from osocrNG.data_utils.raw_names import basic_data_info_mk2 as BIN2
from osocrNG.data_utils.raw_names import raw_data_item_names as RN
from osocrNG.data_utils.data_agents.sync_loaders.abstract_sync_im_text_loader import neko_abstract_im_text_sync_fetching_agent

# well most of our datasets are like this...
# lsct does not go with this agent anyhow
class neko_single_im_text_sync_ordered_fetching_agent(neko_abstract_im_text_sync_fetching_agent):
    PARAM_name="name";
    PARAM_root="root";
    OUTPUT_ids="ids";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.images = this.register_output(this.OUTPUT_images, iocvt_dict);
        this.ids=this.register_output(this.OUTPUT_ids,iocvt_dict);
        this.text = this.register_output(this.OUTPUT_text, iocvt_dict);
        pass;

    def reset(this):
        this.start_id = 0;

    def len(this):
        fullbatchcnt = len(this.allidx) // this.batch_size;
        has_leftover = ((len(this.allidx) % this.batch_size) != 0);
        return fullbatchcnt + has_leftover;

    def set_etc(this, params):
        super().set_etc(params)
        this.name = neko_get_arg(this.PARAM_name, params);
        this.root = neko_get_arg(this.PARAM_root, params);
        this.start_id = 0;
        this.source = neko_im_label_lmdb_holder({
            neko_im_label_lmdb_holder.PARAM_root: this.root,
            neko_im_label_lmdb_holder.PARAM_vert_to_hori: False
        });
        this.allidx = this.source.all_valid_indexes();
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        all_imgs=[];
        all_texts_utf=[];
        all_idxs=[];
        batchidxs=this.allidx[this.start_id:this.start_id+this.batch_size];
        idxcnt=len(batchidxs);
        if(idxcnt):
            for i in batchidxs:
                item=this.source.fetch_item(i);
                if(item is not None):
                    all_imgs.append(item[RN.IMAGE]);
                    all_texts_utf.append(item[RN.LABEL]);
                    all_idxs.append(i);
            this.start_id+=idxcnt;
        if(this.force_grey):
            all_imgs=this.imlst_as_grey(all_imgs);
        if(len(all_imgs)): # add nothing if nothing to add
            workspace.add(this.images,all_imgs);
            workspace.add(this.text,all_texts_utf);
            workspace.add(this.ids,all_idxs);

    @classmethod
    def get_agtcfg(cls,
                   ids,images,  text,
                   batch_size, name, root,force_grey=False
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.OUTPUT_ids:ids, cls.OUTPUT_images: images,
                           cls.OUTPUT_text: text}, cls.PARAM_batch_size: batch_size, cls.PARAM_name: name, cls.PARAM_force_grey: force_grey,
            cls.PARAM_root: root, "modcvt_dict": {}}}


class neko_controlled_single_im_text_sync_ordered_fetching_agent(neko_single_im_text_sync_ordered_fetching_agent):
    INPUT_reset="reset";
    OUTPUT_exhausted="exhausted";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.reset_flg = this.register_input(this.INPUT_reset, iocvt_dict);
        this.exhausted = this.register_output(this.OUTPUT_exhausted, iocvt_dict);
    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        if(this.reset_flg in workspace):
            reset = workspace.get(this.reset_flg);
            if(reset):
                this.reset();
        all_imgs=[];
        all_texts_utf=[];
        all_idxs=[];
        batchidxs=this.allidx[this.start_id:this.start_id+this.batch_size];
        idxcnt=len(batchidxs);
        exhausted = False;
        if(idxcnt):
            for i in batchidxs:
                item=this.source.fetch_item(i);
                if(item is not None):
                    all_imgs.append(item[RN.IMAGE]);
                    all_texts_utf.append(item[RN.LABEL]);
                    all_idxs.append(i);
            this.start_id+=idxcnt;

        else:
            exhausted=True;
        if(this.force_grey):
            all_imgs=this.imlst_as_grey(all_imgs);
        workspace.add(this.exhausted,exhausted);
        workspace.add(this.images,all_imgs);
        workspace.add(this.text,all_texts_utf);
        workspace.add(this.ids,all_idxs);

    @classmethod
    def get_agtcfg(cls,
                   reset,
                   ids,images,  text,exhausted,
                   batch_size, name, root,force_grey=False
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_reset: reset,cls.OUTPUT_ids:ids, cls.OUTPUT_images: images, cls.OUTPUT_exhausted: exhausted,
                           cls.OUTPUT_text: text}, cls.PARAM_batch_size: batch_size, cls.PARAM_name: name,cls.PARAM_force_grey:force_grey,
            cls.PARAM_root: root, "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_single_im_text_sync_ordered_fetching_agent.print_default_setup_scripts();
