
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.data.neko_data_source_NG import neko_named_multi_source_holder
from osocrNG.data_utils.indexer.multi_lmdb_indexer_mk2 import neko_multi_lmdb_enumerator_rand_seed_mk2 as idxmk2
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent as ama
from osocrNG.data_utils.neko_im_label_lmdb_holder import neko_im_label_lmdb_holder
from osocrNG.data_utils.raw_names import basic_data_info_mk2 as BIN2
from osocrNG.data_utils.data_agents.sync_loaders.abstract_sync_im_text_loader import neko_abstract_im_text_sync_random_multidb
# well most of our datasets are like this...
# lsct does not go with this agent anyhow
class neko_multilmdb_im_text_sync_fetching_agent(neko_abstract_im_text_sync_random_multidb):

    def set_etc(this, params):
        super().set_etc(params);
        this.indexer = idxmk2(
            {
                idxmk2.PARAM_seed:this.seed,
                idxmk2.PARAM_lengths:[this.holder.sourced[k].nSamples for k in this.names],
            }
        )

        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        imlst=[];
        utflst=[];
        idxlst=[];

        i=0;
        while i<this.batch_size:
            idx=this.indexer.__next__();
            data = this.holder.fetch_item(idx);
            if (data is not None):
                imlst.append(data[BIN2.IMAGE]);
                utflst.append(data[BIN2.LABEL]);
                idxlst.append(idx);
                i+=1;
        workspace.add(this.images,imlst);
        workspace.add(this.text,utflst);
        workspace.add(this.dscps,idxlst);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   dscps,images, text,
                   batch_size, names, roots, seed
                   ):
        return {"agent": cls, "params": {"iocvt_dict": {cls.OUTPUT_dscps:dscps,cls.OUTPUT_images: images, cls.OUTPUT_text: text},
                                         cls.PARAM_batch_size: batch_size, cls.PARAM_names: names,
                                         cls.PARAM_roots: roots, cls.PARAM_seed: seed, "modcvt_dict": {}}}


if __name__ == '__main__':
    if __name__ == '__main__':
        # neko_balance_fetching_and_mixing_agent.print_default_setup_scripts()
        from neko_2024_NGNW.nets_v6.anchors import get_wna_v6_dcfg_1h1v1r_2_05_smol as trdcfg

        agte = neko_multilmdb_im_text_sync_fetching_agent
        acfg = agte.get_agtcfg(
            "image", "text", 24, ["rctwtrdb_seen"],
            ["/home/lasercat/ssddata/rctwtrdb_seen"]
            , 9
        );
        a = agte(acfg["params"]);
        ws = neko_workspace();
        e = neko_environment();
        a.take_action(ws, e);
        pass;

