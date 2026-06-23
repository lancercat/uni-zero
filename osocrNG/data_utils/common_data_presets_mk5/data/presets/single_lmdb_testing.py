from neko_sdk.cfgtool.argsparse import neko_get_arg
from osocrNG.data_utils.common_data_presets_mk5.virtual_mk5_datafactory import neko_virtual_factory_mk5
from osocrNG.data_utils.common_data_presets_mk5.data.deprecated.list_im_collators import neko_im_text_collate_factory
from osocrNG.data_utils.common_data_presets_mk5.data.datarepos.ordered_lmdb_factory import single_ordered_lmdb_factory
class lmdb_testing(neko_virtual_factory_mk5):
    DATA_PRFX = "dataprfx";
    # the testing does not need expert/source balancing, just grab each and single for once...
    # no random stuff thus no seed needed.
    PARAM_root="root";
    REPO = single_ordered_lmdb_factory;
    COL = neko_im_text_collate_factory;
    PARAM_batch_size="batch_size";
    PARAM_img_size_max_area="img_size_max_area";
    PARAM_thumb_size_hw="thumb_size_hw";
    PARAM_padding_size_hw="padding_size_hw";
    @classmethod
    def arm_agts(cls, acfg, data_prfxs, mod_prfxs, agt_prfxs, params):
        batch_size=neko_get_arg(cls.PARAM_batch_size,params,32);
        acfg=cls.REPO.arm_data_repo(
            acfg,data_prfxs,mod_prfxs,agt_prfxs,
            {cls.REPO.PARAM_batch_size:batch_size,
             cls.REPO.PARAM_root:params[cls.PARAM_root]});
        acfg=cls.COL.arm_collates(
            acfg, {
                cls.COL.DATA_PRFX: data_prfxs[cls.DATA_PRFX] # there is no augmentation, pass down the image itself
            }, mod_prfxs, agt_prfxs,
            {cls.COL.PARAM_thumb_size_hw:params[cls.PARAM_thumb_size_hw],
             cls.COL.PARAM_padding_size_hw:params[cls.PARAM_padding_size_hw],
             cls.COL.PARAM_img_size_max_area:params[cls.PARAM_img_size_max_area]
             });
        return acfg;


