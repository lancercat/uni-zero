from neko_sdk.chunked_FE.res45.res45G2 import neko_r45_layers_origNG, neko_r45_norms_origNG

from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN
from neko_2025_NGNW.common.object_32x_presets.mod_names import project_32x_modnames as MN
from neko_2025_NGNW.common.object_32x_presets.agt_names import project_32x_agtnames as AN

from neko_sdk.chunked_FE.res45.res45g4.res45_g4_g2ish import neko_res45_bogo_g4_g2ish, neko_res45_bogo_g4_g2ish_nf
from neko_2025_NGNW.common.object_32x_presets.cfgutil import virtual_agt_factory, virtual_mod_factory,virtual_part_factory,global_mod_cfg

from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_sdk.neko_framework_NG.libimfe.agents.unified_im_fe_agent import imfe_NG_sub
class neko_res45_backbone_mod_factory_abstract(virtual_mod_factory):
    MOD_PRFX_fe_core = "modprfx_global"
    bn_engine = neko_r45_norms_origNG;
    layers_engine = neko_r45_layers_origNG;
    bogo_fe_engine = neko_res45_bogo_g4_g2ish;
    KEY_fe_cores_registered = "fe_cores_registered";
    KEY_bns_registered = "bns_registered";
    def __init__(this,gmodcfg:global_mod_cfg,external_factory_dict=None):
        super().__init__(gmodcfg,external_factory_dict)
        this.fe_blk_cnt=[None, 3, 4, 6, 6, 3];
        this.fe_strides=[(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)];
        this.fe_relu_inplace=False;
        this.fe_bn_affine=gmodcfg.bn_affine;

    def config_fe_bn(this,modcfg,name):
        modcfg =this.config_saveable(modcfg,this.bn_engine,{
                "strides": this.fe_strides, # Keep the same with fe. And it is actually used to decide how many layers (bns) are in a block.
                "inpch": this.gmodcfg.img_ch,
                "blkcnt": this.fe_blk_cnt,
                "ochs": this.gmodcfg.fe_ochs,
                "affine": this.fe_bn_affine,
            },name);
        modcfg[this.KEY_global_registry][this.KEY_bns_registered][name] = True;
        return modcfg;
    def config_fe_convs(this,modcfg,name):
        modcfg = this.config_saveable(modcfg,this.layers_engine,{
                this.layers_engine.PARAM_in_channels: this.gmodcfg.img_ch,
                this.layers_engine.PARAM_block_counts: this.fe_blk_cnt,
                this.layers_engine.PARAM_out_channels: this.gmodcfg.fe_ochs,
                this.layers_engine.PARAM_strides: this.fe_strides,
                this.layers_engine.PARAM_inplace_relu: this.fe_relu_inplace
            },name);

        modcfg[this.KEY_global_registry][this.KEY_fe_cores_registered][name]=True;
        return modcfg;


class neko_res45_backbone_mod_factory(neko_res45_backbone_mod_factory_abstract):

    def config_fe_layers(this,modcfg,name):
        return this.config_fe_convs(modcfg,name);

    def config_dom_bogofe(this,bogocfg,name,conv_container,bn_container):
        bogocfg[name]={
            "bogo_mod": this.bogo_fe_engine,
            "args":
                {
                    "mod_cvt":
                        {
                            "conv": conv_container,
                            "norm": bn_container,
                        },
                }
        }
        return bogocfg;
    def config_fe(this,modcfg,bogocfg,layers_name,norm_name,bogo_fe_name):
        if(this.KEY_fe_cores_registered not in modcfg[this.KEY_global_registry]):
            modcfg[this.KEY_global_registry][this.KEY_fe_cores_registered]={};
            modcfg[this.KEY_global_registry][this.KEY_bns_registered] = {};

        if(layers_name not in modcfg[this.KEY_global_registry][this.KEY_fe_cores_registered]):
            modcfg=this.config_fe_layers(modcfg,layers_name);
        assert layers_name in  modcfg[this.KEY_global_registry][this.KEY_fe_cores_registered];

        if norm_name not in modcfg[this.KEY_global_registry][this.KEY_bns_registered]:
            modcfg=this.config_fe_bn(modcfg,norm_name);
        bogocfg=this.config_dom_bogofe(bogocfg,bogo_fe_name,layers_name,norm_name);
        return modcfg,bogocfg;



    def arm_module(this,mod_prfx_dict,modcfg,bogocfg,params=None):
    # well if everything looks normal, just arm everything...
        localprfx=mod_prfx_dict[this.MOD_PRFX_local];
        global_modprfx=mod_prfx_dict[this.MOD_PRFX_fe_core];
        modcfg,bogocfg=this.config_fe(modcfg,bogocfg,
            MN.FE_conv(global_modprfx),MN.FE_bn(localprfx),MN.FE(localprfx));
        return modcfg,bogocfg;
class neko_res45_nf_backbone_mod_factory(neko_res45_backbone_mod_factory):
    bogo_fe_engine = neko_res45_bogo_g4_g2ish_nf;


class neko_res45_backbone_agt_factory(virtual_agt_factory):
    def arm_fe_agt(this,agtcfg,modname,img_name,main_feat_name,feats_twr,feats_tag,agtname):
        agtcfg=awa.append_agent_to_cfg(agtcfg,agtname,
            imfe_NG_sub.get_agtcfg(img_name,main_feat_name,feats_twr,feats_tag,modname));
        return agtcfg;
    def arm_agt_core(this, mod_prfx_dict, data_prfx_dict,agtcfg,agt_prfx,params=None):
        localdataprfx=data_prfx_dict[this.DATA_PRFX_local];
        localmodprfx=mod_prfx_dict[this.MOD_PRFX_local];
        return this.arm_fe_agt(
            agtcfg,MN.FE(localmodprfx),
            VN.IM_normed_tensor(localdataprfx),
            VN.I_FEAT_MAP(localdataprfx),
            VN.FEAT_MAP_LST(localdataprfx),
            VN.FEAT_MAP_TAG(localdataprfx),
            AN.FE(agt_prfx));

# look, we do this bcs sometime we only want to change how modules are configured or agents (how mods are called), not both at the same time.
class neko_res45dsbn_fe_nf_part_factory(virtual_part_factory):
    def setup_factories(this):
        super().setup_factories();
        this.set_backbone_daemon_factories();

    # the backbone factory has to be globally held to keep track of shared backbone layers.
    # the backbone factory has to be globally held to keep track of shared backbone layers.
    def set_backbone_daemon_factories(this):
        this.backbone_modf = neko_res45_nf_backbone_mod_factory(this.gmodcfg, {});
        this.backbone_agtf = neko_res45_backbone_agt_factory();


    # core functionality: collated im to feature map
    # to start anew, use a unique share mod prfx.
    def arm_devten_ftwr_mods(this, modcfg, bogocfg, pipeline_mod_prefix, shared_mod_prfx):
        modcfg, bogocfg = this.backbone_modf.arm_module(
            {this.backbone_modf.MOD_PRFX_local: pipeline_mod_prefix,
             this.backbone_modf.MOD_PRFX_fe_core: shared_mod_prfx},
            modcfg, bogocfg, shared_mod_prfx);
        return modcfg, bogocfg;

    def arm_devten_ftwr_agts(this, acfg, data_prfx, mod_prfx):
        # well if there is re
        acfg = this.backbone_agtf.arm_agt_core(
            {this.backbone_agtf.MOD_PRFX_local: mod_prfx},
            {this.backbone_agtf.DATA_PRFX_local: data_prfx}, acfg, data_prfx);
        return acfg

