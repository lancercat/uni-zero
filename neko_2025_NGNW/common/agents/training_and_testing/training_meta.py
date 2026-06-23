from neko_2025_NGNW.common.object_32x_presets.deprecated.training_sampled_meta import training_sampled_meta_factory_single_noto_modality,training_sampled_meta_factory_randkim_modality

from neko_2025_NGNW.common.object_32x_presets.deprecated.training_base_meta_lo import training_data_meta_lo_node


class training_meta_factory:
    def arm_sampled_glyph_meta_agents(this, acfg, data_streams, data_template, meta_streams, meta_template, data_root,
                                      seed):
        for meta in meta_streams:
            for sprfx in meta_streams[meta]["sampled"]:
                # meta sampling and side info loading are done together ---- bcs this help us to modify tdict wrt modality---
                acfg = training_sampled_meta_factory_single_noto_modality.arm_agts(
                    acfg, {
                        training_sampled_meta_factory_single_noto_modality.META_PRFX: meta,
                        training_sampled_meta_factory_single_noto_modality.OUT_META_PRFX: sprfx,
                        training_sampled_meta_factory_single_noto_modality.SAMPLE_REF_DATA_PRFXS:
                            meta_streams[meta]["sampled"][sprfx]["sample_ref"]
                    }, {}, {
                        training_sampled_meta_factory_single_noto_modality.AGT_PRFX: sprfx
                    }, {
                        training_sampled_meta_factory_single_noto_modality.PARAM_SEED: seed,
                        training_sampled_meta_factory_single_noto_modality.PARAM_meta_template: meta_template,
                        training_sampled_meta_factory_single_noto_modality.PARAM_sinfo_path_noto:meta_streams[meta]["path"]
                    });
        return acfg;

    def arm_meta_agents(this, acfg, data_streams, data_template, meta_streams, meta_template, data_root, seed):
        for meta in meta_streams:
            acfg = training_data_meta_lo_node.arm_agts(
                acfg, {training_data_meta_lo_node.META_PRFX: meta}, {}, {training_data_meta_lo_node.AGT_PRFX: meta},
                {training_data_meta_lo_node.PARAM_meta_root: meta_streams[meta]["path"]}
            );
        acfg = this.arm_sampled_glyph_meta_agents(acfg, data_streams, data_template, meta_streams, meta_template,
                                                  data_root, seed);
        return acfg;
class training_meta_factory_fsl(training_meta_factory):
    def arm_sampled_glyph_meta_agents(this, acfg, data_streams, data_template, meta_streams, meta_template, data_root,
                                      seed):
        for meta in meta_streams:
            for sprfx in meta_streams[meta]["sampled"]:
                # meta sampling and side info loading are done together ---- bcs this help us to modify tdict wrt modality---
                acfg = training_sampled_meta_factory_randkim_modality.arm_agts(
                    acfg, {
                        training_sampled_meta_factory_randkim_modality.META_PRFX: meta,
                        training_sampled_meta_factory_randkim_modality.OUT_META_PRFX: sprfx,
                        training_sampled_meta_factory_randkim_modality.SAMPLE_REF_DATA_PRFXS:
                            meta_streams[meta]["sampled"][sprfx]["sample_ref"]
                    }, {}, {
                        training_sampled_meta_factory_randkim_modality.AGT_PRFX: sprfx
                    }, {
                        training_sampled_meta_factory_randkim_modality.PARAM_SEED: seed,
                        training_sampled_meta_factory_randkim_modality.PARAM_meta_template: meta_template,
                        training_sampled_meta_factory_randkim_modality.PARAM_sinfo_lmdb_root:
                            meta_streams[meta]["dbpath"],
                        training_sampled_meta_factory_randkim_modality.PARAM_k: this.k

                });
        return acfg;
    def __init__(this,k):
        this.k=k;


