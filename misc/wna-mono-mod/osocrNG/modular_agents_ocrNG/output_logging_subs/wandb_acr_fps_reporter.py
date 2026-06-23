
import wandb
from neko_sdk.cfgtool.argsparse import neko_get_arg
from osocrNG.modular_agents_ocrNG.output_logging_subs.acr_fps_reporter_wandb_hi import reporter_core_wandb

from osocrNG.modular_agents_ocrNG.output_logging_subs.basic_acr_fps_reporter import case_inv_acr_fps_reporter,case_inv_multi_orientation_acr_fps_reporter
class case_inv_acr_fps_reporter_wandb(case_inv_acr_fps_reporter):
    PARAM_wandb_run="wandb_run";
    def set_etc(this,param):
        super().set_etc(param);
        this.run=neko_get_arg(this.PARAM_wandb_run,param,wandb.run);
        this.recorder=reporter_core_wandb(this.run);# due to historical reasons, symbols in some testing set are not annotated, and such symbols will be recognized as unknown

class case_inv_multi_orientation_acr_fps_reporter_wandb(case_inv_multi_orientation_acr_fps_reporter):
    PARAM_wandb_run="wandb_run";
    def set_etc(this,param):
        this.stat_range_dict=neko_get_arg(
            this.PARAM_stat_range_dict,param,{"horizontal":{"ratio":(1.0,9999)},"vertical":{"ratio":(-9,1.0)},"all":{"ratio":(-9,9999)}}
        );
        this.recorders={};
        this.run=neko_get_arg(this.PARAM_wandb_run,param,wandb.run);

        for k in this.stat_range_dict:
            this.recorders[k]=reporter_core_wandb(this.run);

