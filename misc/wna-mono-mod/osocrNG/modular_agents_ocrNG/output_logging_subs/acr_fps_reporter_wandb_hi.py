import datetime
import os.path
import time

from wandb import wandb_sdk

from neko_sdk.ocr_modules.sptokens import tUNKREP
from osocrNG.modular_agents_ocrNG.output_logging_subs.acr_fps_reporter_hi import reporter_core
class reporter_core_wandb(reporter_core):
    def report(this,eidx,bidx):
        rd={
            os.path.join(this.name,"Epoch"):eidx,
            os.path.join(this.name, "Iter"): bidx,
            os.path.join(this.name, "Total"): this.tot,
            os.path.join(this.name, "ACR"): this.corr / this.tot,
            os.path.join(this.name, "Lenpred_ACR"): this.lcorr / this.tot,
            os.path.join(this.name, "FPS"): this.tot / time.time() - this.start_time,

        }
        this.run.log(rd);
        super().report(eidx,bidx);

    def __init__(this,run:wandb_sdk.wandb_run.Run):
        this.run=run
        pass;
