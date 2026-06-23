import random

import numpy as np

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.lsctNG.libtrdg.libdistort import neko_trdg_mask_deformer,neko_trdg_mask_deformer_wins
from neko_sdk.cfgtool.argsparse import neko_get_arg
from multiprocess.pool import Pool

# apply trdg mask generation for any text mask...

class neko_trdg_mask_deformer_agent(neko_module_wrapping_agent):
    INPUT_mask_image_pil="maskim";
    INPUT_orientation="orientation";
    OUTPUT_deformed_maskim_np="deformed_maskim_np";
    PARAM_size="size";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.mask_image_pil = this.register_input(this.INPUT_mask_image_pil, iocvt_dict);
        this.mask_orientation = this.register_input(this.INPUT_orientation, iocvt_dict);

        this.deformed_maskim_np = this.register_output(this.OUTPUT_deformed_maskim_np, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.size = neko_get_arg(this.PARAM_size, params);
        this.defmod=neko_trdg_mask_deformer({neko_trdg_mask_deformer_agent.PARAM_size:this.size},random.Random(9));
        pass;
    def drive_stub(this,i,o):
        if(i is None):
            return None;
        return np.array(this.defmod.drive(i,o))[:,:,0:1];
    def drive(this,mask_image_pil,mask_image_pil_orient):
        return [this.drive_stub(i,o) for i,o in zip(mask_image_pil,mask_image_pil_orient)];

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        mask_image_pil = workspace.get(this.mask_image_pil);
        mask_image_pil_orient = workspace.get(this.mask_orientation);
        workspace.add(this.deformed_maskim_np,this.drive(mask_image_pil,mask_image_pil_orient));
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   mask_image_pil, orientation,
                   deformed_maskim_np,
                   size
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_mask_image_pil: mask_image_pil, cls.INPUT_orientation: orientation,
                           cls.OUTPUT_deformed_maskim_np: deformed_maskim_np}, cls.PARAM_size: size, "modcvt_dict": {}}}

class neko_trdg_mask_deformer_agent_parallel(neko_trdg_mask_deformer_agent):
    def set_etc(this, params):
        super().set_etc(params);
        this.pool=Pool(neko_get_arg("workers",params,12));

    def drive(this, mask_image_pil, mask_image_pil_orient):
        return this.defmod.drive_as_np_inparallel(mask_image_pil,mask_image_pil_orient,this.pool);

class neko_trdg_mask_deformer_agent_insmsk(neko_module_wrapping_agent):
    OUTPUT_deformed_maskins_np="out_mask_ins";
    INPUT_mask_instance_pil="in_mask_ins";
    INPUT_mask_image_pil="maskim";
    INPUT_mask_image_pil_orient="mask_image_pil_orient";
    OUTPUT_deformed_maskim_np="deformed_maskim_np";
    PARAM_size="size";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.mask_image_pil = this.register_input(this.INPUT_mask_image_pil, iocvt_dict);
        this.mask_instance_pil = this.register_input(this.INPUT_mask_instance_pil, iocvt_dict);
        this.mask_image_pil_orient = this.register_input(this.INPUT_mask_image_pil_orient, iocvt_dict);
        this.deformed_maskim_np = this.register_output(this.OUTPUT_deformed_maskim_np, iocvt_dict);
        this.deformed_maskins_np = this.register_output(this.OUTPUT_deformed_maskins_np, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.size = neko_get_arg(this.PARAM_size, params);
        this.defmod=neko_trdg_mask_deformer_wins({neko_trdg_mask_deformer_agent.PARAM_size:this.size},random.Random(9));

        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        mask_image_pil = workspace.get(this.mask_image_pil);
        mask_instance_pil = workspace.get(this.mask_instance_pil);
        mask_image_pil_orient = workspace.get(this.mask_image_pil_orient);
        defed=[];
        defed_ins=[];
        for m, i, o in zip(mask_image_pil,mask_instance_pil, mask_image_pil_orient):
            if (m is not None):
                gm,insms=this.defmod.drive(m, i, o);
                gm=np.array(gm)[:,:, 0:1];
                insms=[np.array(inm)[:,:, 0:1] for inm in insms];
            else:
                gm,insms=None,None;

            defed.append(gm);
            defed_ins.append(insms);
        workspace.add(this.deformed_maskins_np, defed_ins);

        workspace.add(this.deformed_maskim_np, defed);
        return workspace, environment;

    @classmethod
    def get_agtcfg(cls,
                   mask_image_pil, mask_instance_pil, mask_image_pil_orient,
                   deformed_maskim_np, deformed_maskins_np,
                   size
                   ):
        return {"agent": cls, "params": {
            "iocvt_dict": {cls.INPUT_mask_image_pil: mask_image_pil, cls.INPUT_mask_instance_pil: mask_instance_pil,
                           cls.INPUT_mask_image_pil_orient: mask_image_pil_orient, cls.OUTPUT_deformed_maskim_np: deformed_maskim_np,
                           cls.OUTPUT_deformed_maskins_np: deformed_maskins_np}, cls.PARAM_size: size,
            "modcvt_dict": {}}}

if __name__ == '__main__':
    neko_trdg_mask_deformer_agent_insmsk.print_default_setup_scripts();
