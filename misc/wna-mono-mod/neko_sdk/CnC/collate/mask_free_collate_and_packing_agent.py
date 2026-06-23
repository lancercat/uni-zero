import torch

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.names import default_variable_names as dvn
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.cfgtool.argsparse import neko_get_arg


class neko_mask_free_collate_agent_np(neko_module_wrapping_agent):
    INPUT_raw_image_names = "input_" + dvn.raw_image_name;
    OUTPUT_raw_image_names = "output_" + dvn.raw_image_name;
    MOD_collator = "data_collator";
    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.input_images = this.register_input_list(this.INPUT_raw_image_names, iocvt_dict);
        this.output_images = this.register_output_list(this.OUTPUT_raw_image_names, iocvt_dict);
        this.collator = this.register(this.MOD_collator, modcvt_dict, this.mnames);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        for i in range(len(this.input_images)):
            iname = this.input_images[i];
            oname = this.output_images[i];
            inp = workspace.inter_dict[iname];
            workspace.add(oname, environment.module_dict[this.collator](inp));
        return workspace, environment;

    # the configuration will be mostly found in the class,
    @classmethod
    def config_me(cls, input_image_names, output_image_names, collate_mod_name):
        return {
            "agent": cls,
            "params": {
                "iocvt_dict": {
                    cls.INPUT_raw_image_names: input_image_names,
                    cls.OUTPUT_raw_image_names: output_image_names,
                },
                "modcvt_dict":
                    {
                        cls.MOD_collator: collate_mod_name,
                    }
            }
        }

# The size as not passed as a parameter, its in the module.
# the scope of mvn and collate is different, don't link them together!
# masks are NOT subjects of MVN! (but they are subjects of collate)
# class neko_mask_free_collate_agent(neko_mask_free_collate_agent_np):
#     PARAM_RAW_INP="raw_input";
#
#     def set_etc(this,param):
#         this.is_raw=neko_get_arg(this.PARAM_RAW_INP,param,True);
#     def take_action(this,workspace:neko_workspace,environment:neko_environment):
#         for i in range(len(this.input_images)):
#             iname = this.input_images[i];
#             oname = this.output_images[i];
#             if(this.is_raw):
#                 if(iname not in workspace.inter_dict):
#                     continue;
#                 inp= [torch.tensor(i).permute([2,0,1]).unsqueeze(0) for i in   workspace.inter_dict[iname]];
#             else:
#                 inp=  workspace.inter_dict[iname];
#             workspace.add(oname,environment.module_dict[this.collator](inp));
#         return workspace,environment;
#     # the configuration will be mostly found in the class,
#     @classmethod
#     def config_me(cls,input_image_names,output_image_names,collate_mod_name):
#         return {
#             "agent": cls,
#             "params": {
#                 "iocvt_dict": {
#                     cls.INPUT_raw_image_names: input_image_names,
#                     cls.OUTPUT_raw_image_names: output_image_names,
#                 },
#                 "modcvt_dict":
#                     {
#                         cls.MOD_collator: collate_mod_name,
#                     }
#             }
#         }
#

# The size as not passed as a parameter, its in the module.
# the scope of mvn and collate is different, don't link them together!
# masks are NOT subjects of MVN! (but they are subjects of collate)
import cv2
import numpy as np
class neko_mask_free_collate_agent(neko_mask_free_collate_agent_np):
    PARAM_RAW_INP="raw_input";

    def set_etc(this,param):
        this.is_raw=neko_get_arg(this.PARAM_RAW_INP,param,True);
    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        for i in range(len(this.input_images)):
            iname = this.input_images[i];
            oname = this.output_images[i];
            if(this.is_raw):
                if(iname not in workspace.inter_dict):
                    continue;
                inp= [torch.tensor(i).permute([2,0,1]).unsqueeze(0) for i in   workspace.inter_dict[iname]];
            else:
                inp=  workspace.inter_dict[iname];

            workspace.add(oname,environment.module_dict[this.collator](inp));
        return workspace,environment;
    # the configuration will be mostly found in the class,
    @classmethod
    def config_me(cls,input_image_names,output_image_names,collate_mod_name):
        return {
            "agent": cls,
            "params": {
                "iocvt_dict": {
                    cls.INPUT_raw_image_names: input_image_names,
                    cls.OUTPUT_raw_image_names: output_image_names,
                },
                "modcvt_dict":
                    {
                        cls.MOD_collator: collate_mod_name,
                    }
            }
        }

