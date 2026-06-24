from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.agents.saver_agent import neko_abstract_saving_agent
import os
import cv2

class neko_img_saver_agent(neko_abstract_saving_agent):
    INPUT_img_list="img_list";
    INPUT_raw_image="raw_image";
    def set_mod_io(this,iocvt_dict,modcvt_dict):
        this.imgs=this.register(this.INPUT_img_list,iocvt_dict,this.input_dict);
        this.raws=this.register(this.INPUT_raw_image,iocvt_dict,this.input_dict);


    def take_action(this,workspace:neko_workspace,environment:neko_environment):
        raws=workspace.inter_dict[this.raws];
        imgs=workspace.inter_dict[this.imgs];
        os.makedirs(os.path.join(this.export_path,this.testname),exist_ok=True);
        for i in range(len(imgs)):
            this.cntr += 1;
            if(raws[i].shape[0]<raws[i].shape[1]):
                cv2.imwrite(os.path.join(this.export_path,this.testname,this.prefix+"hori"+str(this.cntr)+".jpg"),imgs[i]);
            else:
                cv2.imwrite(os.path.join(this.export_path,this.testname, this.prefix +"vert"+str(this.cntr) + ".jpg"), imgs[i]);


def get_img_saving_agent(imgs,raws,export_root,prefx):
    return {
            "agent":neko_img_saver_agent,
            "params":{
                neko_img_saver_agent.PARAM_prefix:prefx,
                neko_img_saver_agent.PARAM_save_path:export_root,
                "iocvt_dict":{
                    neko_img_saver_agent.INPUT_img_list: imgs,
                    neko_img_saver_agent.INPUT_raw_image:raws,
                },
                "modcvt_dict": {

                }
            }
        };



class neko_text_saving_agent(neko_abstract_saving_agent):
    INPUT_text_list="text_list";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.texts = this.register(this.INPUT_text_list, iocvt_dict, this.input_dict);

    def take_action(this, workspace: neko_workspace, environment: neko_environment):

        os.makedirs(os.path.join(this.export_path,this.testname),exist_ok=True);
        try:
            txts = workspace.inter_dict[this.texts];
            for i in range(len(txts)):
                this.cntr += 1;
                with open(os.path.join(this.export_path,this.testname, this.prefix + str(this.cntr) + ".txt"), "w") as fp:
                    fp.writelines([txts[i]]);
        except:
            pass;
def get_text_saving_agent(texts,export_root,prefx):
    return {
            "agent":neko_text_saving_agent,
            "params":{
                neko_text_saving_agent.PARAM_prefix:prefx,
                neko_text_saving_agent.PARAM_save_path:export_root,
                "iocvt_dict": {
                    neko_text_saving_agent.INPUT_text_list: texts
                },
                "modcvt_dict": {

                }
            }
    };