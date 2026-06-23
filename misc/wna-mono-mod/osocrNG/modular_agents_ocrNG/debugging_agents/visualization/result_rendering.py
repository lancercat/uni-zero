
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755
from neko_sdk.ocr_modules.charset.etc_cset import latin62
from neko_sdk.ocr_modules.element_renderer import render_gt,render_pred
class neko_result_rendering_agent(neko_module_wrapping_agent):
    INPUT_pred_text = "pred_text";
    INPUT_text_label = "text_label";
    INPUT_tdict = "tdict";
    INPUT_plabel = "plabel";

    INPUT_tensor_proto_img="tensor_proto_img";
    PARAM_training_chars="training_chars";

    OUTPUT_all_patches="gt_patches";
    def fake_mdict(this,proto_img,tdict,plabel):
        pd={};
        apl=plabel.cpu().numpy();
        cti=((proto_img+1)*127).cpu();
        for i in range(len(cti)):
            k=apl[i];
            if (k in pd):
                pass;
            else:
                pd[k]=cti[i:i+1];
        return {
            "protos":pd,
            "label_dict":tdict,
        }


    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.plabel = this.register_input(this.INPUT_plabel, iocvt_dict);
        this.pred_text = this.register_input(this.INPUT_pred_text, iocvt_dict);
        this.tdict = this.register_input(this.INPUT_tdict, iocvt_dict);
        this.text_label = this.register_input(this.INPUT_text_label, iocvt_dict);
        this.tensor_proto_img = this.register_input(this.INPUT_tensor_proto_img, iocvt_dict);

        this.all_patches = this.register_output(this.OUTPUT_all_patches, iocvt_dict);

        pass;

    def set_etc(this, params):
        this.training_chars=neko_get_arg(this.PARAM_training_chars,params,t1_3755.union(latin62));

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        tdict = workspace.get(this.tdict);
        plabel = workspace.get(this.plabel);
        pred_text = workspace.get(this.pred_text);
        tensor_proto_img=workspace.get(this.tensor_proto_img);
        mdict= this.fake_mdict(tensor_proto_img,tdict,plabel);
        text_label=None
        if (this.text_label is not None):
            if (this.text_label in workspace.inter_dict):
                try:
                    text_label = workspace.get(this.text_label);
                except:
                    pass;
            else:
                pass;
        aaps=[];
        for i in range(len(pred_text)):
            aps = [];
            if text_label is not None:
                aps.append(render_gt(mdict, this.training_chars, text_label[i]));
                aps.append(render_pred(mdict, text_label[i], pred_text[i]));
            else:
                aps.append(render_pred(mdict, None,pred_text[i] ));
            aaps.append(aps);
        workspace.add(this.all_patches,aaps);
        return workspace, environment;
def get_neko_result_rendering_agent(
    plabel,pred_text,tdict,tensor_proto_img,text_label,
    all_patches,
    training_chars
):
    engine = neko_result_rendering_agent;return {"agent": engine, "params": {"iocvt_dict": {engine.INPUT_plabel: plabel, engine.INPUT_pred_text: pred_text, engine.INPUT_tdict: tdict, engine.INPUT_tensor_proto_img: tensor_proto_img, engine.INPUT_text_label: text_label, engine.OUTPUT_all_patches: all_patches}, engine.PARAM_training_chars: training_chars, "modcvt_dict": {}}}
