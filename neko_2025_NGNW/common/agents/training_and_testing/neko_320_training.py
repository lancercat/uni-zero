# Standard Library and Third-Party
import time

import tqdm

# Project Configuration and Utilities
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.cfgtool.platform_cfg import neko_platform_cfg
from neko_2025_NGNW.common.object_32x_presets.var_names import project_32x_varnames as VN

# Base Framework Components (neko_sdk.neko_framework_NG)
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_abstract_sync_agent as sa
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent as awa
from neko_sdk.neko_framework_NG.neko_module_setNG import neko_module_opt_setNG
from neko_sdk.log import warn,info,fatal

# Task/Data Arming Functions (osocrNG)

# NGNW Components (neko_2025_NGNW)
# from neko_2025_NGNW.common.agents.training_and_testing.deprecated.training_task_agent import neko_training_loader_agent_noaug
from neko_sdk.die import die

import torch
import torchvision.utils as vutils
import cv2
import numpy as np


def debug_tensor_img(ten, mean=127, std=127, name="debug"):
    """
    Args:
        ten: Tensor of shape [N, 3, H, W]
        mean: The value used to center the data (subtracted during prep)
        std: The value used to scale the data (divided during prep)
        name: Window name
    """
    with torch.no_grad():
        # 1. Clone and Denormalize: x = (x * std) + mean
        # We work on a copy to avoid corrupting the original tensor gradients
        view_ten = ten.detach().clone().float()
        view_ten = view_ten * std + mean

        # 2. Clamp to valid RGB range and cast to byte
        view_ten = torch.clamp(view_ten, 0, 255).byte()

        # 3. Calculate grid layout
        # To respect the 1024px max width, we calculate how many items fit per row
        img_w = view_ten.shape[3]
        num_images = view_ten.shape[0]
        nrow = max(1, 1024 // img_w)

        # 4. Create the tile grid [3, Total_H, Total_W]
        grid = vutils.make_grid(view_ten, nrow=nrow, padding=2)

        # 5. Convert to Numpy HWC for OpenCV
        ndarr = grid.permute(1, 2, 0).cpu().numpy()

        # 6. RGB to BGR for OpenCV display
        ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
        cv2.namedWindow(name, 0)

        cv2.imshow(name, ndarr)
        cv2.waitKey(0)  # Use 0 to pause execution, 1 for real-time flow

def debuginf(this, workspace):
    mprfx = "meta-chs_en_digit-sampled-glyph-armed";
    trtprfx = "training-task-ctw-chs_en_digit-real";
    tprfx = 'task-ctw-chs_en_digit';
    featvecs = workspace.get('data-ctw-abiaug-roi_feat_sequential');
    protos = workspace.get(VN.PROTO(mprfx));

    cls_logi_wounk = featvecs.squeeze(1).squeeze(1).matmul(protos.squeeze(1).T);
    tdict = workspace.get(VN.TDICT(mprfx));
    plabel = workspace.get(VN.PROTO_LABEL(mprfx))
    tlabel = workspace.get(VN.FLATTEN_ALIGNED_TLABEL(trtprfx))
    train_tar_token = workspace.get(VN.GT_TOK_UTF_WUNK(trtprfx));
    # logit_cntr=wsl.get(VN.DENSE_CENTER_PRED_LOGIT(tprfx));
    pred_force = [tdict[plabel[i.item()].item()] for i in cls_logi_wounk.argmax(-1)]
    real_logi = workspace.get('training-task-ctw-chs_en_digit-real-flatten_roi_cls_logit_seq');
    # real_logit_grad = this.modset.real_modulars['per_inst_cls_loss'].model.fakelogit.grad;
    # gt_logit_grad=[];
    # for i in range(real_logit_grad.shape[0]):
    #     gt_logit_grad.append(real_logit_grad[i][tlabel[i].item()].item());
    glyphten = workspace.get('meta-chs_en_digit-sampled-glyph-images_normed_tensors');
    glyphfeats = workspace.get('meta-chs_en_digit-sampled-glyph-feature_map_list');

    samten = workspace.get('data-ctw-abiaug-images_normed_tensors');
    samfeats = workspace.get('data-ctw-abiaug-feature_map_list');

    # so we can release wsl when needed so that this works on smaller gpus--- ideally 20Gib ones
    mmap = workspace.get('meta-chs_en_digit-sampled-glyph-attentionmap');
    fmap = workspace.get('data-ctw-abiaug-attentionmap');
    mim = workspace.get('meta-chs_en_digit-sampled-glyph-images_normed_tensors');
    sim = workspace.get('data-ctw-abiaug-images_normed_tensors');


class neko_training_agent_320(sa):
    PARAM_VITR="virtual_itr";
    PARAM_EPOC="epoch";
    PARAM_PCFG="platform_cfg";
    PARAM_FPBP_ACFG_DIC="fpbp_dict";
    PARAM_CHK_EACH="check_each";
    PARAM_SAVE_EACH="save_each";
    PARAM_TEAGTS="testing_agents";
    PARAM_MODSET="modset";
    PARAM_DEVICES="devices";
    PARAM_TR_LDR_ACFG="training_loader_agent";
    def test_skip_saving(this,eid,iid):
        if (eid == 0 and iid == 0):
            return True; # do not save init
        if(eid == 0):
            if(iid <60000):
                return False; # save early models for debugging
            return True; # do not save E0
        return False; # these are for ablative
    def __init__(this,param):
        super().__init__(param);
        this.pcfg:neko_platform_cfg=neko_get_arg(this.PARAM_PCFG,param);
        fpbpd=neko_get_arg(this.PARAM_FPBP_ACFG_DIC,param);
        this.fpbp_agt_dict= {k:awa.make(fpbpd[k]) for k in fpbpd}; # actually an awa, but maybe we can hack it so after bp it drops the workspace
        this.vitr=neko_get_arg(this.PARAM_VITR,param,200000);
        this.vepoc=neko_get_arg(this.PARAM_EPOC,param,5);
        this.check_each=neko_get_arg(this.PARAM_CHK_EACH,param,10000); # now check each has been separated with test each--- for ablative we still do a trajectory average--- but that does not prevent us keeping more testing iters
        this.save_each = neko_get_arg(this.PARAM_SAVE_EACH, param,
                                       20000);  # now check each has been separated with test each--- for ablative we still do a trajectory average--- but that does not prevent us keeping more testing iters

        this.ats = [teacfg["agent"](teacfg["params"]) for teacfg in neko_get_arg(this.PARAM_TEAGTS,param)];
        this.modset:neko_module_opt_setNG=neko_get_arg(this.PARAM_MODSET,param);
        this.devices=neko_get_arg(this.PARAM_DEVICES,param,["cuda:0"]);
        this.training_loader=neko_get_arg(this.PARAM_TR_LDR_ACFG,param);
    # before testing and training, the user is responsible to move modset so that it will work with the registered placement (this.devices)
    def test(this,eid=9,bid=9):
        this.modset.eval_mode();
        rdicts=[];
        for tt in this.ats:
            e=neko_environment(modset=this.modset);
            e.batch_idx=bid;
            e.epoch_idx=eid;
            ws=neko_workspace(None,eid,bid,this.devices);
            tt.take_action(ws,e);
            rdicts.append(ws.logdict);
            pass;

        this.modset.train_mode();
        return rdicts;

    def fpbpopt(this,ws,e,debug):
        for k in this.fpbp_agt_dict:
            wsl = ws.shallow_fork();
            this.fpbp_agt_dict[k].take_action(wsl, e);
            if (debug):
                print(wsl.logdict);
                # if('meta-cub-200-tr-sampled-imshot-images_normed_tensors' in wsl.inter_dict):
                #     debug_tensor_img(wsl.get('meta-cub-200-tr-sampled-imshot-images_normed_tensors'))
                # if(wsl.has('cub-200-abiaug-images_normed_tensors')):
                #     debug_tensor_img(wsl.get('cub-200-abiaug-images_normed_tensors'))
                # if(wsl.has('cub-200-noaug-images_normed_tensors')):
                #     debug_tensor_img(wsl.get('cub-200-noaug-images_normed_tensors'))
                # #
                # # debug only
                # this.debuginf(wsl)
        # if (debug):
        #     this.modset.debug_grad();
        this.modset.norm_grad();
        this.modset.update_d([]);
        this.modset.zero_grad();
        pass;
        # if (not valid):
        #     panic(wsl, e);
        #     fatal(badbois);
        # assert (workspace.inter_dict['training-task-ctw-chs_en_digit-real-tensor_label_aligned'].min() > 1);
    def trainitr(this,debug=True):
        trda=awa.make(this.training_loader);
        # this.test(0,0);
        # time.sleep(3);
        e=neko_environment(modset=this.modset);
        this.modset.train_mode()
        ws = neko_workspace(epoch_idx=0, batch_idx=0, devices=this.devices);
        # ws=ows.shallow_fork();
        info("have dl, commencing data fetch");
        trda.take_action(ws, e);
        info("have data, commencing fp");
        for k in this.fpbp_agt_dict:
            wsl = ws.shallow_fork();
            info("forked data, commencing fp");
            this.fpbp_agt_dict[k].take_action(wsl, e);
            info("done fpbp data, commencing opt");
                #
                # # debug only
                # this.debuginf(wsl)
        # if (debug):
        #     this.modset.debug_grad();
        info("done fpbp data, commencing gradnorm");

        this.modset.norm_grad();
        info("done norm, commencing upd");
        this.modset.update_d([]);
        info("done norm, commencing zerograd");
        this.modset.zero_grad();
        exit(9);

        for eid in range(this.vepoc):
            e.epoch_idx = eid;
            for bid in tqdm.tqdm(range(this.vitr)):
                e.batch_idx=bid;
                # print("CURR at", eid, bid);
                if (bid > 0):
                    if (bid % this.check_each == 0):
                        this.test(eid, bid);
                    if (bid % this.save_each == 0):
                        this.modset.save_as(this.pcfg.save_root, eid, bid);
                ws = neko_workspace(epoch_idx=eid, batch_idx=bid, devices=this.devices);
                # ws=ows.shallow_fork();
                trda.take_action(ws, e);
                this.fpbpopt(ws,e,debug);

            e.modset.update_opt(this.vepoc);
            this.test(eid+1, 0);
            this.modset.save_as_necessary(this.pcfg.save_root, eid+1, 0);
            pass;
    def train(this,debug=False):
        trda=awa.make(this.training_loader);
        # this.test(0,0);
        e=neko_environment(modset=this.modset);
        this.modset.train_mode()
        for eid in range(this.vepoc):
            e.epoch_idx = eid;

            for bid in tqdm.tqdm(range(this.vitr)):
                e.batch_idx=bid;
                # print("CURR at", eid, bid);
                if (bid > 0):
                    if (bid % this.check_each == 0):
                        this.test(eid, bid);
                    if (bid % this.save_each == 0):
                        this.modset.save_as(this.pcfg.save_root, eid, bid);
                ws = neko_workspace(epoch_idx=eid, batch_idx=bid, devices=this.devices);
                # ws=ows.shallow_fork();
                # t=time.time();
                trda.take_action(ws, e);
                # tt=time.time()-t;
                # print("data",tt);
                # t=time.time();
                this.fpbpopt(ws,e,debug);
                if (debug):
                    this.test(eid, 0);
                # tt=time.time()-t;
                # print("fpbpopt",tt);
            e.modset.update_opt(this.vepoc);
            this.test(eid+1, 0);
            this.modset.save_as_necessary(this.pcfg.save_root, eid+1, 0);
            pass;
    def train_debugdata(this,debug=False):
        trda=awa.make(this.training_loader);
        # this.test(0,0);
        e=neko_environment(modset=this.modset);
        this.modset.train_mode()
        for eid in range(this.vepoc):
            e.epoch_idx = eid;
            for bid in tqdm.tqdm(range(this.vitr)):
                e.batch_idx=bid;
                info("CURR at", eid, bid);
                if (bid > 0):
                    if (bid % this.check_each == 0):
                        this.test(eid, bid);
                    if (bid % this.save_each == 0):
                        this.modset.save_as(this.pcfg.save_root, eid, bid);
                ws = neko_workspace(epoch_idx=eid, batch_idx=bid, devices=this.devices);
                # ws=ows.shallow_fork();

                info("Getting DATA at", eid, bid);
                trda.take_action(ws, e);
                info("Got DATA at", eid, bid);

                # this.fpbpopt(ws,e,debug);
            e.modset.update_opt(this.vepoc);
            this.test(eid+1, 0);
            this.modset.save_as_necessary(this.pcfg.save_root, eid+1, 0);
            pass;
