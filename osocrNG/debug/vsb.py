import numpy as np;
def debug_vsb(workspace):
    GT,VPR,SPR=workspace.inter_dict['hori_raw_label'],workspace.inter_dict['hori_main_mvb_test_detached_pred_dict'],workspace.inter_dict['hori_main_ducky_pred_text'];
    AGG=[SPR[i]==VPR[i] for i in range(len(GT))];
    AGGIDX=np.where(AGG);
    AGGPRED=[VPR[i] for i in AGGIDX[0]];
    AGGGT=[GT[i] for i in AGGIDX[0]];
    return {
    "VPR":VPR,
    "GT":GT,
    "SPR":SPR,
    "ACC_SPR":np.mean([SPR[i]==GT[i] for i in range(len(GT))]),
    "ACC_VPR":np.mean([VPR[i]==GT[i] for i in range(len(GT))]),
    "AGG":np.mean(AGG),
    "AGG_ACR":np.mean([AGGPRED[i]==AGGGT[i] for i in range(len(AGGGT))]),
    "AGGIDX":AGGIDX,
    "AGGPRED":AGGPRED,
    "AGGGT":AGGGT,
    }
