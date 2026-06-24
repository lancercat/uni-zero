from osocrNG.modular_agents_ocrNG.fe_agents.fpn_like import get_fpn_like_fe
from osocrNG.modular_agents_ocrNG.se_agents.se_agent import get_simple_spatial_embedding
from neko_sdk.neko_framework_NG.agents.neko_detacher_agent import get_neko_list_detacher_agent
from neko_sdk.neko_framework_NG.UAE.neko_agent_wrapping_agent import neko_agent_wrapping_agent

def get_detached_se_fpn(backbone_feat,backbone_feat_detached,feat_se,feat_out,endpoints_out,semod,fpnmod):
    dc=get_neko_list_detacher_agent([backbone_feat],[backbone_feat_detached]);
    sc=get_simple_spatial_embedding(backbone_feat_detached,feat_se,semod);
    fec=get_fpn_like_fe(feat_se,endpoints_out,feat_out,fpnmod);
    return {
        "agent": neko_agent_wrapping_agent,
        "params": {
            "agent_list": ["detach", "spatial_embedding", "fpn"],
            "detach":dc,
            "spatial_embedding": sc,
            "fpn": fec
        }
    };
def get_attached_se_fpn(backbone_feat,feat_se,feat_out,endpoints_out,semod,fpnmod):
    sc=get_simple_spatial_embedding(backbone_feat,feat_se,semod);
    fec=get_fpn_like_fe(feat_se,endpoints_out,feat_out,fpnmod);
    return {
        "agent": neko_agent_wrapping_agent,
        "params": {
            "agent_list": [ "spatial_embedding", "fpn"],
            "spatial_embedding": sc,
            "fpn": fec
        }
    };

# since v6 detaching is not handled by se agents.
def get_se_fpn(backbone_feat,feat_se,feat_out,endpoints_out,semod,fpnmod):
    sc=get_simple_spatial_embedding(backbone_feat,feat_se,semod);
    fec=get_fpn_like_fe(feat_se,endpoints_out,feat_out,fpnmod);
    return {
        "agent": neko_agent_wrapping_agent,
        "params": {
            "agent_list": [ "spatial_embedding", "fpn"],
            "spatial_embedding": sc,
            "fpn": fec
        }
    };